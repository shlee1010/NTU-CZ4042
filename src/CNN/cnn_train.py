import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import itertools
import os
import copy
from tqdm import tqdm
import tempfile

from dataset import SpectrogramDataset
from cnn_model import EfficientNetClassifier

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.best_model_state = None

    def check(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    loader_tqdm = tqdm(loader, desc="Training", leave=False)
    for images, labels in loader_tqdm:
        images, labels = images.to(device), labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    loader_tqdm = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, labels in loader_tqdm:
            images, labels = images.to(device), labels.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def train_final_model(best_params, config):
    print("\n" + "="*50)
    print("Phase 1: final model training(all train1 data use)")
    print(f"best HP: {best_params}")
    print("="*50)

    device = torch.device(config["device"])
    
    full_train_dataset = SpectrogramDataset(
        csv_file=config["data_csv"],
        img_dir=config["train_img_dir"]
    )
    train_loader = DataLoader(
        full_train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    
    model = EfficientNetClassifier(
        num_classes=config["num_classes"],
        dropout_rate=best_params['dropout_rate']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    optimal_epochs = best_params.get('avg_epochs', config['num_epochs'])
    print(f"final training start... (total {optimal_epochs} Epochs - CV 평균)")

    for epoch in range(optimal_epochs):
        avg_loss, avg_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Final Train Epoch {epoch+1}/{optimal_epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}")

    save_path = os.path.join(config['model_save_dir'], "best_cnn_phase1.pth")
    try:
        torch.save(model.state_dict(), save_path)
        print(f"\n Phase 1 final model save completed: {save_path}")
    except Exception as e:
        print(f"\n modle save failed: {e}")
        
    config_path = os.path.join(config['model_save_dir'], "best_cnn_config.json")
    try:
        import json
        with open(config_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Best HP save completed: {config_path}")
    except Exception as e:
        print(f"\nconfig save failed: {e}")

def main():
    CONFIG = {
        "data_csv": "/home/UG/n2402401d/SC4001_Project/FINAL/data/tracks_train1.csv",
        "test_csv": "/home/UG/n2402401d/SC4001_Project/FINAL/data/tracks_test.csv",
        "train_img_dir": "/home/UG/n2402401d/SC4001_Project/FINAL/data/spectrograms_train1",
        "test_img_dir": "/home/UG/n2402401d/SC4001_Project/FINAL/data/spectrograms_test",
        "results_csv": "/home/UG/n2402401d/SC4001_Project/FINAL/cnn_cv_results.csv",
        "model_save_dir": "/home/UG/n2402401d/SC4001_Project/FINAL/cnn_cv",
        "num_classes": 5,
        "k_folds": 3,
        "num_epochs": 50,
        "num_workers": 4,
        "patience": 5,
        "min_delta": 0.001,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    param_grid = {
        'learning_rate': [1e-3, 5e-4, 1e-4],
        'batch_size': [64],
        'dropout_rate': [0.3, 0.4, 0.5]
    }
    
    print(f"--- CNN HP tuning start (K={CONFIG['k_folds']}) ---")
    print(f"Device: {CONFIG['device']}")
    
    os.makedirs(CONFIG['model_save_dir'], exist_ok=True)
    
    df_full = pd.read_csv(CONFIG["data_csv"])
    print(f"Total data: {len(df_full)} samples")
    
    kf = KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    
    all_results = []
    best_avg_accuracy = -1.0
    best_params = {}

    param_combinations = list(itertools.product(
        param_grid['learning_rate'],
        param_grid['batch_size'],
        param_grid['dropout_rate']
    ))
    
    print(f"Total {len(param_combinations)} combination test.")
    
    for combo_idx, (lr, batch_size, dropout) in enumerate(param_combinations):
        
        params_key = f"lr={lr}, batch={batch_size}, dropout={dropout}"
        print("\n" + "="*50)
        print(f"Combination {combo_idx+1}/{len(param_combinations)}: {params_key}")
        print("="*50)
        
        fold_accuracies = []
        fold_epochs = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(df_full)):
            print(f"--- Fold {fold+1}/{CONFIG['k_folds']} ---")
            
            df_train = df_full.iloc[train_idx].reset_index(drop=True)
            df_val = df_full.iloc[val_idx].reset_index(drop=True)
            
            temp_dir = tempfile.mkdtemp()
            train_csv = os.path.join(temp_dir, f'train_fold{fold}.csv')
            val_csv = os.path.join(temp_dir, f'val_fold{fold}.csv')
            
            df_train.to_csv(train_csv, index=False)
            df_val.to_csv(val_csv, index=False)
            
            train_dataset = SpectrogramDataset(train_csv, CONFIG["train_img_dir"])
            val_dataset = SpectrogramDataset(val_csv, CONFIG["train_img_dir"])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=CONFIG["num_workers"],
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=CONFIG["num_workers"],
                pin_memory=True
            )
            
            print(f"  Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
            
            model = EfficientNetClassifier(
                num_classes=CONFIG['num_classes'],
                dropout_rate=dropout
            ).to(CONFIG['device'])
            
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            early_stopper = EarlyStopper(
                patience=CONFIG['patience'], 
                min_delta=CONFIG['min_delta']
            )

            stopped_epoch = CONFIG['num_epochs']


            for epoch in range(CONFIG['num_epochs']):
                
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
                val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, CONFIG['device'])
                
                print(f"  Epoch {epoch+1}/{CONFIG['num_epochs']}: Train Acc: {train_acc:.4f} | Valid Acc: {val_acc:.4f}")
                
                if early_stopper.check(val_loss, model):
                    stopped_epoch = epoch + 1
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            if early_stopper.best_model_state:
                model.load_state_dict(early_stopper.best_model_state)
            
            _, final_val_acc = validate_one_epoch(model, val_loader, criterion, CONFIG['device'])
            print(f"  Fold {fold+1} Best Valid Accuracy: {final_val_acc:.4f}")
            fold_accuracies.append(final_val_acc)
            fold_epochs.append(stopped_epoch)
        
        avg_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        avg_epochs = int(np.mean(fold_epochs))
        
        print(f"--- Combination {params_key} result ---")
        print(f"  Average Valid Accuracy: {avg_acc:.4f} (+/- {std_acc:.4f})")
        print(f"  Average Epochs: {avg_epochs}")

        run_data = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'dropout_rate': dropout,
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'avg_epochs': avg_epochs
        }
        all_results.append(run_data)
        
        if avg_acc > best_avg_accuracy:
            best_avg_accuracy = avg_acc
            best_params = run_data
            print(f"New best combination find")

    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING FINISHED")
    print("="*50)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by='avg_accuracy', ascending=False)
    
    try:
        results_df.to_csv(CONFIG['results_csv'], index=False)
        print(f"Tuning result save completed: {CONFIG['results_csv']}")
    except Exception as e:
        print(f"Result save failed: {e}")
    
    print("\n--- Best HP combination ---")
    print(results_df.head(1))
    print(f"Best Params: {best_params}")
    print(f"Best Average Accuracy: {best_avg_accuracy:.4f}")

    if best_avg_accuracy > -1.0:
        train_final_model(best_params, config=CONFIG)
    else:
        print("Skipped final training - tuning failed or can not find valid result.")

if __name__ == "__main__":
    main()