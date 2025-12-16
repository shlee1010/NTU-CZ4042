import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import itertools
import tempfile
from cnn_model import EfficientNetClassifier



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_width, hidden_depth, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_width))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_width))
        layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(hidden_depth - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_width))
            layers.append(nn.Dropout(dropout_rate))
            
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_width, output_dim)

    def forward(self, x):
        x = self.network(x)
        x = self.output_layer(x)
        return x



def load_and_freeze_mlp(model_path):

    print(f"Loading state dict from: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    state_dict = torch.load(model_path, map_location=device)

    try:
        input_dim = state_dict['network.0.weight'].shape[1]
        hidden_width = state_dict['network.0.weight'].shape[0]
        original_output_dim = state_dict['output_layer.weight'].shape[0]
        dropout_rate = 0.4 
        
        max_linear_index = max(
            int(k.split('.')[1]) for k in state_dict 
            if k.startswith('network.') and k.endswith('.weight') and (int(k.split('.')[1]) % 4 == 0)
        )
        hidden_depth = (max_linear_index // 4) + 1

    except Exception as e:
        print(f"Error: Could not infer model architecture from state_dict keys. {e}")
        print("Please ensure the MLP class definition matches the saved model.")
        return None

    print(f"Inferred Original Params: input_dim={input_dim}, hidden_width={hidden_width}, hidden_depth={hidden_depth}")
    print(f"Original output dim (genres): {original_output_dim}")

    model = MLP(
        input_dim=input_dim,
        output_dim=original_output_dim,
        hidden_width=hidden_width,
        hidden_depth=hidden_depth,
        dropout_rate=dropout_rate
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    print("Successfully loaded pre-trained weights into model.")

    print("Freezing all pre-trained layer weights (setting requires_grad=False)...")
    for param in model.parameters():
        param.requires_grad = False

    print("Model loaded and frozen.")
    return model

def load_and_freeze_cnn(model_path):
    """
    Loads the saved EfficientNet classification model and freezes all its weights.
    """
    print(f"Loading state dict from: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    state_dict = torch.load(model_path, map_location=device)

    try:
        original_num_classes = state_dict['backbone.classifier.1.weight'].shape[0]
        original_dropout_rate = 0.4 
        
    except KeyError:
        print("Error: Could not find 'backbone.classifier.1.weight' in state_dict.")
        print("Please ensure 'final_cnn_model.pth' matches 'EfficientNetClassifier'.")
        return None
    except Exception as e:
        print(f"Error inferring model parameters: {e}")
        return None

    print(f"Inferred Original Params: num_classes={original_num_classes}, dropout_rate={original_dropout_rate}")

    model = EfficientNetClassifier(
        num_classes=original_num_classes,
        dropout_rate=original_dropout_rate
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    print("Successfully loaded pre-trained weights into model.")

    print("Freezing all pre-trained layer weights (setting requires_grad=False)...")
    for param in model.parameters():
        param.requires_grad = False

    print("Model loaded and frozen.")
    return model


class LogitFusionModel(nn.Module):

    def __init__(self, cnn_model_path, mlp_model_path, 
                 num_final_classes=5,
                 fusion_hidden_width=256, 
                 fusion_hidden_depth=2,
                 fusion_dropout=0.4):
        super().__init__()
        
        self.cnn_extractor = load_and_freeze_cnn(model_path=cnn_model_path)
        self.mlp_extractor = load_and_freeze_mlp(model_path=mlp_model_path)
        
        self.final_classifier = MLP(
            input_dim=10,
            output_dim=num_final_classes,
            hidden_width=fusion_hidden_width,
            hidden_depth=fusion_hidden_depth,
            dropout_rate=fusion_dropout
        )
        
        if self.cnn_extractor:
            self.cnn_extractor.eval()
        if self.mlp_extractor:
            self.mlp_extractor.eval()

    def forward(self, x_spectrogram, x_librosa):
        with torch.no_grad():
            cnn_features = self.cnn_extractor(x_spectrogram) 
            mlp_features = self.mlp_extractor(x_librosa)     
        
        fused_features = torch.cat([cnn_features, mlp_features], dim=1) 
        output = self.final_classifier(fused_features)
        
        return output


class MusicDataset(Dataset):
    def __init__(self, tracks_df, features_df, spectrogram_dir, transform=None, genre_map=None):
        self.tracks_df = tracks_df.reset_index(drop=True)
        self.features_df = features_df
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        self.genre_map = genre_map if genre_map else {}

    def __len__(self):
        return len(self.tracks_df)

    def __getitem__(self, idx):
        row = self.tracks_df.iloc[idx]
        track_id = row['track_id']
        genre_id = row['genre_id']
        
        label = self.genre_map.get(genre_id, genre_id)
        
        try:
            librosa_features = self.features_df.loc[track_id].values
            librosa_tensor = torch.tensor(librosa_features, dtype=torch.float32)
        except KeyError:
            librosa_tensor = torch.zeros((518,), dtype=torch.float32)

        img_name = f"{track_id:06d}.png"
        img_path = os.path.join(self.spectrogram_dir, img_name)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            spectrogram_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            return (spectrogram_tensor, librosa_tensor), label

        if self.transform:
            spectrogram_tensor = self.transform(img)
        else:
            spectrogram_tensor = transforms.ToTensor()(img)
            
        return (spectrogram_tensor, librosa_tensor), label


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() 
    total_loss = 0.0
    for (spectrogram, librosa), labels in loader:
        spectrogram, librosa, labels = spectrogram.to(device), librosa.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(spectrogram, librosa)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval() 
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (spectrogram, librosa), labels in loader:
            spectrogram, librosa, labels = spectrogram.to(device), librosa.to(device), labels.to(device)
            
            logits = model(spectrogram, librosa)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def train_final_model(best_result, config):
    print("\n" + "="*80)
    print("FINAL MODEL TRAINING")
    print(f"Best HP: width={best_result['hidden_width']}, depth={best_result['hidden_depth']}")
    print(f"Optimal Epochs (from CV): {best_result['avg_epochs']}")
    print("="*80)

    device = config['device']
    
    all_tracks_df = pd.read_csv(config['tracks_file'])
    all_features_df = pd.read_csv(config['features_file']).set_index('track_id')
    
    unique_genres = sorted(all_tracks_df['genre_id'].unique())
    num_final_classes = len(unique_genres)
    genre_to_idx_map = {genre: idx for idx, genre in enumerate(unique_genres)}
    
    full_dataset = MusicDataset(
        tracks_df=all_tracks_df,
        features_df=all_features_df,
        spectrogram_dir=config['spectrogram_dir'],
        transform=config['transform'],
        genre_map=genre_to_idx_map
    )
    
    full_loader = DataLoader(
        full_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    print(f"Full dataset: {len(full_dataset)} samples")
    
    model = LogitFusionModel(
        cnn_model_path=config['cnn_path'],
        mlp_model_path=config['mlp_path'],
        num_final_classes=num_final_classes,
        fusion_hidden_width=best_result['hidden_width'],
        fusion_hidden_depth=best_result['hidden_depth'],
        fusion_dropout=config['fusion_dropout']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    optimal_epochs = best_result['avg_epochs']
    print(f"\nStarting final training for {optimal_epochs} epochs...")
    
    for epoch in range(optimal_epochs):
        train_loss = train_one_epoch(model, full_loader, criterion, optimizer, device)
        
        if (epoch + 1) % 5 == 0 or epoch == optimal_epochs - 1:
            print(f"Epoch {epoch + 1}/{optimal_epochs} | Train Loss: {train_loss:.4f}")
    
    final_model_path = os.path.join(config['model_save_dir'], 'best_feature_fusion_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    import json
    config_path = os.path.join(config['model_save_dir'], 'best_feature_fusion_config.json')
    config_dict = {
        'hidden_width': best_result['hidden_width'],
        'hidden_depth': best_result['hidden_depth'],
        'avg_epochs': best_result['avg_epochs'],
        'mean_accuracy': best_result['mean_accuracy'],
        'std_accuracy': best_result['std_accuracy'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'fusion_dropout': config['fusion_dropout']
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Config saved: {config_path}")



def main():
    BASE_DATA_DIR = '/home/UG/yumin002/SC4001_Project/FINAL/data/train2/'
    MODEL_SAVE_DIR = '/home/UG/yumin002/SC4001_Project/FINAL/fusion_logit/models/'

    TRACKS_FILE = os.path.join(BASE_DATA_DIR, 'tracks_train2.csv')
    FEATURES_FILE = os.path.join(BASE_DATA_DIR, 'features_train2.csv')
    SPECTROGRAM_DIR = os.path.join(BASE_DATA_DIR, 'spectrograms_train2')

    CNN_PATH = '/home/UG/yumin002/SC4001_Project/FINAL/cnn_cv/best_cnn_phase1.pth'
    MLP_PATH = '/home/UG/yumin002/SC4001_Project/FINAL/mlp_cv/final_mlp_model.pth'
    
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_FOLDS = 3
    EARLY_STOPPING_PATIENCE = 5 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FUSION_DROPOUT = 0.4
    
    print(f"Using device: {DEVICE}")
    
    HP_GRID = {
        'hidden_width': [16, 32, 64],
        'hidden_depth': [1, 2, 3]
    }
    
    IMG_SIZE = 224
    spectrogram_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("="*80)
    print("FEATURE-LEVEL FUSION WITH GRID SEARCH")
    print("="*80)
    
    all_tracks_df = pd.read_csv(TRACKS_FILE)
    all_features_df = pd.read_csv(FEATURES_FILE).set_index('track_id')
    
    unique_genres = sorted(all_tracks_df['genre_id'].unique())
    num_final_classes = len(unique_genres)
    genre_to_idx_map = {genre: idx for idx, genre in enumerate(unique_genres)}
    print(f"Genre map: {genre_to_idx_map}")
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    print(f"\nStarting Grid Search: {len(HP_GRID['hidden_width']) * len(HP_GRID['hidden_depth'])} combinations")
    
    grid_results = []
    
    for hidden_width in HP_GRID['hidden_width']:
        for hidden_depth in HP_GRID['hidden_depth']:
            print("\n" + "="*80)
            print(f"TESTING HP: hidden_width={hidden_width}, hidden_depth={hidden_depth}")
            print("="*80)
            
            fold_accuracies = []
            fold_epochs = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(all_tracks_df)):
                print(f"\n--- FOLD {fold + 1}/{NUM_FOLDS} ---")
                
                df_train = all_tracks_df.iloc[train_idx].reset_index(drop=True)
                df_val = all_tracks_df.iloc[val_idx].reset_index(drop=True)
                
                print(f"Fold {fold + 1} split: {len(df_train)} train, {len(df_val)} val samples.")
            
                train_dataset = MusicDataset(
                    tracks_df=df_train,
                    features_df=all_features_df,
                    spectrogram_dir=SPECTROGRAM_DIR,
                    transform=spectrogram_transform,
                    genre_map=genre_to_idx_map
                )
                val_dataset = MusicDataset(
                    tracks_df=df_val,
                    features_df=all_features_df,
                    spectrogram_dir=SPECTROGRAM_DIR,
                    transform=spectrogram_transform,
                    genre_map=genre_to_idx_map
                )
            
                train_loader = DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
                )
            
                print(f"Initializing model with width={hidden_width}, depth={hidden_depth}...")
                
                model = LogitFusionModel(
                    cnn_model_path=CNN_PATH,
                    mlp_model_path=MLP_PATH,
                    num_final_classes=num_final_classes,
                    fusion_hidden_width=hidden_width,
                    fusion_hidden_depth=hidden_depth,
                    fusion_dropout=FUSION_DROPOUT
                ).to(DEVICE)
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                
                best_val_accuracy = 0.0
                epochs_no_improve = 0 
                best_model_path = os.path.join(
                    MODEL_SAVE_DIR, 
                    f'hp_w{hidden_width}_d{hidden_depth}_fold{fold}.pth'
                )

                best_val_accuracy = 0.0
                epochs_no_improve = 0
                stopped_epoch = NUM_EPOCHS

                print(f"Starting Training for Fold {fold + 1}...")
                for epoch in range(NUM_EPOCHS):
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
                    val_loss, val_accuracy = validate(model, val_loader, criterion, DEVICE)
                    
                    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Accuracy: {val_accuracy:.4f}")
                    
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        epochs_no_improve = 0
                        torch.save(model.state_dict(), best_model_path)
                        print(f"  -> New best model saved")
                    else:
                        epochs_no_improve += 1
                        
                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        stopped_epoch = epoch + 1
                        print(f"  -> Early stopping at epoch {epoch + 1}")
                        break 

                print(f"--- Fold {fold + 1} Complete ---")
                print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
                fold_accuracies.append(best_val_accuracy)
                fold_epochs.append(stopped_epoch)
            
            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            avg_epochs = int(np.mean(fold_epochs))
            
            result = {
                'hidden_width': hidden_width,
                'hidden_depth': hidden_depth,
                'fold_accuracies': fold_accuracies,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'avg_epochs': avg_epochs
            }
            grid_results.append(result)
            
            print("\n" + "-"*80)
            print(f"HP: width={hidden_width}, depth={hidden_depth}")
            print(f"Fold Accuracies: {[round(acc, 4) for acc in fold_accuracies]}")
            print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
            print(f"Average Epochs: {avg_epochs}")
            print("-"*80)
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    
    grid_results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
    
    print("\nAll HP Combinations:")
    for i, result in enumerate(grid_results, 1):
        print(f"{i}. width={result['hidden_width']}, depth={result['hidden_depth']} | "
              f"Mean Acc: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})")
    
    print("\n" + "="*80)
    print("BEST HYPERPARAMETER COMBINATION:")
    best_result = grid_results[0]
    print(f"  hidden_width: {best_result['hidden_width']}")
    print(f"  hidden_depth: {best_result['hidden_depth']}")
    print(f"  Mean CV Accuracy: {best_result['mean_accuracy']:.4f} (+/- {best_result['std_accuracy']:.4f})")
    print(f"  Average Epochs: {best_result['avg_epochs']}")
    print("="*80)
    
    results_df = pd.DataFrame([
        {
            'hidden_width': r['hidden_width'],
            'hidden_depth': r['hidden_depth'],
            'mean_accuracy': r['mean_accuracy'],
            'std_accuracy': r['std_accuracy'],
            'fold_0_acc': r['fold_accuracies'][0],
            'fold_1_acc': r['fold_accuracies'][1],
            'fold_2_acc': r['fold_accuracies'][2]
        }
        for r in grid_results
    ])
    results_csv_path = os.path.join(MODEL_SAVE_DIR, 'grid_search_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")

    config_dict = {
        'device': DEVICE,
        'tracks_file': TRACKS_FILE,
        'features_file': FEATURES_FILE,
        'spectrogram_dir': SPECTROGRAM_DIR,
        'cnn_path': CNN_PATH,
        'mlp_path': MLP_PATH,
        'transform': spectrogram_transform,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'fusion_dropout': FUSION_DROPOUT,
        'model_save_dir': MODEL_SAVE_DIR
    }
    train_final_model(best_result, config_dict)

if __name__ == "__main__":
    main()
