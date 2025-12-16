import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold 
import itertools
from collections import defaultdict
import copy

class CustomFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feature_tensor, label_tensor

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

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def train_final_model(best_params, num_epochs, all_features_df, all_tracks_df, dropout_rate):
    print(f"\n--- Re-training final model on ALL data for {num_epochs} epochs ---")
    
    merged_df = pd.merge(all_features_df, all_tracks_df, on='track_id')
    feature_cols = [col for col in all_features_df.columns if col != 'track_id']
    target_col = 'genre_id'
    X = merged_df[feature_cols] 
    y = merged_df[target_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"  > Final model label mapping. Original: {np.unique(y)}, Encoded: {np.unique(y_encoded)}")
    
    output_dim = len(le.classes_) 
    input_dim = len(feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    full_dataset = CustomFeatureDataset(X_scaled, y_encoded)
    train_loader = DataLoader(full_dataset, batch_size=128, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim, 
        hidden_width=best_params['width'],
        hidden_depth=best_params['depth'],
        dropout_rate=dropout_rate 
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            print(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

    try:
        torch.save(model.state_dict(), 'final_mlp_model.pth')
        print(f"\nSuccessfully saved final model to 'final_mlp_model.pth'")
    except Exception as e:
        print(f"\nError saving model: {e}")

    return model

def main():

    DATA_PATH = '/home/UG/yumin002/SC4001_Project/FINAL/data/train1/'
    TRACKS_FILE = 'tracks_train1.csv' 
    FEATURES_FILE = 'features_train1.csv' 
    
    OUTPUT_FILE = 'mlp_cv_results.csv' 
    NUM_FOLDS = 3
    
    param_grid = {
        'learning_rate': [0.0001, 0.0005, 0.001],
        'hidden_depth': [2, 3, 4],
        'hidden_width': [64, 128, 256]
    }
    
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    DROPOUT_RATE = 0.4
    
    EARLY_STOPPING_PATIENCE = 5
    MIN_DELTA = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    try:
        print("Step 1: Loading tracks_df...", flush=True)
        tracks_df = pd.read_csv(f"{DATA_PATH}{TRACKS_FILE}")
        print(f"  > tracks_df loaded. Shape: {tracks_df.shape}", flush=True)
        
        print("Step 2: Loading features_df...", flush=True)
        features_df = pd.read_csv(f"{DATA_PATH}{FEATURES_FILE}")
        print(f"  > features_df loaded. Shape: {features_df.shape}", flush=True)

        print("Step 3: Merging dataframes...", flush=True)
        merged_df = pd.merge(features_df, tracks_df, on='track_id')
        print(f"  > Data merged. Shape: {merged_df.shape}", flush=True)

        print("Step 4: Defining feature columns...", flush=True)
        feature_cols = [col for col in features_df.columns if col != 'track_id']
        target_col = 'genre_id'
        
        X = merged_df[feature_cols] 
        y = merged_df[target_col]
        print(f"  > Features (X) shape: {X.shape}", flush=True)
        
        print("Step 5: Encoding labels (y)...", flush=True)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"  > Label mapping complete. Original IDs: {np.unique(y)}, Encoded IDs: {np.unique(y_encoded)}")
        
        output_dim = len(le.classes_) 
        input_dim = len(feature_cols)
        print(f"  > Data ready. Input features: {input_dim}, Output classes: {output_dim}", flush=True)
        
    except FileNotFoundError as e:
        print(f"ERROR: FILE NOT FOUND! {e}", flush=True)
        print("  > Please ensure 'tracks_split_mlp_cnn_with_3folds.csv' and 'features_train.csv' are in the 'data/' directory.")
        return
    except KeyError as e:
        print(f"ERROR: Missing column {e}. Check your CSV files.", flush=True)
        return
    except Exception as e:
        print(f"ERROR during data loading: {e}", flush=True)
        return

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    print(f"\n--- KFold Initialized ({NUM_FOLDS} splits, shuffle=True) ---", flush=True)

    all_results = []
    best_avg_accuracy = -1.0
    best_params = {}
    
    best_run_epochs = [] 
    
    param_combinations = list(itertools.product(
        param_grid['learning_rate'],
        param_grid['hidden_depth'],
        param_grid['hidden_width']
    ))

    print(f"\nStarting {NUM_FOLDS}-fold CV for {len(param_combinations)} hyperparameter combinations...", flush=True)

    for lr, depth, width in param_combinations:
        params_key = f"lr={lr}, depth={depth}, width={width}"
        print(f"\n--- Testing params: {params_key} (Dropout={DROPOUT_RATE}) ---", flush=True)
        
        fold_accuracies = []
        current_run_epochs = [] 

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
            
            print(f"  Fold {fold+1}/{NUM_FOLDS}", flush=True)
            
            print(f"    Fold {fold+1} Step A: Splitting data (using KFold indices)...", flush=True)
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
            
            try:
                print(f"    Fold {fold+1} Step B: Scaling data (StandardScaler)...", flush=True)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train) 
                X_val_scaled = scaler.transform(X_val)
                print(f"    Fold {fold+1} Step C: Creating datasets...", flush=True)
            except Exception as e:
                print(f"    ERROR during StandardScaler: {e}", flush=True)
                print("    Skipping this fold.", flush=True)
                continue 
            
            train_dataset = CustomFeatureDataset(X_train_scaled, y_train)
            val_dataset = CustomFeatureDataset(X_val_scaled, y_val)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            print(f"    Fold {fold+1} Step D: Loaders created. Starting training...", flush=True)
            
            model = MLP(
                input_dim, 
                output_dim, 
                hidden_width=width, 
                hidden_depth=depth, 
                dropout_rate=DROPOUT_RATE 
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            early_stopper = EarlyStopper(
                patience=EARLY_STOPPING_PATIENCE, 
                min_delta=MIN_DELTA
            )
            
            stopped_at_epoch = NUM_EPOCHS 

            for epoch in range(NUM_EPOCHS):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
                
                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch {epoch+1}/{NUM_EPOCHS}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                if early_stopper.check(val_loss, model):
                    print(f"    Early stopping at epoch {epoch+1}")
                    stopped_at_epoch = epoch + 1 
                    break 
            
            if early_stopper.best_model_state:
                model.load_state_dict(early_stopper.best_model_state)
            
            final_val_loss, final_val_acc = evaluate_model(model, val_loader, criterion, device)
            print(f"    Fold {fold+1} Best Val Accuracy (at min loss): {final_val_acc:.4f}", flush=True)
            fold_accuracies.append(final_val_acc)
            
            current_run_epochs.append(stopped_at_epoch)

        avg_acc = np.mean(fold_accuracies) if fold_accuracies else 0
        std_acc = np.std(fold_accuracies) if fold_accuracies else 0
        run_data = {
            'learning_rate': lr,
            'hidden_depth': depth,
            'hidden_width': width,
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc
        }
        all_results.append(run_data)
        
        print(f"  Params: {params_key}, Avg Accuracy: {avg_acc:.4f} (+/- {std_acc:.4f})", flush=True)

        if avg_acc > best_avg_accuracy:
            best_avg_accuracy = avg_acc
            best_params = {
                'lr': lr, 
                'depth': depth, 
                'width': width
            }
            best_run_epochs = current_run_epochs.copy()
            print(f"  > New best run! Saving epochs: {best_run_epochs}", flush=True)

    print("\n--- Saving Results ---", flush=True)
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by='avg_accuracy', ascending=False)
    
    try:
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved results to {OUTPUT_FILE}", flush=True)
    except PermissionError:
        print(f"Error: Could not write to {OUTPUT_FILE}. Check permissions.", flush=True)
    
    print("\n--- Best Hyperparameters Found ---", flush=True)
    print(f"Parameters: {best_params} (with fixed Dropout={DROPOUT_RATE})", flush=True)
    print(f"Best Average Accuracy: {best_avg_accuracy:.4f}", flush=True)
    
    final_train_epochs = NUM_EPOCHS 
    if best_run_epochs:
        final_train_epochs = int(np.mean(best_run_epochs))
        print(f"Optimal epochs (from CV): {final_train_epochs} (avg of {best_run_epochs})", flush=True)
    else:
        print(f"Warning: Could not determine optimal epochs. Defaulting to {NUM_EPOCHS}", flush=True)

    if best_avg_accuracy > -1.0:
        print("Loading data for final re-training...", flush=True)
        tracks_df_final = pd.read_csv(f"{DATA_PATH}{TRACKS_FILE}")
        features_df_final = pd.read_csv(f"{DATA_PATH}{FEATURES_FILE}")

        train_final_model(
            best_params, 
            final_train_epochs,
            features_df_final, 
            tracks_df_final, 
            DROPOUT_RATE
        )

if __name__ == "__main__":
    main()