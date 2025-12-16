import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import sys

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

if __name__ == "__main__":
    BEST_HIDDEN_WIDTH = 64
    BEST_HIDDEN_DEPTH = 3
    
    FIXED_DROPOUT_RATE = 0.4
    
    TRAIN_DATA_PATH = '/home/UG/yumin002/SC4001_Project/FINAL/data/'
    TRAIN_TRACKS_FILE = f'{TRAIN_DATA_PATH}train1/tracks_train1.csv'
    TRAIN_FEATURES_FILE = f'{TRAIN_DATA_PATH}train1/features_train1.csv'

    TEST_TRACKS_FILE = '/home/UG/yumin002/SC4001_Project/FINAL/data/test/tracks_test.csv'
    TEST_FEATURES_FILE = '/home/UG/yumin002/SC4001_Project/FINAL/data/test/features_test.csv'
    
    MODEL_FILE = '/home/UG/yumin002/SC4001_Project/FINAL/mlp_cv/final_mlp_model.pth'

    try:
        print("Step 1: Loading TRAINING data to fit scaler and encoder...", flush=True)
        train_tracks_df = pd.read_csv(TRAIN_TRACKS_FILE)
        train_features_df = pd.read_csv(TRAIN_FEATURES_FILE)
        
        train_merged_df = pd.merge(train_features_df, train_tracks_df, on='track_id')
        
        feature_cols = [col for col in train_features_df.columns if col != 'track_id']
        target_col = 'genre_id'

        X_train = train_merged_df[feature_cols]
        y_train = train_merged_df[target_col]

        le = LabelEncoder()
        le.fit(y_train)
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        input_dim = len(feature_cols)
        output_dim = len(le.classes_)
        
        print(f"  > Scaler and Encoder fitted. Input dim: {input_dim}, Output dim: {output_dim}", flush=True)

        print("Step 2: Loading TEST data...", flush=True)
        test_tracks_df = pd.read_csv(TEST_TRACKS_FILE)
        test_features_df = pd.read_csv(TEST_FEATURES_FILE)
        
        test_merged_df = pd.merge(test_features_df, test_tracks_df, on='track_id')

        X_test = test_merged_df[feature_cols]
        y_test = test_merged_df[target_col]

        print("Step 3: Transforming TEST data using fitted scaler/encoder...", flush=True)
        X_test_scaled = scaler.transform(X_test)
        
        y_test_encoded = le.transform(y_test)
        
        test_dataset = CustomFeatureDataset(X_test_scaled, y_test_encoded)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        print(f"  > Test data loaded. Samples: {len(test_dataset)}", flush=True)

        print("Step 4: Initializing model structure...", flush=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  > Using device: {device}", flush=True)

        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_width=BEST_HIDDEN_WIDTH,
            hidden_depth=BEST_HIDDEN_DEPTH,
            dropout_rate=FIXED_DROPOUT_RATE
        ).to(device)
        
        print(f"  > Model structure: {BEST_HIDDEN_DEPTH} hidden layers, {BEST_HIDDEN_WIDTH} width")

        print(f"Step 5: Loading saved model weights from '{MODEL_FILE}'...", flush=True)

        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        
        model.eval()
        
        criterion = nn.CrossEntropyLoss()

        print("Step 6: Computing test accuracy...", flush=True)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        print("\n" + "="*30)
        print("       FINAL TEST RESULTS")
        print("="*30)
        print(f"  Test Loss:     {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc * 100:.2f}%")
        print("="*30)

    except FileNotFoundError as e:
        print(f"\nERROR: FILE NOT FOUND! {e}", flush=True)
        print("  > Please ensure all file paths at the top of the script are correct.", flush=True)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", flush=True)
        import traceback
        traceback.print_exc()