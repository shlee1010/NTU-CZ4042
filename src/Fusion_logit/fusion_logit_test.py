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


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (spectrograms, librosa), labels in dataloader:
            spectrograms = spectrograms.to(device)
            librosa = librosa.to(device)
            labels = labels.to(device)
            
            outputs = model(spectrograms, librosa)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


if __name__ == "__main__":
    BEST_HIDDEN_WIDTH = 32
    BEST_HIDDEN_DEPTH = 2
    BEST_DROPOUT = 0.4
    BEST_BATCH_SIZE = 64
    
    DATA_PATH = '/home/UG/yumin002/SC4001_Project/FINAL/data/test/'
    TEST_TRACKS_FILE = f'{DATA_PATH}tracks_test.csv'
    TEST_FEATURES_FILE = f'{DATA_PATH}features_test.csv'
    TEST_IMG_DIR = f'{DATA_PATH}spectrograms_test'
    
    MODEL_FILE = '/home/UG/yumin002/SC4001_Project/FINAL/fusion_logit/best_feature_fusion_final.pth'
    CNN_PATH = '/home/UG/yumin002/SC4001_Project/FINAL/cnn_cv/best_cnn_phase1.pth'
    MLP_PATH = '/home/UG/yumin002/SC4001_Project/FINAL/mlp_cv/final_mlp_model.pth'
    
    NUM_CLASSES = 5

    try:
        print("=" * 70)
        print("FEATURE-LEVEL FUSION TEST EVALUATION")
        print("=" * 70)
        
        print("\nStep 1: Loading TEST data...", flush=True)
        test_tracks_df = pd.read_csv(TEST_TRACKS_FILE)
        test_features_df = pd.read_csv(TEST_FEATURES_FILE).set_index('track_id')
        print(f"  > Test tracks: {len(test_tracks_df)} samples", flush=True)
        
        unique_genres = sorted(test_tracks_df['genre_id'].unique())
        genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
        print(f"  > Genre mapping: {genre_to_idx}", flush=True)
        
        print("\nStep 2: Creating test dataset...", flush=True)
        
        spectrogram_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = MusicDataset(
            tracks_df=test_tracks_df,
            features_df=test_features_df,
            spectrogram_dir=TEST_IMG_DIR,
            transform=spectrogram_transform,
            genre_map=genre_to_idx
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=BEST_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        print(f"  > Test dataset created: {len(test_dataset)} samples", flush=True)
        
        print("\nStep 3: Initializing model structure...", flush=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  > Using device: {device}", flush=True)
        
        model = LogitFusionModel(
            cnn_model_path=CNN_PATH,
            mlp_model_path=MLP_PATH,
            num_final_classes=NUM_CLASSES,
            fusion_hidden_width=BEST_HIDDEN_WIDTH,
            fusion_hidden_depth=BEST_HIDDEN_DEPTH,
            fusion_dropout=BEST_DROPOUT
        ).to(device)
        
        print(f"  > Fusion model: width={BEST_HIDDEN_WIDTH}, depth={BEST_HIDDEN_DEPTH}, dropout={BEST_DROPOUT}")
        
        print(f"\nStep 4: Loading saved model weights from '{MODEL_FILE}'...", flush=True)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.eval()
        print("  > Model weights loaded successfully", flush=True)
        
        print("\nStep 5: Computing test accuracy...", flush=True)
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        print("\n" + "=" * 70)
        print("                FINAL TEST RESULTS")
        print("=" * 70)
        print(f"  Test Loss:     {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc * 100:.2f}%")
        print("=" * 70)
        
        print("\n--- Test Configuration ---")
        print(f"  Batch Size:        {BEST_BATCH_SIZE}")
        print(f"  Hidden Width:      {BEST_HIDDEN_WIDTH}")
        print(f"  Hidden Depth:      {BEST_HIDDEN_DEPTH}")
        print(f"  Fusion Dropout:    {BEST_DROPOUT}")
        print(f"  Num Classes:       {NUM_CLASSES}")
        print(f"  Device:            {device}")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nERROR: FILE NOT FOUND! {e}", flush=True)
        print("  > Please ensure all file paths are correct.", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
