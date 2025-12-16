import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import sys

from dataset import SpectrogramDataset
from cnn_model import EfficientNetClassifier

def evaluate_model(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

if __name__ == "__main__":
    BEST_DROPOUT_RATE = 0.5
    BEST_BATCH_SIZE = 64
    
    DATA_PATH = '/home/UG/n2402401d/SC4001_Project/FINAL/data/'
    TEST_TRACKS_FILE = f'{DATA_PATH}tracks_test.csv'
    TEST_IMG_DIR = f'{DATA_PATH}spectrograms_test'
    
    MODEL_FILE = '/home/UG/n2402401d/SC4001_Project/FINAL/cnn_cv/best_cnn_phase1.pth'
    
    NUM_CLASSES = 5 

    try:
        print("=" * 70)
        print("CNN TEST EVALUATION")
        print("=" * 70)
        
        print("\nStep 1: Loading TEST data...", flush=True)
        test_tracks_df = pd.read_csv(TEST_TRACKS_FILE)
        print(f"  > Test tracks loaded: {len(test_tracks_df)} samples", flush=True)
        
        print("\nStep 2: Creating test dataset...", flush=True)
        test_dataset = SpectrogramDataset(
            csv_file=TEST_TRACKS_FILE,
            img_dir=TEST_IMG_DIR
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
        
        model = EfficientNetClassifier(
            num_classes=NUM_CLASSES,
            dropout_rate=BEST_DROPOUT_RATE
        ).to(device)
        
        print(f"  > Model structure: EfficientNet-B0 with dropout={BEST_DROPOUT_RATE}")
        
        print(f"\nStep 4: Loading saved model weights from '{MODEL_FILE}'...", flush=True)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.eval()
        print("  > Model weights loaded successfully", flush=True)
        
        print("\nStep 5: Computing test accuracy...", flush=True)
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        print("\n" + "=" * 70)
        print("                    FINAL TEST RESULTS")
        print("=" * 70)
        print(f"  Test Loss:     {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc * 100:.2f}%")
        print("=" * 70)
        
        print("\n--- Test Configuration ---")
        print(f"  Batch Size:    {BEST_BATCH_SIZE}")
        print(f"  Dropout Rate:  {BEST_DROPOUT_RATE}")
        print(f"  Num Classes:   {NUM_CLASSES}")
        print(f"  Device:        {device}")
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