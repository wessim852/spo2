import os
import sys
import glob
import av 
from PIL import Image 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add rPPG-Toolbox to path
sys.path.append(os.path.join(os.getcwd(), 'rPPG-Toolbox'))
from models_custom import PhysNetConstrained
from loss_custom import PhysiologicalLoss

class UltraSafeUBFCDataset(Dataset):
    def __init__(self, data_path, subjects_to_load=None, chunk_length=128, image_size=72):
        self.chunk_length = chunk_length
        self.image_size = image_size
        self.clips = []
        self.labels = []
        self.subject_ids = [] # Track which subject each chunk belongs to
        
        vid_files = sorted(glob.glob(os.path.join(data_path, "**", "vid.avi"), recursive=True))
        
        for vid_path in vid_files:
            subject_dir = os.path.dirname(vid_path)
            folder_name = os.path.basename(subject_dir)
            
            # Filter by subject if list provided
            if subjects_to_load is not None:
                if not any(sub in folder_name for sub in subjects_to_load):
                    continue

            gt_path = os.path.join(subject_dir, "ground_truth.txt")
            if not os.path.exists(gt_path): continue
                
            try:
                gt_data = np.loadtxt(gt_path)
                spo2_signal = None
                for row_idx in range(min(gt_data.shape[0], 2)):
                    row = gt_data[row_idx]
                    mean_val = np.mean(row)
                    if 85 <= mean_val <= 101:
                        spo2_signal = row
                        break
                    elif 80 <= mean_val <= 115:
                        spo2_signal = row
                
                if spo2_signal is None: continue
                spo2_signal = np.clip(spo2_signal, 80.0, 100.0)
                
            except: continue

            # Video loading
            frames = []
            try:
                container = av.open(vid_path)
                stream = container.streams.video[0]
                for frame in container.decode(stream):
                    img = frame.to_image().resize((self.image_size, self.image_size), Image.BILINEAR)
                    frames.append(np.array(img))
                container.close()
            except: continue
            
            if len(frames) == 0: continue
            frames = np.array(frames, dtype=np.float32)

            # Preprocessing
            n, h, w, c = frames.shape
            diff_data = np.zeros((n, h, w, c), dtype=np.float32)
            for j in range(n - 1):
                diff_data[j] = (frames[j+1] - frames[j]) / (frames[j+1] + frames[j] + 1e-7)
            diff_data = diff_data / (np.std(diff_data) + 1e-7)

            resampled_spo2 = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(spo2_signal)),
                spo2_signal
            )

            for start in range(0, n - chunk_length + 1, chunk_length):
                clip = np.transpose(diff_data[start:start+chunk_length], (3, 0, 1, 2))
                lbl = resampled_spo2[start:start+chunk_length]
                
                # --- DATA AUGMENTATION: OVERSAMPLING LOW SpO2 ---
                # If any frame in this chunk is < 96%, repeat it 10x
                if np.min(lbl) < 96.0:
                    for _ in range(10):
                        self.clips.append(clip)
                        self.labels.append(lbl)
                        self.subject_ids.append(folder_name)
                else:
                    self.clips.append(clip)
                    self.labels.append(lbl)
                    self.subject_ids.append(folder_name)

    def __len__(self): return len(self.clips)
    def __getitem__(self, idx):
        return torch.tensor(self.clips[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32), \
               self.subject_ids[idx]

def train_hackathon():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    FRAME_NUM = 128
    DATA_PATH = "./data"

    # --- SUBJECT SPLIT ---
    # We explicitly put subject5 (breath hold) in validation
    val_subs = ["subject5"]
    train_subs = ["subject1", "subject4", "subject8", "subject9", "subject10", "subject11", "subject12", "subject3"]

    print("Loading Training Set...")
    train_ds = UltraSafeUBFCDataset(DATA_PATH, subjects_to_load=train_subs, chunk_length=FRAME_NUM)
    print("Loading Validation Set...")
    val_ds = UltraSafeUBFCDataset(DATA_PATH, subjects_to_load=val_subs, chunk_length=FRAME_NUM)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = PhysNetConstrained(frames=FRAME_NUM).to(device)
    criterion = PhysiologicalLoss(lambda_smooth=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_maes = [], []
    best_mae = float('inf')
    os.makedirs("PreTrainedModels", exist_ok=True)

    for epoch in range(100):
        model.train()
        epoch_loss, n_b = 0, 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        
        train_losses.append(epoch_loss / n_b)

        model.eval()
        total_mae, n_v = 0, 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                outputs, _, _, _ = model(inputs.to(device))
                total_mae += torch.mean(torch.abs(outputs - labels.to(device))).item()
                n_v += 1
        
        avg_mae = total_mae / n_v
        val_maes.append(avg_mae)

        # Save Best Model
        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save(model.state_dict(), "PreTrainedModels/best_model.pth")
            save_msg = " (Model Saved!)"
        else:
            save_msg = ""

        print(f"Epoch [{epoch+1}/100] - Loss: {train_losses[-1]:.4f} - Val MAE: {avg_mae:.4f}%{save_msg}")

    # --- FINAL PLOT ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.plot(train_losses); plt.title('Train Loss')
    plt.subplot(1, 3, 2); plt.plot(val_maes, color='orange'); plt.title('Val MAE (%)')
    
    plt.subplot(1, 3, 3)
    # Load Best Model for Plotting
    if os.path.exists("PreTrainedModels/best_model.pth"):
        model.load_state_dict(torch.load("PreTrainedModels/best_model.pth", map_location=device))
        print("Loaded best model for final SpO2 comparison plot.")
    
    model.eval()
    with torch.no_grad():
        # Get a sample from the validation loader
        inputs, labels, sub_id = next(iter(val_loader))
        outputs, _, _, _ = model(inputs.to(device))
        
        plt.plot(outputs[0].cpu().detach().numpy(), color='red', label='Predicted SpO2')
        plt.plot(labels[0].cpu().numpy(), color='green', label='True SpO2')
        plt.title(f'SpO2 Comparison ({sub_id[0]})')
        plt.ylabel('SpO2 (%)')
        plt.legend()
        plt.ylim(80, 105)

    plt.tight_layout(); plt.savefig('check_physio.png')
    print("Done. Saved check_physio.png with general SpO2 comparison.")

if __name__ == "__main__":
    train_hackathon()
