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
from models_custom import PhysNetBaseline, PhysNetConstrained
from loss_custom import PhysiologicalLoss

class SafeComparisonDataset(Dataset):
    def __init__(self, data_path, subjects, chunk_length=128, image_size=72):
        self.chunk_length = chunk_length
        self.image_size = image_size
        self.clips, self.labels = [], []
        
        vid_files = sorted(glob.glob(os.path.join(data_path, "**", "vid.avi"), recursive=True))
        for vid_path in tqdm(vid_files, desc="Loading Data"):
            folder = os.path.basename(os.path.dirname(vid_path))
            if not any(s in folder for s in subjects): continue
            
            gt_path = os.path.join(os.path.dirname(vid_path), "ground_truth.txt")
            try:
                gt_data = np.loadtxt(gt_path)
                spo2 = None
                for r in range(min(gt_data.shape[0], 2)):
                    if 85 <= np.mean(gt_data[r]) <= 101:
                        spo2 = gt_data[r]; break
                if spo2 is None: continue
                spo2 = np.clip(spo2, 80.0, 100.0)
                
                container = av.open(vid_path)
                frames = [np.array(f.to_image().resize((72, 72), Image.BILINEAR)) for f in container.decode(video=0)]
                container.close()
                frames = np.array(frames, dtype=np.float32)
                
                n = frames.shape[0]
                diff = np.zeros_like(frames)
                for j in range(n-1): diff[j] = (frames[j+1]-frames[j])/(frames[j+1]+frames[j]+1e-7)
                diff /= (np.std(diff)+1e-7)
                
                res_spo2 = np.interp(np.linspace(0,1,n), np.linspace(0,1,len(spo2)), spo2)
                for s in range(0, n-chunk_length+1, chunk_length):
                    self.clips.append(np.transpose(diff[s:s+chunk_length], (3,0,1,2)))
                    self.labels.append(res_spo2[s:s+chunk_length])
            except: continue

    def __len__(self): return len(self.clips)
    def __getitem__(self, i): 
        return torch.tensor(self.clips[i], dtype=torch.float32), \
               torch.tensor(self.labels[i], dtype=torch.float32)

def train_and_evaluate(model_type, train_loader, val_loader, device):
    print(f"Initializing {model_type} approach...")
    if model_type == 'baseline':
        model = PhysNetBaseline(frames=128).to(device)
        criterion = nn.MSELoss()
    else:
        model = PhysNetConstrained(frames=128).to(device)
        criterion = PhysiologicalLoss(lambda_smooth=1.0)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print(f"Training {model_type} for 5 epochs...")
    for epoch in range(5):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); out, _, _, _ = model(inputs)
            loss = criterion(out, labels); loss.backward(); optimizer.step()
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            out, _, _, _ = model(inputs.to(device))
            all_preds.append(out.cpu().numpy()); all_labels.append(labels.numpy())
    
    mae = np.mean(np.abs(np.concatenate(all_preds) - np.concatenate(all_labels)))
    return model, all_preds[0][0], all_labels[0][0], mae

def run_final_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_subs = ["subject1", "subject4", "subject8", "subject9"]
    val_subs = ["subject5"] 
    
    print("--- Loading Data ---")
    train_ds = SafeComparisonDataset("./data", train_subs)
    val_ds = SafeComparisonDataset("./data", val_subs)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    print("\n--- Training Approach 1: Baseline (Data-Driven) ---")
    m1, pred1, truth, mae1 = train_and_evaluate('baseline', train_loader, val_loader, device)
    
    print("\n--- Training Approach 2: Constrained (Physiological) ---")
    m2, pred2, _, mae2 = train_and_evaluate('constrained', train_loader, val_loader, device)

    print(f"\nFinal Results:\nBaseline MAE: {mae1:.4f}%\nConstrained MAE: {mae2:.4f}%")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Subplot 1: Time Series
    ax1.plot(truth, color='black', label='True SpO2', linewidth=2)
    ax1.plot(pred1, color='red', linestyle='--', label='Baseline Pred')
    ax1.plot(pred2, color='green', label='Constrained Pred')
    ax1.set_title('SpO2 Estimation Comparison'); ax1.set_ylabel('SpO2 (%)'); ax1.legend()
    
    # Subplot 2: Bar Chart
    ax2.bar(['Baseline', 'Constrained'], [mae1, mae2], color=['red', 'green'])
    ax2.set_title('Mean Absolute Error (Lower is Better)'); ax2.set_ylabel('MAE (%)')
    
    plt.tight_layout(); plt.savefig('final_hackathon_comparison.png')
    print("\nComparison visualization saved as 'final_hackathon_comparison.png'")

if __name__ == "__main__":
    run_final_comparison()
