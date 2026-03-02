import os
import sys
import glob
import av 
from PIL import Image 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add rPPG-Toolbox to path
sys.path.append(os.path.join(os.getcwd(), 'rPPG-Toolbox'))
from models_custom import PhysNetBaseline, PhysNetConstrained

class SafeComparisonDataset(Dataset):
    def __init__(self, data_path, subjects, chunk_length=128, image_size=72):
        self.chunk_length = chunk_length
        self.clips, self.labels = [], []
        
        vid_files = sorted(glob.glob(os.path.join(data_path, "**", "vid.avi"), recursive=True))
        for vid_path in vid_files:
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

def run_final_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    BEST_MODEL_PATH = "PreTrainedModels/best_model.pth"
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: {BEST_MODEL_PATH} not found. Please run train_hackathon.py first.")
        return

    # 1. Setup Models
    print("--- Initializing Models ---")
    # Untrained Baseline
    baseline_model = PhysNetBaseline(frames=128).to(device)
    baseline_model.eval()
    
    # Our Best Constrained Model
    constrained_model = PhysNetConstrained(frames=128).to(device)
    print(f"Loading pre-trained weights from {BEST_MODEL_PATH}...")
    constrained_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    constrained_model.eval()

    # 2. Load Evaluation Data (Subject 5 - the breath hold subject)
    print("--- Loading Validation Data (Subject 5) ---")
    val_ds = SafeComparisonDataset("./data", ["subject5"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    if len(val_ds) == 0:
        print("Error: Could not load subject5 data.")
        return

    # 3. Perform Inference
    print("--- Running Inference ---")
    all_baseline_preds = []
    all_constrained_preds = []
    all_truth = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            
            # Baseline Inference
            out_b, _, _, _ = baseline_model(inputs)
            all_baseline_preds.append(out_b.cpu().numpy())
            
            # Constrained Inference
            out_c, _, _, _ = constrained_model(inputs)
            all_constrained_preds.append(out_c.cpu().numpy())
            
            all_truth.append(labels.numpy())

    # Calculate MAEs
    baseline_mae = np.mean(np.abs(np.concatenate(all_baseline_preds) - np.concatenate(all_truth)))
    constrained_mae = np.mean(np.abs(np.concatenate(all_constrained_preds) - np.concatenate(all_truth)))

    print(f"\nBenchmark Results on Subject 5:")
    print(f"Untrained Baseline MAE: {baseline_mae:.4f}%")
    print(f"Pre-trained Constrained MAE: {constrained_mae:.4f}%")

    # 4. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Subplot 1: Time Series Comparison
    # We take the first chunk for plotting
    truth = all_truth[0][0]
    pred_b = all_baseline_preds[0][0]
    # Apply constraint ONLY for visualization purposes so it appears in the 80-105% range
    pred_b_viz = 80 + 20 * (1 / (1 + np.exp(-pred_b)))
    pred_c = all_constrained_preds[0][0]
    
    ax1.plot(truth, color='black', label='Ground Truth', linewidth=2)
    ax1.plot(pred_b_viz, color='red', linestyle='--', label='Baseline (Untrained)')
    ax1.plot(pred_c, color='green', label='OxyVision (Best Model)', linewidth=2)
    ax1.set_title('Real-time SpO2 Estimation Benchmarking'); ax1.set_ylabel('SpO2 (%)'); ax1.legend()
    ax1.set_ylim(80, 105)
    
    # Subplot 2: Bar Chart
    ax2.bar(['Baseline', 'OxyVision'], [baseline_mae, constrained_mae], color=['red', 'green'])
    ax2.set_title('Mean Absolute Error Comparison'); ax2.set_ylabel('MAE (%)')
    
    plt.tight_layout(); plt.savefig('final_hackathon_comparison.png')
    print("\nBenchmark visualization saved as 'final_hackathon_comparison.png'")

if __name__ == "__main__":
    run_final_comparison()
