import torch
import torch.nn as nn

class PhysiologicalLoss(nn.Module):
    """
    PhysiologicalLoss: A custom loss function for SpO2/rPPG estimation.
    Calculates MSE and adds a temporal smoothness penalty.
    Formula: L_total = L_MSE + lambda_smooth * L_smooth + lambda_freq * L_freq
    """
    def __init__(self, lambda_smooth=1.0, lambda_freq=0.0):
        super(PhysiologicalLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_smooth = lambda_smooth
        self.lambda_freq = lambda_freq

    def forward(self, preds, labels):
        """
        Calculates the combined loss.
        
        Args:
            preds (torch.Tensor): Predicted signal of shape (Batch, Time).
            labels (torch.Tensor): Ground truth signal of shape (Batch, Time).
            
        Returns:
            loss (torch.Tensor): The weighted sum of MSE, smoothness, and frequency losses.
        """
        # 1. MSE Loss (Data-driven loss)
        l_mse = self.mse(preds, labels)
        
        # 2. Smoothness Loss (Temporal constraint)
        # Penalize large differences between consecutive time steps to avoid jitter.
        # L_smooth = mean(abs(preds[t] - preds[t-1]))
        if preds.shape[1] > 1:
            diff = preds[:, 1:] - preds[:, :-1]
            l_smooth = torch.mean(torch.abs(diff))
        else:
            l_smooth = torch.tensor(0.0, device=preds.device)
            
        # 3. Frequency Loss (Place holder/Constraint)
        # In a full implementation, this might involve FFT-based penalties
        # for predictions outside the human heart rate range (0.7 - 3.0 Hz).
        l_freq = torch.tensor(0.0, device=preds.device)
        
        # Total combined loss
        loss = l_mse + self.lambda_smooth * l_smooth + self.lambda_freq * l_freq
        
        return loss
