import torch
import torch.nn as nn
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX as PhysNet

class PhysNetBaseline(PhysNet):
    """
    Standard PhysNet model without physiological constraints.
    Outputs raw linear values for data-driven regression.
    """
    def __init__(self, frames=128):
        super(PhysNetBaseline, self).__init__(frames=frames)
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

    def forward(self, x):
        x_visual = x
        [batch, channel, length, width, height] = x.shape
        x = self.ConvBlock1(x); x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x); x_visual6464 = self.ConvBlock3(x); x = self.MaxpoolSpaTem(x_visual6464)
        x = self.ConvBlock4(x); x_visual3232 = self.ConvBlock5(x); x = self.MaxpoolSpaTem(x_visual3232)
        x = self.ConvBlock6(x); x_visual1616 = self.ConvBlock7(x); x = self.MaxpoolSpa(x_visual1616)
        x = self.ConvBlock8(x); x = self.ConvBlock9(x)
        x = self.upsample(x); x = self.upsample2(x)
        x = self.poolspa(x)
        x = self.ConvBlock10(x)
        rPPG = x.view(-1, length)
        return rPPG, x_visual, x_visual3232, x_visual1616

class PhysNetConstrained(PhysNet):
    """
    PhysNet model with a physiological constraint on the output.
    Inherits from PhysNet and overwrites the final layer to output SpO2.
    Bounds predictions between 80 and 100 using a scaled Sigmoid function.
    """
    def __init__(self, frames=128):
        super(PhysNetConstrained, self).__init__(frames=frames)
        # Overwrite the final layer: nn.Conv3d(64, 1, [1, 1, 1])
        # This acts as a pointwise linear layer on the 64 feature maps
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass with SpO2 range constraint.
        
        Args:
            x (torch.Tensor): Input video frames [Batch, 3, T, H, W].
            
        Returns:
            spo2_constrained (torch.Tensor): SpO2 signal bounded in [80, 100].
            x_visual, x_visual3232, x_visual1616: Intermediate features for visualization/loss.
        """
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        # Standard PhysNet backbone
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock2(x)
        x_visual6464 = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)
        x_visual3232 = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x_visual3232)

        x = self.ConvBlock6(x)
        x_visual1616 = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x_visual1616)

        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.upsample(x)
        x = self.upsample2(x)

        # Global spatial pooling to get (Batch, 64, T, 1, 1)
        x = self.poolspa(x)
        
        # Final prediction layer
        x = self.ConvBlock10(x)  # shape: (Batch, 1, T, 1, 1)
        
        # Flatten to (Batch, T)
        raw_output = x.view(-1, length)
        
        # Apply scaled Sigmoid to force the output to be between 80 and 100.
        # Formula: 80 + 20 * sigmoid(x)
        spo2_constrained = 80 + 20 * torch.sigmoid(raw_output)
        
        # Return constrained output and visualization features as expected by toolbox
        return spo2_constrained, x_visual, x_visual3232, x_visual1616
