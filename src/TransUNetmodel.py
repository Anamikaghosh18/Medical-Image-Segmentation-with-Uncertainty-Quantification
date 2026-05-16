import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class TransformerBottleneck(nn.Module):
    """
    THE CORE DIFFERENCE: Replaces the CNN bottleneck with Self-Attention.
    Instead of passing a 3x3 local filter over the image, this allows every pixel 
    in the bottleneck to "attend" (look at) every other pixel simultaneously to understand global context.
    """
    def __init__(self, dim=1024, num_heads=8):
        super().__init__()
        # Standard Multi-Head Self Attention layer
        self.attention = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=num_heads, 
            dim_feedforward=dim * 4,
            batch_first=True,
            activation='gelu'
        )
        
    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        B, C, H, W = x.shape
        
        # 1. Flatten the image into a sequence of "tokens" (like words in a sentence)
        # Shape becomes (Batch, Height*Width, Channels)
        x_seq = x.view(B, C, H * W).permute(0, 2, 1) 
        
        # 2. Apply Global Self-Attention
        x_seq = self.attention(x_seq)
        
        # 3. Fold back into a 2D image map
        # Shape becomes (Batch, Channels, Height, Width)
        x_out = x_seq.permute(0, 2, 1).view(B, C, H, W)
        
        return x_out

class TransUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # --- CNN Encoder (Extracts local textures/edges) ---
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # --- CNN to Transformer Transition ---
        self.pre_bottleneck = DoubleConv(512, 1024)
        
        # --- TRANSFORMER BOTTLENECK (Extracts global relationships) ---
        self.bottleneck = TransformerBottleneck(dim=1024, num_heads=8)
        
        # --- CNN Decoder (Reconstructs high resolution map) ---
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Pre-bottleneck CNN
        b_cnn = self.pre_bottleneck(self.pool4(e4))
        
        # Transformer Bottleneck
        b_trans = self.bottleneck(b_cnn)
        
        # Decoder (with skip connections)
        d4 = self.up4(b_trans)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)

if __name__ == "__main__":
    model = TransUNet()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("TransUNet output shape:", y.shape)
