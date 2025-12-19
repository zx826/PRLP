import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelFusionBlock(nn.Module):
    """
    A block to fuse multi-channel features using spatial and channel attention.
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelFusionBlock, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x_ca = x * channel_att

        # Spatial Attention
        avg_map = torch.mean(x_ca, dim=1, keepdim=True)
        max_map, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv_spatial(torch.cat([avg_map, max_map], dim=1)))
        x_sa = x_ca * spatial_att
        return x_sa

class FusionNet(nn.Module):
    """
    Encoder-Decoder network to fuse 6-channel input into a 3-channel output.
    Uses ChannelFusionBlock for strong feature fusion.
    """
    def __init__(self, in_channels=6, out_channels=3, base_filters=64):
        super(FusionNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            ChannelFusionBlock(base_filters)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            ChannelFusionBlock(base_filters*2)
        )

        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            ChannelFusionBlock(base_filters)
        )

        # Final output conv
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)      # B x C x H x W
        p1 = self.pool1(e1)    # B x C x H/2 x W/2
        e2 = self.enc2(p1)     # B x 2C x H/2 x W/2
        d1 = self.up1(e2)      # B x C x H x W
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return out



class StConv(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, expansion=16):
        super(StConv, self).__init__()
        self.reduce = nn.Conv2d(in_channels, expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expansion)
        self.act = nn.ReLU(inplace=True)
        
        # 通道注意力（Squeeze-and-Excitation）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expansion, expansion // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(expansion // 4, expansion, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.project = nn.Conv2d(expansion, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.reduce(x)       # (B, 6, H, W) -> (B, expansion, H, W)
        x = self.bn1(x)
        x = self.act(x)
        
        se_weight = self.se(x)   # 通道注意力
        x = x * se_weight
        
        out = self.project(x)    # (B, expansion, H, W) -> (B, 3, H, W)
        return out
# if __name__ == '__main__':
#     # test
#     model = FusionNet()
#     dummy = torch.randn(2, 6, 256, 256)
#     out = model(dummy)
#     print(out.shape)  # should be (2, 3, 128, 128)
