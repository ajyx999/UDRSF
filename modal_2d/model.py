import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modal_2d.classifier import VitBlock, PatchEmbedding2D
from utils_2d.warp import warp2D


# ======================
# Basic utilities
# ======================

image_warp = warp2D()  # Usage: image_warp(image, flow)


def project(tokens, image_size):
    """
    Convert ViT tokens [B, N, C] back to a feature map [B, C, H, W].
    Assumes N = (W/16) * (H/16).
    """
    W, H = image_size
    x = rearrange(tokens, 'b (w h) c -> b c w h', w=W // 16, h=H // 16)
    return x


def compute_high_freq(x):
    """
    Extract high-frequency components with a Laplacian filter.
    x: [B, C, H, W]
    """
    lap = torch.tensor(
        [[0.,  1., 0.],
         [1., -4., 1.],
         [0.,  1., 0.]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    lap = lap.repeat(x.shape[1], 1, 1, 1)  # [C,1,3,3]
    x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
    return F.conv2d(x_pad, lap, groups=x.shape[1])


# ======================
# Restormer-style blocks
# ======================

class MDTA(nn.Module):
    """
    Attention block for channel/spatial feature interaction.
    """
    def __init__(self, out_c):
        super(MDTA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        self.conv4 = nn.Conv2d(out_c, out_c, 1, 1, 0)

    def forward(self, x):
        x_o = x
        x = F.layer_norm(x, x.shape[-2:])
        B, C, W, H = x.shape
        q = self.conv1(x)
        q = rearrange(q, 'b c w h -> b (w h) c')   # [B,N,C]

        k = self.conv2(x)
        k = rearrange(k, 'b c w h -> b c (w h)')   # [B,C,N]

        v = self.conv3(x)
        v = rearrange(v, 'b c w h -> b (w h) c')   # [B,N,C]

        A = torch.matmul(k, q)                     # [B,C,C]
        A = rearrange(A, 'b c1 c2 -> b (c1 c2)', c1=C, c2=C)
        A = torch.softmax(A, dim=1)
        A = rearrange(A, 'b (c1 c2) -> b c1 c2', c1=C, c2=C)

        v = torch.matmul(v, A)                     # [B,N,C]
        v = rearrange(v, 'b (h w) c -> b c h w', h=W, w=H, c=C)
        return self.conv4(v) + x_o


class GDFN(nn.Module):
    """
    Feed-forward block for local feature refinement.
    """
    def __init__(self, out_c):
        super(GDFN, self).__init__()
        self.Dconv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c * 4, 1, 1, 0),
            nn.Conv2d(out_c * 4, out_c * 4, 3, 1, 1)
        )
        self.Dconv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c * 4, 1, 1, 0),
            nn.Conv2d(out_c * 4, out_c * 4, 3, 1, 1)
        )
        self.conv = nn.Conv2d(out_c * 4, out_c, 1, 1, 0)

    def forward(self, x):
        x_o = x
        x = F.layer_norm(x, x.shape[-2:])
        x = F.gelu(self.Dconv1(x)) * self.Dconv2(x)
        x = x_o + self.conv(x)
        return x


class Restormer(nn.Module):
    """
    Feature extraction and refinement block.
    """
    def __init__(self, in_c, out_c):
        super(Restormer, self).__init__()
        self.mlp = nn.Conv2d(in_c, out_c, 1, 1, 0)
        self.mdta = MDTA(out_c)
        self.gdfn = GDFN(out_c)

    def forward(self, feature):
        feature = self.mlp(feature)
        feature = self.mdta(feature)
        return self.gdfn(feature)


# ======================
# ViT classification and token interaction
# ======================

class model_classifer_lite(nn.Module):
    """
    Lightweight classifier that directly takes ViT tokens as input.
    Used for domain/modality prediction.
    """
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(model_classifer_lite, self).__init__()
        self.hidden_size = 256
        self.embedding = PatchEmbedding2D(in_c=in_c, embedding_dim=256, patch_size=patch_size)

        self.vit_blks = nn.Sequential()
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(
                    hidden_size=256,
                    num_heads=num_heads,
                    vit_drop=0.1,
                    qkv_bias=False,
                    mlp_dim=256,
                    mlp_drop=0.0
                )
            )
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 2)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        # x: [B, N, C] tokens
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)
        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]


class Classifier_lite(nn.Module):
    """
    Standard ViT classifier with patch embedding.
    """
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(Classifier_lite, self).__init__()
        self.hidden_size = 256
        self.embedding = PatchEmbedding2D(in_c=in_c, embedding_dim=256, patch_size=patch_size)

        self.vit_blks = nn.Sequential()
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(
                    hidden_size=256,
                    num_heads=num_heads,
                    vit_drop=0.1,
                    qkv_bias=False,
                    mlp_dim=256,
                    mlp_drop=0.0
                )
            )
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 2)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        # x: [B, in_c, H, W]
        x = self.embedding(x)  # [B, N, C]
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)
        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]


class Transfer(nn.Module):
    """
    Exchange token-level information between two inputs and update them with ViT blocks.
    """
    def __init__(self, num_vit, num_heads):
        super(Transfer, self).__init__()
        self.num_vit = num_vit
        self.num_heads = num_heads
        self.hidden_dim = 256
        self.cls1 = nn.Parameter(torch.zeros(1, 1, 256))
        self.cls2 = nn.Parameter(torch.zeros(1, 1, 256))

        self.VitBLK1 = nn.Sequential()
        for i in range(self.num_vit):
            self.VitBLK1.add_module(
                name=f'vit{i}',
                module=VitBlock(
                    hidden_size=self.hidden_dim,
                    num_heads=self.num_heads,
                    vit_drop=0.0,
                    qkv_bias=False,
                    mlp_dim=256,
                    mlp_drop=0.0
                )
            )

        self.VitBLK2 = nn.Sequential()
        for i in range(self.num_vit):
            self.VitBLK2.add_module(
                name=f'vit{i}',
                module=VitBlock(
                    hidden_size=self.hidden_dim,
                    num_heads=self.num_heads,
                    vit_drop=0.0,
                    qkv_bias=False,
                    mlp_dim=256,
                    mlp_drop=0.0
                )
            )

    def forward(self, x1, x2, cls1, cls2):
        # x1, x2: [B, N, C]; cls1, cls2: [B, C]
        cls1, cls2 = cls1.unsqueeze(1), cls2.unsqueeze(1)
        cls1 = cls1.expand(-1, x1.shape[1], -1)
        cls2 = cls2.expand(-1, x1.shape[1], -1)

        x1, x2 = x1 + cls2, x2 + cls1

        class_token1 = self.cls1.expand(x1.shape[0], -1, -1)
        class_token2 = self.cls2.expand(x1.shape[0], -1, -1)

        x1 = torch.cat((x1, class_token1), dim=1)
        x2 = torch.cat((x2, class_token2), dim=1)

        x1 = self.VitBLK1(x1)
        x2 = self.VitBLK2(x2)

        class_token1 = x1[:, 0, :]
        class_token2 = x2[:, 0, :]

        return x1[:, 1:, :], x2[:, 1:, :], class_token1, class_token2


# ======================
# Encoder
# ======================

class Encoder(nn.Module):
    """
    Two-stage feature encoder.
    The first stage extracts structural features.
    The second stage produces higher-level features.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.rb1 = Restormer(1, 8)
        self.rb2 = Restormer(8, 3)

    def forward(self, img):
        f = self.rb1(img)    # [B, 8, H, W]
        f_ = self.rb2(f)     # [B, 3, H, W]
        return f, f_


# ======================
# Token interaction wrapper
# ======================

class ModelTransfer_lite(nn.Module):
    """
    Wrapper that performs image classification, token interaction,
    and token-based modality prediction.
    """
    def __init__(self, num_vit, num_heads, img_size):
        super(ModelTransfer_lite, self).__init__()
        self.img_size = img_size
        self.transfer = Transfer(num_vit=num_vit, num_heads=num_heads)
        self.classifier = Classifier_lite(
            in_c=3, num_heads=4, num_vit_blk=2,
            img_size=self.img_size, patch_size=16
        )
        self.modal_dis = model_classifer_lite(
            in_c=3, num_heads=4, num_vit_blk=2,
            img_size=self.img_size, patch_size=16
        )

    def forward(self, img1, img2):
        pre1, cls1, x1_ = self.classifier(img1)
        pre2, cls2, x2_ = self.classifier(img2)

        x1, x2, new_cls1, new_cls2 = self.transfer(x1_, x2_, cls1, cls2)

        feature_pred1, _, _ = self.modal_dis(x1)
        feature_pred2, _, _ = self.modal_dis(x2)

        return pre1, pre2, feature_pred1, feature_pred2, x1, x2, x1_, x2_


# ======================
# Structural uncertainty estimation
# ======================

class StructuralUncertaintyHead2D(nn.Module):
    """
    Estimate mean, variance, and spatial uncertainty from feature maps.
    """
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2 * in_channels, 3, padding=1)
        )

    def forward(self, feat):
        stats = self.conv(feat)               # [B,2C,H,W]
        mu, log_var = stats.chunk(2, dim=1)   # [B,C,H,W], [B,C,H,W]
        log_var = torch.clamp(log_var, -10.0, 10.0)
        sigma = torch.exp(0.5 * log_var)
        U_struct = sigma.mean(dim=1, keepdim=True)
        return mu, sigma, U_struct


class RegNet_lite(nn.Module):
    """
    Registration network that predicts multi-scale flow fields
    and applies uncertainty-guided feature weighting.
    """
    def __init__(self, in_channels=256, base_channels=128, alpha=10.0):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.alpha = alpha

        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.unc_head = StructuralUncertaintyHead2D(in_channels=base_channels)

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels // 2, base_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels // 4, base_channels // 8, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.flow_16 = nn.Conv2d(base_channels, 2, 3, padding=1)
        self.flow_32 = nn.Conv2d(base_channels, 2, 3, padding=1)
        self.flow_64 = nn.Conv2d(base_channels // 2, 2, 3, padding=1)
        self.flow_128 = nn.Conv2d(base_channels // 4, 2, 3, padding=1)
        self.flow_256 = nn.Conv2d(base_channels // 8, 2, 3, padding=1)

        self.last_unc = None

    def forward(self, f1, f2):
        # f1, f2: [B,256,16,16]
        f_cat = torch.cat([f1, f2], dim=1)
        x = self.enc_conv1(f_cat)
        x = self.enc_conv2(x)

        mu_s, sigma_s, U_struct = self.unc_head(x)
        R_struct = torch.sigmoid(-self.alpha * U_struct)
        x = x * R_struct

        self.last_unc = {
            "mu": mu_s,
            "sigma": sigma_s,
            "U_struct": U_struct,
            "R_struct": R_struct,
        }

        x16 = x
        flow16 = self.flow_16(x16)

        x32 = self.dec1(x16)
        flow32 = self.flow_32(x32)

        x64 = self.dec2(x32)
        flow64 = self.flow_64(x64)

        x128 = self.dec3(x64)
        flow128 = self.flow_128(x128)

        x256 = self.dec4(x128)
        flow256 = self.flow_256(x256)

        flow16_up = F.interpolate(flow16, size=(256, 256), mode='bilinear', align_corners=True)
        flow32_up = F.interpolate(flow32, size=(256, 256), mode='bilinear', align_corners=True)
        flow64_up = F.interpolate(flow64, size=(256, 256), mode='bilinear', align_corners=True)
        flow128_up = F.interpolate(flow128, size=(256, 256), mode='bilinear', align_corners=True)
        flow256_up = flow256

        flow = (flow16_up + flow32_up + flow64_up + flow128_up + flow256_up) / 5.0

        flows = [
            flow16, flow16,
            flow32, flow32,
            flow64, flow64,
            flow128, flow128,
            flow256, flow256,
        ]
        flow_neg = torch.zeros_like(flow)
        flow_pos = flow

        if self.training:
            # Keep the original low-resolution features during training
            # to reduce memory usage.
            f1_reg, f2_reg = f1, f2
        else:
            # During inference, upsample and warp the feature maps.
            f1_up = F.interpolate(f1, size=(256, 256), mode='bilinear', align_corners=True)
            f2_up = F.interpolate(f2, size=(256, 256), mode='bilinear', align_corners=True)
            f2_reg = image_warp(f2_up, flow)
            f1_reg = f1_up

        return f1_reg, f2_reg, flows, flow, flow_neg, flow_pos


# ======================
# Frequency uncertainty and channel weighting
# ======================

class FreqUncHead2D(nn.Module):
    """
    Predict mean and variance statistics from high-frequency features.
    """
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2 * in_channels, 3, padding=1)
        )

    def forward(self, hf):
        stats = self.conv(hf)
        mu, log_var = stats.chunk(2, dim=1)
        log_var = torch.clamp(log_var, -10.0, 10.0)
        log_sigma = 0.5 * log_var
        return mu, log_sigma


class FreqRestormerFuse(nn.Module):
    """
    Fuse two feature maps with high-frequency uncertainty estimation
    and channel-wise weighting.
    """
    def __init__(self, in_c, alpha_f=5.0, reduction=16):
        super().__init__()
        self.in_c = in_c
        self.alpha_f = alpha_f

        self.unc_ct = FreqUncHead2D(in_c)
        self.unc_mr = FreqUncHead2D(in_c)

        hidden = max((2 * in_c) // reduction, 1)
        self.ca_fc1 = nn.Linear(2 * in_c, hidden)
        self.ca_fc2 = nn.Linear(hidden, 2 * in_c)

        self.last_info = None

    def forward(self, F_ct, F_mr):
        B, C, H, W = F_ct.shape

        # 1) Extract high-frequency information
        HF_ct = compute_high_freq(F_ct)
        HF_mr = compute_high_freq(F_mr)

        # 2) Estimate uncertainty statistics
        mu_ct, log_sigma_ct = self.unc_ct(HF_ct)
        mu_mr, log_sigma_mr = self.unc_mr(HF_mr)

        sigma_ct = torch.exp(log_sigma_ct)
        sigma_mr = torch.exp(log_sigma_mr)

        # 3) Aggregate uncertainty at the channel level
        U_ct_c = sigma_ct.mean(dim=[2, 3])  # [B,C]
        U_mr_c = sigma_mr.mean(dim=[2, 3])  # [B,C]
        U_all = torch.cat([U_ct_c, U_mr_c], dim=1)  # [B,2C]

        # 4) Convert uncertainty to channel weights
        U_norm = U_all / (U_all.mean(dim=1, keepdim=True) + 1e-6)
        R_all = torch.exp(-self.alpha_f * U_norm)

        z = F.relu(self.ca_fc1(R_all))
        w_all = torch.sigmoid(self.ca_fc2(z))  # [B,2C]
        w_all = w_all.view(B, 2 * C, 1, 1)

        # 5) Apply channel weighting and residual enhancement
        X = torch.cat([F_ct, F_mr], dim=1)     # [B,2C,H,W]
        X_weighted = X * w_all
        X_enh = X + X_weighted

        F_ct_enh, F_mr_enh = torch.chunk(X_enh, 2, dim=1)

        self.last_info = {
            "HF_ct": HF_ct,
            "HF_mr": HF_mr,
            "mu_ct": mu_ct,
            "mu_mr": mu_mr,
            "log_sigma_ct": log_sigma_ct,
            "log_sigma_mr": log_sigma_mr,
            "U_ct_c": U_ct_c,
            "U_mr_c": U_mr_c,
            "U_all": U_all,
            "R_all": R_all,
            "w_all": w_all,
        }

        return F_ct_enh, F_mr_enh


# ======================
# Multi-scale fusion network
# ======================

class UpSampler_V2(nn.Module):
    """
    Upsample two feature branches and one fused feature branch together.
    """
    def __init__(self, in_c, out_c):
        super(UpSampler_V2, self).__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, AU_F, BU_F, feature):
        AU_F = self.up1(AU_F)
        BU_F = self.up1(BU_F)
        feature = self.up3(feature)
        return AU_F, BU_F, feature


class FusionNet_FreqUnc(nn.Module):
    """
    Multi-scale fusion network with frequency-based channel weighting.
    """
    def __init__(self):
        super(FusionNet_FreqUnc, self).__init__()
        self.cn = [256, 64, 32, 16, 12, 8]

        self.freq_fuse1 = FreqRestormerFuse(in_c=self.cn[0])  # 16x16
        self.freq_fuse2 = FreqRestormerFuse(in_c=self.cn[1])  # 32x32
        self.freq_fuse3 = FreqRestormerFuse(in_c=self.cn[2])  # 64x64
        self.freq_fuse4 = FreqRestormerFuse(in_c=self.cn[3])  # 128x128

        self.F1 = Restormer(in_c=self.cn[0] * 2, out_c=self.cn[1])
        self.up_sample1 = UpSampler_V2(in_c=self.cn[0], out_c=self.cn[1])

        self.F2 = Restormer(in_c=self.cn[1] * 3, out_c=self.cn[2])
        self.up_sample2 = UpSampler_V2(in_c=self.cn[1], out_c=self.cn[2])

        self.F3 = Restormer(in_c=self.cn[2] * 3, out_c=self.cn[3])
        self.up_sample3 = UpSampler_V2(in_c=self.cn[2], out_c=self.cn[3])

        self.F4 = Restormer(in_c=self.cn[3] * 3, out_c=self.cn[4])
        self.up_sample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.outLayer = nn.Sequential(
            Restormer(in_c=self.cn[4] + 16, out_c=self.cn[4]),
            Restormer(in_c=self.cn[4], out_c=1),
            nn.Sigmoid()
        )

        self.last_freq_info = None

    def forward(self, AS_F, BS_F, AU_F, BU_F, flow):
        freq_infos = []

        # Scale 1: 16x16
        flow_d = F.interpolate(flow, size=BU_F.size()[2:4], mode='bilinear', align_corners=True) / 16
        BU_F_w = image_warp(BU_F, flow_d)

        AU_F_enh, BU_F_w_enh = self.freq_fuse1(AU_F, BU_F_w)
        freq_infos.append(self.freq_fuse1.last_info)

        feature = self.F1(torch.cat((AU_F_enh, BU_F_w_enh), dim=1))

        # Scale 2: 32x32
        AU_F, BU_F, feature = self.up_sample1(AU_F_enh, BU_F_w_enh, feature)
        flow_d = F.interpolate(flow, size=BU_F.size()[2:4], mode='bilinear', align_corners=True) / 8
        BU_F_w = image_warp(BU_F, flow_d)

        AU_F_enh, BU_F_w_enh = self.freq_fuse2(AU_F, BU_F_w)
        freq_infos.append(self.freq_fuse2.last_info)

        feature = self.F2(torch.cat([torch.cat([AU_F_enh, BU_F_w_enh], dim=1), feature], dim=1))

        # Scale 3: 64x64
        AU_F, BU_F, feature = self.up_sample2(AU_F_enh, BU_F_w_enh, feature)
        flow_d = F.interpolate(flow, size=BU_F.size()[2:4], mode='bilinear', align_corners=True) / 4
        BU_F_w = image_warp(BU_F, flow_d)

        AU_F_enh, BU_F_w_enh = self.freq_fuse3(AU_F, BU_F_w)
        freq_infos.append(self.freq_fuse3.last_info)

        feature = self.F3(torch.cat([torch.cat([AU_F_enh, BU_F_w_enh], dim=1), feature], dim=1))

        # Scale 4: 128x128
        AU_F, BU_F, feature = self.up_sample3(AU_F_enh, BU_F_w_enh, feature)
        flow_d = F.interpolate(flow, size=BU_F.size()[2:4], mode='bilinear', align_corners=True) / 2
        BU_F_w = image_warp(BU_F, flow_d)

        AU_F_enh, BU_F_w_enh = self.freq_fuse4(AU_F, BU_F_w)
        freq_infos.append(self.freq_fuse4.last_info)

        feature = self.F4(torch.cat([torch.cat([AU_F_enh, BU_F_w_enh], dim=1), feature], dim=1))

        # Final fusion at full resolution
        feature = self.up_sample4(feature)
        BS_F_w = image_warp(BS_F, flow)
        S_F = torch.cat([AS_F, BS_F_w], dim=1)
        fusion = self.outLayer(torch.cat([feature, S_F], dim=1))

        self.last_freq_info = freq_infos
        return fusion


# ======================
# Full network
# ======================

class CTMR_UncFusionNet(nn.Module):
    """
    Full CT/MR processing pipeline including:
    feature extraction, token interaction, registration, and fusion.
    """
    def __init__(self, img_size=256):
        super().__init__()
        self.encoder = Encoder()
        self.transfer = ModelTransfer_lite(num_vit=2, num_heads=4, img_size=[img_size, img_size])
        self.reg_net = RegNet_lite()
        self.fusion_net = FusionNet_FreqUnc()
        self.img_size = img_size

    def forward(self, img_ct, img_mr):
        # Extract low-level and high-level features
        AS_F, feat_ct = self.encoder(img_ct)  # [B,8,256,256], [B,3,256,256]
        BS_F, feat_mr = self.encoder(img_mr)

        # Perform token interaction and classification
        pre1, pre2, feature_pred1, feature_pred2, feat1_tok, feat2_tok, AU_F_tok, BU_F_tok = \
            self.transfer(feat_ct, feat_mr)

        # Convert tokens back to feature maps
        F_ct_256 = project(feat1_tok, [self.img_size, self.img_size])
        F_mr_256 = project(feat2_tok, [self.img_size, self.img_size])
        AU_F = project(AU_F_tok, [self.img_size, self.img_size])
        BU_F = project(BU_F_tok, [self.img_size, self.img_size])

        # Estimate alignment flow
        f1_reg, f2_reg, flows, flow, flow_neg, flow_pos = self.reg_net(F_ct_256, F_mr_256)

        # Fuse features into the final image
        fusion_img = self.fusion_net(AS_F, BS_F, AU_F, BU_F, flow)

        # Warp the MR image with the predicted flow
        img_mr_warp = image_warp(img_mr, flow)

        return {
            "fusion": fusion_img,
            "flow": flow,
            "flows": flows,
            "img_mr_warp": img_mr_warp,
            "reg_unc": self.reg_net.last_unc,
            "pre1": pre1,
            "pre2": pre2,
            "feature_pred1": feature_pred1,
            "feature_pred2": feature_pred2,
            "freq_unc": self.fusion_net.last_freq_info,
        }


if __name__ == "__main__":
    B, C, H, W = 1, 1, 256, 256
    img_ct = torch.rand(B, C, H, W)
    img_mr = torch.rand(B, C, H, W)

    model = CTMR_UncFusionNet(img_size=256)
    out = model(img_ct, img_mr)
    print("fusion:", out["fusion"].shape)
    print("flow  :", out["flow"].shape)