import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.ssim_loss import SSIMLoss


# ======================
# Sobel 梯度算子
# ======================

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        # x: [B,1,H,W] 或 [B,C,H,W]，这里默认单通道输入
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


# ======================
# 基础 loss
# ======================

def L1_loss(tensor1, tensor2):
    loss = nn.L1Loss()
    return loss(tensor1, tensor2)


def r_loss(flow):
    """
    光流平滑正则：一阶梯度的 L2
    flow: [B,2,H,W]
    """
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d / 3.0
    return grad


def ssim_loss(img1, img2):
    """
    MONAI 的 SSIM loss，data_range 设为 1.0（图像已归一化到 [0,1]）
    """
    device = img1.device
    data_range = torch.tensor(1.0).to(device)
    return SSIMLoss(spatial_dims=2, data_range=data_range)(img1, img2)


# ======================
# 现有的梯度（边缘）保持损失（空间域）
# ======================

def gradient_loss(fusion_img, img1, img2):
    """
    使用 Sobel 梯度算子，在梯度域上让 fusion 接近 max(grad(img1), grad(img2))
    """
    grad_filter = Sobelxy().requires_grad_(False).to(fusion_img.device)
    fusion_img_g = grad_filter(fusion_img)
    max_g_img1_2 = torch.maximum(grad_filter(img1), grad_filter(img2))
    return L1_loss(fusion_img_g, max_g_img1_2)


# ======================
# 新增：拉普拉斯高频提取 + 频域保持损失
# ======================

def laplacian_hf(x):
    """
    拉普拉斯核提取高频分量（类似你模型里的 compute_high_freq）:
    x: [B,1,H,W] 或 [B,C,H,W]
    返回: [B,C,H,W] 高频响应
    """
    B, C, H, W = x.shape
    lap_kernel = torch.tensor(
        [[0.,  1., 0.],
         [1., -4., 1.],
         [0.,  1., 0.]],
        device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)  # [1,1,3,3]
    lap_kernel = lap_kernel.repeat(C, 1, 1, 1)      # [C,1,3,3]
    x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
    return F.conv2d(x_pad, lap_kernel, groups=C)


def freq_keep_loss(fusion_img, img1, warped_img2):
    """
    频域保持损失 L_freq_keep:
    - 用 Laplacian 提取三幅图的高频
    - 约束 |HF_fuse| 接近 max(|HF_img1|, |HF_img2_warp|)
    fusion_img:  融合结果 [B,1,H,W]
    img1:        CT 原图  [B,1,H,W]
    warped_img2: MR warp 后图 [B,1,H,W]
    """
    HF_fuse = laplacian_hf(fusion_img)
    HF1 = laplacian_hf(img1)
    HF2 = laplacian_hf(warped_img2)

    HF_max = torch.maximum(torch.abs(HF1), torch.abs(HF2))
    return L1_loss(torch.abs(HF_fuse), HF_max)


# ======================
# 总损失 regFusion_loss（已集成 L_freq_keep）
# ======================

def regFusion_loss(label1, label2,
                   pre1, pre2,
                   feature_pred1, feature_pred2,
                   flow, flows, warped_img2, flow_GT,
                   img1, img1_2, fusion_img,
                   parameter,
                   lambda_freq=0.5):
    """
    参数解释:
        label1, label2       : 分类标签 (CT/MR 对应模态标签)
        pre1, pre2           : 分类网络输出 (logits)
        feature_pred1,2      : 模态迁移后的 domain classifier 输出
        flow, flows          : 最终 flow 及多尺度 flow 列表
        warped_img2          : MR warp 到 CT 空间后的图像
        flow_GT              : (如果有) 真实配准场，可选
        img1                 : CT 原图
        img1_2               : Encoder 第二层特征重构图（或对齐参考图，根据你原来的定义）
        fusion_img           : 最终融合图像
        parameter            : 原代码中 SSIM(fusion, warped_img2) 的权重
        lambda_freq          : 频域保持损失的权重（默认 0.5，可在训练脚本中调）
    返回:
        cls_loss, transfer_loss, flow_loss, fu_loss, reg_loss, ssim1, ssim2
    """

    # -------- 1) 分类 / 迁移相关 --------
    # 分类损失：两模态各自的分类
    cls_loss = (nn.CrossEntropyLoss()(pre1, label1) +
                nn.CrossEntropyLoss()(pre2, label2))

    # 模态迁移后的 domain classifier 输出，希望趋近于 [0.5, 0.5]（模态不可分）
    trans_label = torch.tensor([0.5, 0.5]).expand(feature_pred1.shape[0], -1).to(feature_pred1.device)
    transfer_loss = (nn.CrossEntropyLoss()(feature_pred1, trans_label) +
                     nn.CrossEntropyLoss()(feature_pred2, trans_label))

    # -------- 2) flow 平滑正则 --------
    flow_loss = torch.tensor(0.0, device=feature_pred1.device)
    alpha = 0.0001
    for i in range(len(flows) // 2):
        flow_loss = flow_loss + (r_loss(flows[i]) + r_loss(flows[i + 1])) * alpha
        alpha *= 10
    flow_loss = flow_loss + r_loss(flow)

    # 如果你有 flow_GT，这里也可以增加监督项：L1(flow, flow_GT) or endpoint error
    # 这里暂时沿用你原来的定义，不额外加

    # -------- 3) 融合相关损失（空间域 + 频域） --------
    # SSIM：fusion 分别逼近 CT 和 warp 后 MR
    ssim1 = ssim_loss(fusion_img, img1)
    ssim2 = ssim_loss(fusion_img, warped_img2)

    # 空间域：像素级 max 保持 + 梯度 max 保持
    l1_max = L1_loss(fusion_img, torch.maximum(img1, warped_img2))
    grad_keep = gradient_loss(fusion_img, img1, warped_img2)

    # 新增：频域保持损失（高频 max 保持）
    L_freq = freq_keep_loss(fusion_img, img1, warped_img2)

    # 融合总损失 fu_loss
    fu_loss = (ssim1 +
               parameter * ssim2 +
               0.5 * l1_max +
               grad_keep +
               lambda_freq * L_freq)

    # -------- 4) 配准重建损失（对齐质量） --------
    # reg_loss 趋向于让 img1_2 和 warped_img2 在结构上对齐
    reg_loss = ssim_loss(img1_2, warped_img2) + L1_loss(img1_2, warped_img2)

    return cls_loss, transfer_loss, flow_loss, fu_loss, reg_loss, ssim1, ssim2
