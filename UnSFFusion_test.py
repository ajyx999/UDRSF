import os
import warnings
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from dataset.BrainDataset_2D import TestData
from utils_2d.warp import warp2D

from modal_2d.model import (
    Encoder,
    ModelTransfer_lite,
    RegNet_lite,
    FusionNet_FreqUnc,
    project,
)

from utils_2d.utils import rgb2ycbcr, ycbcr2rgb

warnings.filterwarnings('ignore')

# Configuration
modal = 'SPECT'  # 'CT' or 'PET' or 'SPECT'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

image_warp = warp2D()

# Load checkpoint
checkpoint_path = './checkpoint'
ckpt_file = os.path.join(checkpoint_path, f'UnSFFusion_{modal}.pkl')
assert os.path.exists(ckpt_file), f'Checkpoint not found: {ckpt_file}'

checkpoint = torch.load(ckpt_file, map_location=device)

encoder = Encoder().to(device)
transfer = ModelTransfer_lite(num_vit=2, num_heads=4, img_size=[256, 256]).to(device)
reg_net = RegNet_lite().to(device)
fusion_net = FusionNet_FreqUnc().to(device)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
transfer.load_state_dict(checkpoint['transfer_state_dict'])
reg_net.load_state_dict(checkpoint['reg_net_state_dict'])
fusion_net.load_state_dict(checkpoint['fusion_net_state_dict'])

encoder.eval()
transfer.eval()
reg_net.eval()
fusion_net.eval()

# Dataset
if modal == 'CT':
    val_dataset = TestData(
        img1_folder=f'./data/testData/{modal}/MRI',
        img2_folder=f'./data/testData/{modal}/{modal}',
        modal=modal
    )
else:
    val_dataset = TestData(
        img1_folder=f'./data/testData/{modal}/MRI',
        img2_folder=f'./data/testData/{modal}/{modal}_RGB',
        modal=modal
    )

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    pin_memory=True,
    shuffle=False,
    num_workers=0
)


def _norm01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.min()
    return x / (x.max() + eps)


@torch.no_grad()
def save_reg_unc_vis(
    save_path: str,
    img_mri: torch.Tensor,
    img_mod_y: torch.Tensor,
    img_mod_y_warp: torch.Tensor,
    flow: torch.Tensor,
    reg_unc: dict,
):
    """
    Save a 2x3 visualization:
    1) MRI
    2) modality(Y)
    3) warped modality(Y)
    4) U_struct overlay on MRI
    5) R_struct overlay on MRI
    6) flow magnitude
    """
    assert img_mri.dim() == 4 and img_mri.shape[0] == 1, "Visualization expects batch_size=1"
    B, _, H, W = img_mri.shape

    U = reg_unc["U_struct"]
    R = reg_unc["R_struct"]

    U_up = F.interpolate(U, size=(H, W), mode="bilinear", align_corners=False)
    R_up = F.interpolate(R, size=(H, W), mode="bilinear", align_corners=False)

    mag = torch.sqrt(flow[:, 0:1] ** 2 + flow[:, 1:2] ** 2)

    mri = _norm01(img_mri[0, 0].float().detach().cpu())
    mod = _norm01(img_mod_y[0, 0].float().detach().cpu())
    modw = _norm01(img_mod_y_warp[0, 0].float().detach().cpu())
    Uv = _norm01(U_up[0, 0].float().detach().cpu())
    Rv = _norm01(R_up[0, 0].float().detach().cpu())
    mv = _norm01(mag[0, 0].float().detach().cpu())

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.title("MRI")
    plt.imshow(mri, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title(f"{modal} (Y)")
    plt.imshow(mod, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title(f"Warped {modal} (Y)")
    plt.imshow(modw, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("U_struct overlay on MRI")
    plt.imshow(mri, cmap="gray")
    plt.imshow(Uv, alpha=0.55, cmap="jet")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("R_struct overlay on MRI")
    plt.imshow(mri, cmap="gray")
    plt.imshow(Rv, alpha=0.55, cmap="jet")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("|flow| magnitude")
    plt.imshow(mv, cmap="jet")
    plt.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def validate_mask(encoder, transfer, reg_net, fusion_net, dataloader, modal, device):
    epoch_iterator = tqdm(dataloader, desc='Val', ncols=150, leave=True, position=0)

    figure_save_path = f"./fusion_results/{modal}_result"
    os.makedirs(os.path.join(figure_save_path, "MRI"), exist_ok=True)
    os.makedirs(os.path.join(figure_save_path, f"{modal}"), exist_ok=True)
    os.makedirs(os.path.join(figure_save_path, "Fusion"), exist_ok=True)
    os.makedirs(os.path.join(figure_save_path, f"{modal}_align"), exist_ok=True)
    os.makedirs(os.path.join(figure_save_path, f"{modal}_label"), exist_ok=True)
    os.makedirs(os.path.join(figure_save_path, "RegUncVis"), exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(epoch_iterator):
            img1, img2, file_name = batch
            img1 = img1.to(device)
            img2 = img2.to(device)
            H, W = img1.shape[2], img1.shape[3]

            if modal != 'CT':
                img2_ycbcr = rgb2ycbcr(img2)
                img2_cbcr = img2_ycbcr[:, 1:3, :, :]
                img2_y = img2_ycbcr[:, 0:1, :, :]
            else:
                img2_y = img2

            # Encoder forward
            AS_F, feat1 = encoder(img1)
            BS_F, feat2 = encoder(img2_y)

            # Token interaction
            pre1, pre2, feat_pred1, feat_pred2, feat1_t, feat2_t, AU_F, BU_F = transfer(feat1, feat2)

            # Token to feature map
            feat1_t = project(feat1_t, [H, W]).to(device)
            feat2_t = project(feat2_t, [H, W]).to(device)
            AU_F = project(AU_F, [H, W]).to(device)
            BU_F = project(BU_F, [H, W]).to(device)

            # Registration forward
            _, _, flows, flow, _, _ = reg_net(feat1_t, feat2_t)

            # Warp the second modality image
            warped_img2_y = image_warp(img2_y, flow)

            # Save registration uncertainty visualization
            reg_unc = reg_net.last_unc
            name = file_name[0] if isinstance(file_name, (list, tuple)) else str(file_name)
            base, ext = os.path.splitext(name)
            vis_name = base + "_regunc.png"
            vis_path = os.path.join(figure_save_path, "RegUncVis", vis_name)

            if reg_unc is not None and ("U_struct" in reg_unc) and ("R_struct" in reg_unc):
                save_reg_unc_vis(
                    save_path=vis_path,
                    img_mri=img1,
                    img_mod_y=img2_y,
                    img_mod_y_warp=warped_img2_y,
                    flow=flow,
                    reg_unc=reg_unc
                )

            # Fusion forward
            fusion_y = fusion_net(AS_F, BS_F, AU_F, BU_F, flow)

            # Save outputs
            if modal != 'CT':
                fusion_cbcr = image_warp(img2_cbcr, flow)
                fusion_ycbcr = torch.cat((fusion_y, fusion_cbcr), dim=1)
                fusion_rgb = ycbcr2rgb(fusion_ycbcr)

                warped_ycbcr = torch.cat((warped_img2_y, fusion_cbcr), dim=1)
                warped_rgb = ycbcr2rgb(warped_ycbcr)
            else:
                fusion_rgb = fusion_y
                warped_rgb = warped_img2_y

            save_image(img1.cpu(), os.path.join(figure_save_path, f"MRI/{name}"))
            save_image(img2_y.cpu(), os.path.join(figure_save_path, f"{modal}/{name}"))
            save_image(fusion_rgb.cpu(), os.path.join(figure_save_path, f"Fusion/{name}"))
            save_image(warped_rgb.cpu(), os.path.join(figure_save_path, f"{modal}_align/{name}"))


if __name__ == '__main__':
    validate_mask(encoder, transfer, reg_net, fusion_net, val_dataloader, modal, device)