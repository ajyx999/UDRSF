import os
import warnings
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.BrainDataset_2D import RegDataset_F
from modal_2d.model import (
    Encoder,
    ModelTransfer_lite,
    RegNet_lite,
    FusionNet_FreqUnc,
    project,
)
from utils_2d.loss import (
    regFusion_loss,
    freq_keep_loss,
)
from utils_2d.warp import warp2D
from colorama import Style

warnings.filterwarnings('ignore')
print(f'{Style.RESET_ALL}')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
image_warp = warp2D()


def train(modal,
          train_batch_size,
          lr,
          num_epoch,
          beta1,
          beta2,
          resume):

    checkpoint_root = './checkpoint'
    os.makedirs(checkpoint_root, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    train_dataset = RegDataset_F(
        root='./data',
        mode='train',
        model=modal,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=64
    )

    img_size = 256
    encoder = Encoder().to(device)
    transfer = ModelTransfer_lite(num_vit=2, num_heads=4, img_size=[img_size, img_size]).to(device)
    reg_net = RegNet_lite().to(device)
    fusion_net = FusionNet_FreqUnc().to(device)

    for par in transfer.modal_dis.parameters():
        par.requires_grad = False

    optimizer = Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer1 = Adam(transfer.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer2 = Adam(reg_net.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer3 = Adam(fusion_net.parameters(), lr=lr, betas=(beta1, beta2))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr * 1e-2)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epoch, eta_min=lr * 1e-2)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epoch, eta_min=lr * 1e-2)
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=num_epoch, eta_min=lr * 1e-2)

    epoch_loss_values = []
    align_loss_values = []
    flow_loss_values = []
    fusion_loss_values = []
    reg_loss_values = []

    cls_loss_value = []
    transfer_loss_value = []
    freq_keep_values = []

    start_epoch = 0

    best_total_loss = float('inf')
    best_epoch = -1
    best_ckpt_path = os.path.join(checkpoint_root, f'UnSFFusion_best_{modal}.pkl')

    if resume:
        ckpt_dir = "./checkpoint/checkpoint/"
        ckpt_file = os.path.join(ckpt_dir, "checkpoint_123.pkl")
        assert os.path.exists(ckpt_file), f"checkpoint not found: {ckpt_file}"
        ckpt = torch.load(ckpt_file, map_location='cpu')

        encoder.load_state_dict(ckpt["encoder_state_dict"])
        transfer.load_state_dict(ckpt["transfer_state_dict"])
        reg_net.load_state_dict(ckpt["reg_net_state_dict"])
        fusion_net.load_state_dict(ckpt["fusion_net_state_dict"])

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        optimizer1.load_state_dict(ckpt["optimizer1_state_dict"])
        optimizer2.load_state_dict(ckpt["optimizer2_state_dict"])
        optimizer3.load_state_dict(ckpt["optimizer3_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        scheduler.last_epoch = start_epoch
        scheduler1.last_epoch = start_epoch
        scheduler2.last_epoch = start_epoch
        scheduler3.last_epoch = start_epoch

        epoch_loss_values = ckpt["epoch_loss_values"]
        cls_loss_value = ckpt["cls_loss_value"]
        transfer_loss_value = ckpt["transfer_loss_value"]
        flow_loss_values = ckpt["flow_loss_value"]
        fusion_loss_values = ckpt["fusion_loss_value"]
        reg_loss_values = ckpt["reg_loss_values"]
        align_loss_values = cls_loss_value
        freq_keep_values = ckpt.get("freq_keep_values", [])

        if len(epoch_loss_values) > 0:
            best_total_loss = min(epoch_loss_values)
            best_epoch = int(epoch_loss_values.index(best_total_loss)) + 1

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    def update_plot_and_save():
        ax.clear()
        ax.plot(epoch_loss_values, label='Total Loss')
        ax.plot(align_loss_values, label='Align Loss (cls)')
        ax.plot(fusion_loss_values, label='Fusion Loss')
        ax.plot(flow_loss_values, label='Flow Loss')
        ax.plot(reg_loss_values, label='Reg Loss')
        if len(freq_keep_values) > 0:
            ax.plot(freq_keep_values, label='Freq Keep Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curves')
        ax.legend()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        fig.savefig(os.path.join(checkpoint_root, "loss_curve.png"))

    for epoch in range(start_epoch, num_epoch):
        encoder.train()
        transfer.train()
        reg_net.train()
        fusion_net.train()

        epoch_total = 0.0
        epoch_cls = 0.0
        epoch_transfer = 0.0
        epoch_flow = 0.0
        epoch_fusion = 0.0
        epoch_reg = 0.0
        epoch_freq_keep = 0.0

        parameter = 1.0
        lambda_freq = 0.5

        pbar = tqdm(train_dataloader, ncols=150, desc=f"Train Epoch ({epoch + 1}/{num_epoch})")
        step = 0

        for step, batch in enumerate(pbar):
            img1, img1_2, img2, flow_GT, label1, label2 = batch

            if modal == 'CT':
                img1 = img1.to(device)
                img2 = img2.to(device)
                img1_2 = img1_2.to(device)
            else:
                img1 = img1.to(device)
                img2 = img2[:, 0:1].to(device)
                img1_2 = img1_2[:, 0:1].to(device)

            flow_GT = flow_GT.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)

            AS_F, feat_ct = encoder(img1)
            BS_F, feat_mr = encoder(img2)

            (
                pre1, pre2,
                feat_pred1, feat_pred2,
                feat1_tok, feat2_tok,
                AU_F_tok, BU_F_tok
            ) = transfer(feat_ct, feat_mr)

            F_ct_256 = project(feat1_tok, [img_size, img_size]).to(device)
            F_mr_256 = project(feat2_tok, [img_size, img_size]).to(device)
            AU_F = project(AU_F_tok, [img_size, img_size]).to(device)
            BU_F = project(BU_F_tok, [img_size, img_size]).to(device)

            f1_reg, f2_reg, flows, flow, flow_neg, flow_pos = reg_net(F_ct_256, F_mr_256)

            fusion_img = fusion_net(AS_F, BS_F, AU_F, BU_F, flow)

            warped_img2 = image_warp(img2, flow)

            (
                cls_loss,
                transfer_loss,
                flow_loss,
                fusion_loss,
                reg_loss,
                ssim1,
                ssim2
            ) = regFusion_loss(
                label1, label2,
                pre1, pre2,
                feat_pred1, feat_pred2,
                flow, flows, warped_img2, flow_GT,
                img1, img1_2, fusion_img,
                parameter,
                lambda_freq=lambda_freq
            )

            L_freq_keep = freq_keep_loss(fusion_img, img1, warped_img2)

            loss = cls_loss + transfer_loss + flow_loss + fusion_loss + reg_loss

            epoch_total += loss.item()
            epoch_cls += cls_loss.item()
            epoch_transfer += transfer_loss.item()
            epoch_flow += flow_loss.item()
            epoch_fusion += fusion_loss.item()
            epoch_reg += reg_loss.item()
            epoch_freq_keep += L_freq_keep.item()

            optimizer.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epoch} "
                f"loss={loss.item():.4f} "
                f"align={cls_loss.item():.4f} "
                f"fusion={fusion_loss.item():.4f} "
                f"flow={flow_loss.item():.4f} "
                f"L_freq={L_freq_keep.item():.4f}"
            )

        if step == 0:
            continue

        mean_total = epoch_total / step
        mean_cls = epoch_cls / step
        mean_transfer = epoch_transfer / step
        mean_flow = epoch_flow / step
        mean_fusion = epoch_fusion / step
        mean_reg = epoch_reg / step
        mean_freq_keep = epoch_freq_keep / step

        epoch_loss_values.append(mean_total)
        align_loss_values.append(mean_cls)
        transfer_loss_value.append(mean_transfer)
        flow_loss_values.append(mean_flow)
        fusion_loss_values.append(mean_fusion)
        reg_loss_values.append(mean_reg)
        cls_loss_value.append(mean_cls)
        freq_keep_values.append(mean_freq_keep)

        if mean_total < best_total_loss:
            best_total_loss = mean_total
            best_epoch = epoch + 1
            best_ckpt = {
                "encoder_state_dict": encoder.state_dict(),
                "transfer_state_dict": transfer.state_dict(),
                "reg_net_state_dict": reg_net.state_dict(),
                "fusion_net_state_dict": fusion_net.state_dict(),
                "epoch": epoch,
                "best_total_loss": best_total_loss,
            }
            torch.save(best_ckpt, best_ckpt_path)
            print(f"[Best] Update @ Epoch {best_epoch}: best_total_loss={best_total_loss:.6f} -> saved to {best_ckpt_path}")

        print(
            f"[Epoch {epoch + 1}/{num_epoch}] "
            f"total={mean_total:.4f}  "
            f"align={mean_cls:.4f}  "
            f"fusion={mean_fusion:.4f}  "
            f"flow={mean_flow:.4f}  "
            f"reg={mean_reg:.4f}  "
            f"L_freq_keep={mean_freq_keep:.4f}"
        )

        scheduler.step()
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

        transfer.modal_dis.load_state_dict(transfer.classifier.state_dict())

        if (epoch + 1) % 5 == 0:
            ckpt_dir = "./checkpoint/checkpoint/"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = {
                "encoder_state_dict": encoder.state_dict(),
                "transfer_state_dict": transfer.state_dict(),
                "reg_net_state_dict": reg_net.state_dict(),
                "fusion_net_state_dict": fusion_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer1_state_dict": optimizer1.state_dict(),
                "optimizer2_state_dict": optimizer2.state_dict(),
                "optimizer3_state_dict": optimizer3.state_dict(),
                "epoch": epoch,
                "epoch_loss_values": epoch_loss_values,
                "cls_loss_value": cls_loss_value,
                "transfer_loss_value": transfer_loss_value,
                "flow_loss_value": flow_loss_values,
                "fusion_loss_value": fusion_loss_values,
                "reg_loss_values": reg_loss_values,
                "freq_keep_values": freq_keep_values,
            }
            torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_123.pkl"))

        update_plot_and_save()

    plt.ioff()
    fig.savefig(os.path.join(checkpoint_root, "loss_curve.png"))
    plt.show()

    final_ckpt = {
        "encoder_state_dict": encoder.state_dict(),
        "transfer_state_dict": transfer.state_dict(),
        "reg_net_state_dict": reg_net.state_dict(),
        "fusion_net_state_dict": fusion_net.state_dict(),
    }
    torch.save(final_ckpt, os.path.join(checkpoint_root, f'UnSFFusion_123_{modal}.pkl'))

    print(f"Best model: epoch={best_epoch}, best_total_loss={best_total_loss:.6f}, path={best_ckpt_path}")
    print("Training is completed.")


if __name__ == '__main__':
    modal = 'CT'
    train_batch_size = 8
    lr = 5e-5
    num_epoch = 3000
    beta1 = 0.9
    beta2 = 0.999
    resume = False

    train(
        modal,
        train_batch_size,
        lr,
        num_epoch,
        beta1,
        beta2,
        resume
    )