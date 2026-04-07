# metric.py
import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from skimage.metrics import structural_similarity as ssim  # pip install scikit-image


# =========================================================
# QSSIM: average SSIM between MRI-F and Other-F
# =========================================================
def compute_qssim(mri_y: torch.Tensor, other_y: torch.Tensor, fusion_y: torch.Tensor) -> float:
    """
    mri_y, other_y, fusion_y: [B,1,H,W] or [B,3,H,W], values in [0,1]
    Evaluation is performed on luminance only, so if the input has 3 channels,
    only the first channel is used.
    """
    if mri_y.shape[1] > 1:
        mri_y = mri_y[:, 0:1, :, :]
    if other_y.shape[1] > 1:
        other_y = other_y[:, 0:1, :, :]
    if fusion_y.shape[1] > 1:
        fusion_y = fusion_y[:, 0:1, :, :]

    m = mri_y.squeeze().detach().cpu().numpy()
    o = other_y.squeeze().detach().cpu().numpy()
    f = fusion_y.squeeze().detach().cpu().numpy()
    s1 = ssim(m, f, data_range=1.0)
    s2 = ssim(o, f, data_range=1.0)
    return s1 + s2


# ===================== QABF (Xydeas & Petrovic, Q_AB/F) =====================
# The original torch implementation is replaced here with the traditional
# implementation you provided (cv2/numpy + paper parameters).
def _to_gray_float64_np(x):
    """
    Supports torch.Tensor / np.ndarray
    Input can be [B,C,H,W] / [C,H,W] / [H,W] / [H,W,3]
    Output is [H,W] float64 (numpy)
    """
    if torch.is_tensor(x):
        x = x.detach()
        # Take the first batch and the first channel
        if x.ndim == 4:
            x = x[0, 0]  # [B,C,H,W] -> [H,W]
        elif x.ndim == 3:
            x = x[0]     # [C,H,W] -> [H,W] (default: use channel 0)
        x = x.cpu().numpy()

    x = np.asarray(x)
    if x.ndim == 3:
        # If the input is a color image, convert it to grayscale
        # OpenCV assumes BGR order by default; if the actual input is RGB,
        # the overall trend is usually not affected significantly.
        if x.shape[2] == 3:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        else:
            x = x[..., 0]
    return x.astype(np.float64, copy=False)


def _conv2_same_np(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Equivalent to:
    0 padding + 180° kernel flip + valid convolution
    cv2.filter2D performs correlation, so the kernel is flipped here
    to implement convolution.
    """
    k_flip = cv2.flip(k, -1)
    out = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=k_flip, borderType=cv2.BORDER_CONSTANT)
    return out


def _grad_strength_orientation_np(img: np.ndarray, eps: float = 1e-12):
    """
    Reproduces the original getArray logic:
      SAx = conv(h3, img), SAy = conv(h1, img)
      g = sqrt(SAx^2 + SAy^2)
      a = atan(SAy/SAx), and if SAx==0 -> pi/2
    """
    # Sobel kernels, consistent with the QABF code you provided
    h1 = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)   # y
    h3 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)     # x

    SAx = _conv2_same_np(img, h3)
    SAy = _conv2_same_np(img, h1)

    g = np.sqrt(SAx * SAx + SAy * SAy + eps)

    # Keep the same behavior as the original implementation (not atan2)
    a = np.where(np.abs(SAx) < eps, math.pi / 2.0, np.arctan(SAy / (SAx + eps)))
    return g, a


def _q_map_np(aS, gS, aF, gF,
              Tg=0.9994, kg=-15.0, Dg=0.5,
              Ta=0.9879, ka=-22.0, Da=0.8,
              eps: float = 1e-12):
    """
    Compute pixel-wise QSF (QAF or QBF)
    Uses stable relative gradient strength:
    GSF = min(gS/gF, gF/gS), then clipped to [0,1]
    """
    ratio1 = gS / (gF + eps)
    ratio2 = gF / (gS + eps)
    GSF = np.minimum(ratio1, ratio2)
    GSF = np.clip(GSF, 0.0, 1.0)

    delta = np.abs(aS - aF)  # |αA-αF|
    AAF = np.abs(delta - (np.pi / 2.0)) / (np.pi / 2)  # ||αA-αF| - π/2| / (π/2)
    AAF = np.clip(AAF, 0.0, 1.0)

    Qg = Tg / (1.0 + np.exp(kg * (GSF - Dg)))
    Qa = Ta / (1.0 + np.exp(ka * (AAF - Da)))

    return Qg * Qa


@torch.no_grad()
def compute_qabf(
    imgA: torch.Tensor,
    imgB: torch.Tensor,
    imgF: torch.Tensor,
    *,
    L: float = 1.0,
    Tg: float = 0.9994,
    kg: float = -15.0,
    Dg: float = 0.5,
    Ta: float = 0.9879,
    ka: float = -22.0,
    Da: float = 0.8,
    eps: float = 1e-12,
) -> float:
    """
    Q_AB/F (QABF) metric wrapped from the traditional implementation you provided.
    Supports torch.Tensor inputs (converted to cpu numpy internally).
    Larger values indicate better fusion quality.
    """
    A = _to_gray_float64_np(imgA)
    B = _to_gray_float64_np(imgB)
    Fm = _to_gray_float64_np(imgF)

    if A.shape != B.shape or A.shape != Fm.shape:
        raise ValueError(f"Input shapes must match, got A{A.shape}, B{B.shape}, F{Fm.shape}")

    gA, aA = _grad_strength_orientation_np(A, eps=eps)
    gB, aB = _grad_strength_orientation_np(B, eps=eps)
    gF, aF = _grad_strength_orientation_np(Fm, eps=eps)

    QAF = _q_map_np(aA, gA, aF, gF, Tg=Tg, kg=kg, Dg=Dg, Ta=Ta, ka=ka, Da=Da, eps=eps)
    QBF = _q_map_np(aB, gB, aF, gF, Tg=Tg, kg=kg, Dg=Dg, Ta=Ta, ka=ka, Da=Da, eps=eps)

    wA = np.power(gA, L)
    wB = np.power(gB, L)

    num = np.sum(QAF * wA + QBF * wB)
    den = np.sum(wA + wB) + eps
    return float(num / den)


# ===================== QCV (Chen–Varshney 2007, paper-form implementation) =====================
# No 0–1 normalization is applied to the final result.
# Smaller Q values indicate better fusion quality.
def _normalize1_uint8_like(x: torch.Tensor) -> torch.Tensor:
    x = x.squeeze().to(torch.float64)
    da = x.max()
    xiao = x.min()
    if (da == 0) and (xiao == 0):
        return x.clone()
    newdata = (x - xiao) / (da - xiao + 1e-12)
    return torch.round(newdata * 255.0)


def _to_hw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert any shape [B,C,H,W] / [C,H,W] / [H,W] into [H,W]
    by taking the first batch and the first channel.
    """
    x = x.to(torch.float64)
    if x.ndim >= 3:
        x = x.view(-1, *x.shape[-2:])[0]
    return x


def _sobel_mag(x255: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], dtype=torch.float64, device=x255.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                       [0.,  0.,  0.],
                       [1.,  2.,  1.]], dtype=torch.float64, device=x255.device).view(1, 1, 3, 3)
    x = x255[None, None, ...]
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-12).squeeze(0).squeeze(0)
    return g  # [H,W], float64


def _crop_to_multiple(x: torch.Tensor, win: int) -> torch.Tensor:
    H, W = x.shape[-2], x.shape[-1]
    Hc = (H // win) * win
    Wc = (W // win) * win
    return x[..., :Hc, :Wc]


def _block_sum_pow(x: torch.Tensor, win: int, alpha: float) -> torch.Tensor:
    x4 = x[None, None, ...]
    patches = F.unfold(x4, kernel_size=win, stride=win)  # [1, win*win, L]
    val = (patches ** alpha).sum(dim=1)                  # [1, L]
    Hc = x4.shape[-2] // win
    Wc = x4.shape[-1] // win
    return val.view(Hc, Wc)                              # [nH, nW]


def _block_mean_sq(x: torch.Tensor, win: int) -> torch.Tensor:
    x4 = x[None, None, ...]
    patches = F.unfold(x4, kernel_size=win, stride=win)  # [1, win*win, L]
    ms = (patches ** 2).mean(dim=1)                      # [1, L]
    Hc = x4.shape[-2] // win
    Wc = x4.shape[-1] // win
    return ms.view(Hc, Wc)                               # [nH, nW]


def _theta_m_len3(H: int, W: int, device) -> torch.Tensor:
    # MATLAB: [u,v]=freqspace([H,W],'meshgrid'); u=(W/8)*u; v=(H/8)*v
    u_lin = torch.linspace(-1.0, 1.0, steps=W, dtype=torch.float64, device=device)
    v_lin = torch.linspace(-1.0, 1.0, steps=H, dtype=torch.float64, device=device)
    v, u = torch.meshgrid(v_lin, u_lin, indexing='ij')
    u = (W / 8.0) * u
    v = (H / 8.0) * v
    r = torch.sqrt(u * u + v * v)
    theta_m = 2.6 * (0.0192 + 0.144 * r) * torch.exp(-(0.144 * r) ** 1.1)
    return theta_m  # aligned with fftshift


@torch.no_grad()
def qcv_hvs_metric(
    xF: torch.Tensor,    # fused image
    x1: torch.Tensor,    # source image 1 (MRI)
    x2: torch.Tensor,    # source image 2 (CT/PET)
    win: int = 16,       # paper setting: 16×16 blocks for Lenna
    alpha: float = 2.0   # paper setting: a=2 for two source images
) -> float:
    """
    QCV from Chen & Varshney 2007:
      Q = sum_l( k1*D1 + k2*D2 ) / sum_l( k1 + k2 )
    Smaller values indicate lower fusion error and better quality.
    """
    im1 = _normalize1_uint8_like(_to_hw(x1))
    im2 = _normalize1_uint8_like(_to_hw(x2))
    fus = _normalize1_uint8_like(_to_hw(xF))

    g1 = _sobel_mag(im1)
    g2 = _sobel_mag(im2)

    im1 = _crop_to_multiple(im1, win)
    im2 = _crop_to_multiple(im2, win)
    fus = _crop_to_multiple(fus, win)
    g1  = _crop_to_multiple(g1,  win)
    g2  = _crop_to_multiple(g2,  win)

    H, W = fus.shape

    f1 = im1 - fus
    f2 = im2 - fus

    Hc = _theta_m_len3(H, W, device=fus.device)
    F1  = torch.fft.fft2(f1)
    F2  = torch.fft.fft2(f2)
    F1s = torch.fft.fftshift(F1)
    F2s = torch.fft.fftshift(F2)
    Df1 = torch.fft.ifft2(torch.fft.ifftshift(F1s * Hc)).real
    Df2 = torch.fft.ifft2(torch.fft.ifftshift(F2s * Hc)).real

    ramda1 = _block_sum_pow(g1, win, alpha)  # sum(G^alpha)
    ramda2 = _block_sum_pow(g2, win, alpha)
    D1     = _block_mean_sq(Df1, win)        # mean((filtered error)^2)
    D2     = _block_mean_sq(Df2, win)

    num = (ramda1 * D1 + ramda2 * D2).sum()
    den = (ramda1 + ramda2).sum()
    Q = (num / (den + 1e-12)).item()
    return float(Q)


def compute_qcv(mri_y: torch.Tensor, other_y: torch.Tensor, fusion_y: torch.Tensor,
                win: int = 16, alpha: float = 2.0) -> float:
    return qcv_hvs_metric(fusion_y, mri_y, other_y, win=win, alpha=alpha)


# ================= QVIFF (Han et al. 2013 - strict MATLAB equivalence to VIFF_Public) =================
# Only this section is replaced with the logic from your VIFF_Public/ComVidVindG code.
from scipy.signal import convolve2d  # pip install scipy
from skimage.color import rgb2lab    # pip install scikit-image


def _fspecial_gaussian(N: int, sigma: float) -> np.ndarray:
    """MATLAB: fspecial('gaussian', N, sigma)"""
    ax = np.arange(-(N - 1) / 2.0, (N - 1) / 2.0 + 1.0, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    h /= np.sum(h)
    return h


def _filter2_valid(win: np.ndarray, img: np.ndarray) -> np.ndarray:
    """MATLAB: filter2(win, img, 'valid') == conv2(img, rot90(win,2), 'valid')"""
    return convolve2d(img, np.rot90(win, 2), mode="valid")


def ComVidVindG(ref: np.ndarray, dist: np.ndarray, sq: float):
    """
    MATLAB: [Tg1,Tg2,Tg3]=ComVidVindG(ref,dist,sq)
    Returns:
      Tg1: Num (list, len=4), VID matrix at each scale
      Tg2: Den (list, len=4), VIND matrix at each scale
      Tg3: G   (list, len=4), g matrix at each scale
    """
    sigma_nsq = sq
    Num, Den, G = [], [], []

    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = _fspecial_gaussian(N, N / 5.0)

        if scale > 1:
            ref = _filter2_valid(win, ref)
            dist = _filter2_valid(win, dist)
            ref = ref[0::2, 0::2]
            dist = dist[0::2, 0::2]

        mu1 = _filter2_valid(win, ref)
        mu2 = _filter2_valid(win, dist)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = _filter2_valid(win, ref * ref) - mu1_sq
        sigma2_sq = _filter2_valid(win, dist * dist) - mu2_sq
        sigma12 = _filter2_valid(win, ref * dist) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        mask1 = sigma1_sq < 1e-10
        g[mask1] = 0
        sv_sq[mask1] = sigma2_sq[mask1]
        sigma1_sq[mask1] = 0

        mask2 = sigma2_sq < 1e-10
        g[mask2] = 0
        sv_sq[mask2] = 0

        mask3 = g < 0
        sv_sq[mask3] = sigma2_sq[mask3]
        g[mask3] = 0

        sv_sq[sv_sq <= 1e-10] = 1e-10

        VID = np.log10(1.0 + (g ** 2) * sigma1_sq / (sv_sq + sigma_nsq))
        VIND = np.log10(1.0 + sigma1_sq / sigma_nsq)

        G.append(g)
        Num.append(VID)
        Den.append(VIND)

    return Num, Den, G


def VIFF_Public(Im1: np.ndarray, Im2: np.ndarray, ImF: np.ndarray) -> float:
    """
    MATLAB: F=VIFF_Public(Im1,Im2,ImF)
    Notes:
      - The original MATLAB implementation uses sq=0.005*255*255,
        which assumes inputs are in the range 0–255.
      - For color images: convert with srgb2lab and use the L channel.
      - For grayscale images: use them directly.
    """
    sq = 0.005 * 255.0 * 255.0
    C = 1e-7

    Im1 = np.asarray(Im1)
    Im2 = np.asarray(Im2)
    ImF = np.asarray(ImF)

    # Color space transformation
    if Im1.ndim == 3 and Im1.shape[2] == 3:
        def _to_lab_L(x):
            x = x.astype(np.float64, copy=False)
            # rgb2lab expects float values in [0,1]
            if x.max() > 1.0:
                x = x / 255.0
            return rgb2lab(x)[..., 0]  # L channel

        Ix1 = _to_lab_L(Im1)
        Ix2 = _to_lab_L(Im2)
        IxF = _to_lab_L(ImF)
    else:
        Ix1, Ix2, IxF = Im1, Im2, ImF

    T1p = Ix1.astype(np.float64, copy=False)
    T2p = Ix2.astype(np.float64, copy=False)
    Trp = IxF.astype(np.float64, copy=False)

    p = np.array([1.0, 0.0, 0.15, 1.0], dtype=np.float64) / 2.15

    T1N, T1D, T1G = ComVidVindG(T1p, Trp, sq)
    T2N, T2D, T2G = ComVidVindG(T2p, Trp, sq)

    F_scales = np.zeros(4, dtype=np.float64)

    for i in range(4):
        M_Z1 = T1N[i]
        M_Z2 = T2N[i]
        M_M1 = T1D[i]
        M_M2 = T2D[i]
        M_G1 = T1G[i]
        M_G2 = T2G[i]

        Lmask = M_G1 < M_G2

        M_Z12 = M_Z2.copy()
        M_Z12[Lmask] = M_Z1[Lmask]

        M_M12 = M_M2.copy()
        M_M12[Lmask] = M_M1[Lmask]

        eps = 1e-12
        VID = np.sum(M_Z12)
        VIND = np.sum(M_M12)
        F_scales[i] = VID / (VIND + eps)

    return float(np.sum(F_scales * p))


def compute_qviff(mri_y: torch.Tensor, other_y: torch.Tensor, fusion_y: torch.Tensor) -> float:
    # Keep the same input logic as the other compute_* metrics: torch -> numpy, default [0,1] range
    if mri_y.shape[1] > 1:
        mri_y = mri_y[:, 0:1, :, :]
    if other_y.shape[1] > 1:
        other_y = other_y[:, 0:1, :, :]
    if fusion_y.shape[1] > 1:
        fusion_y = fusion_y[:, 0:1, :, :]

    m = mri_y.squeeze().detach().cpu().clamp(0, 1).numpy()
    o = other_y.squeeze().detach().cpu().clamp(0, 1).numpy()
    f = fusion_y.squeeze().detach().cpu().clamp(0, 1).numpy()

    # The original VIFF_Public implementation assumes 0–255 input
    # (sq=0.005*255^2), so convert here for strict consistency.
    m255 = (m * 255.0).astype(np.float64, copy=False)
    o255 = (o * 255.0).astype(np.float64, copy=False)
    f255 = (f * 255.0).astype(np.float64, copy=False)

    return VIFF_Public(m255, o255, f255)


# ===================== Piella Fusion Metric (Piella 2003/2004) =====================
# Notes:
# - Input follows your current code style: torch [B,1,H,W] or [B,3,H,W], default range [0,1]
# - Internally converted to 0–255 (float64) for more stable values and better consistency
#   with traditional implementations
# - sw:
#     1: simple average of Q_map
#     2: weighted by max(varA, varB) as saliency
#     3: intensity-domain Q * (edge-domain Qw ^ alpha)

def _to_1x1_hw_torch(x: torch.Tensor) -> torch.Tensor:
    """Convert any shape [B,C,H,W] / [C,H,W] / [H,W] into [1,1,H,W] by taking the first batch and first channel."""
    if not torch.is_tensor(x):
        raise TypeError("Input must be torch.Tensor")

    x = x.detach()
    if x.ndim == 4:
        x = x[0, 0]  # [H,W]
    elif x.ndim == 3:
        x = x[0]     # [H,W]
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

    x = x.to(torch.float64)
    return x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def _gaussian_window(win_size: int, sigma: float, device, dtype=torch.float64) -> torch.Tensor:
    """Generate a [1,1,win,win] Gaussian window for local SSIM statistics."""
    if win_size % 2 == 0 or win_size < 3:
        raise ValueError("win_size must be odd and >= 3")
    coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size // 2)
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / (g.sum() + 1e-12)
    w = (g[:, None] * g[None, :])
    w = w / (w.sum() + 1e-12)
    return w.view(1, 1, win_size, win_size)


def ssim_index_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    *,
    win_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 255.0,
    eps: float = 1e-12,
):
    """
    Return (ssim_val, ssim_map, sigma1_sq, sigma2_sq)
    Uses 'valid' convolution (no padding), so the output size is:
    [1,1,H-win+1,W-win+1]
    """
    x1 = _to_1x1_hw_torch(img1)
    x2 = _to_1x1_hw_torch(img2)

    # Normalize to 0–255 range
    x1 = x1.clamp(0.0, 1.0) * data_range
    x2 = x2.clamp(0.0, 1.0) * data_range

    device = x1.device
    win = _gaussian_window(win_size, sigma, device=device, dtype=x1.dtype)

    mu1 = F.conv2d(x1, win, padding=0)
    mu2 = F.conv2d(x2, win, padding=0)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(x1 * x1, win, padding=0) - mu1_sq
    sigma2_sq = F.conv2d(x2 * x2, win, padding=0) - mu2_sq
    sigma12   = F.conv2d(x1 * x2, win, padding=0) - mu1_mu2

    # Standard SSIM constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + eps)
    ssim_val = ssim_map.mean()

    return ssim_val, ssim_map, sigma1_sq, sigma2_sq


def _edge_mag_piella(x: torch.Tensor) -> torch.Tensor:
    """
    Piella edge magnitude, consistent with your previous implementation:
      k1 = [[1,0,-1],[1,0,-1],[1,0,-1]]
      k2 = [[1,1,1],[0,0,0],[-1,-1,-1]]
    Output shape: [1,1,H,W] (same padding)
    """
    x = _to_1x1_hw_torch(x).clamp(0.0, 1.0) * 255.0  # edge calculation is more stable in 0–255 scale

    k1 = torch.tensor([[1., 0., -1.],
                       [1., 0., -1.],
                       [1., 0., -1.]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    k2 = torch.tensor([[ 1.,  1.,  1.],
                       [ 0.,  0.,  0.],
                       [-1., -1., -1.]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    gx = F.conv2d(x, k1, padding=1)
    gy = F.conv2d(x, k2, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag


@torch.no_grad()
def metric_piella(
    img1: torch.Tensor,   # source image A
    img2: torch.Tensor,   # source image B
    fuse: torch.Tensor,   # fused image
    sw: int = 1,          # 1/2/3
    alpha: float = 1.0,   # exponent used when sw=3
    *,
    win_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 255.0,
) -> float:
    # ---- Step 1: intensity-domain SSIM statistics (used for lambda and Q_map) ----
    _, _, sigma1_sq, sigma2_sq = ssim_index_torch(
        img1, img2, win_size=win_size, sigma=sigma, data_range=data_range
    )

    buffer_ = sigma1_sq + sigma2_sq
    zero_mask = (buffer_ == 0)
    sigma1_sq = sigma1_sq + zero_mask * 0.5
    sigma2_sq = sigma2_sq + zero_mask * 0.5

    buffer_ = sigma1_sq + sigma2_sq
    eps_t = torch.finfo(sigma1_sq.dtype).eps
    lam = sigma1_sq / (buffer_ + eps_t)

    _, ssim_map1, _, _ = ssim_index_torch(
        fuse, img1, win_size=win_size, sigma=sigma, data_range=data_range
    )
    _, ssim_map2, _, _ = ssim_index_torch(
        fuse, img2, win_size=win_size, sigma=sigma, data_range=data_range
    )

    Q_map = lam * ssim_map1 + (1.0 - lam) * ssim_map2

    if sw == 1:
        return float(Q_map.mean().item())

    if sw == 2:
        Cw = torch.maximum(sigma1_sq, sigma2_sq)
        cw = Cw / (Cw.sum() + eps_t)
        Q = (cw * Q_map).sum()
        return float(Q.item())

    if sw == 3:
        # 3.1 intensity-domain weighted Q (equivalent to sw=2)
        Cw_int = torch.maximum(sigma1_sq, sigma2_sq)
        cw_int = Cw_int / (Cw_int.sum() + eps_t)
        Q_intensity = (cw_int * Q_map).sum()

        # 3.2 edge magnitude maps
        fuseF = _edge_mag_piella(fuse)
        img1F = _edge_mag_piella(img1)
        img2F = _edge_mag_piella(img2)

        # 3.3 repeat SSIM and weighting in the edge domain to get Qw
        _, _, sigma1_sq_e, sigma2_sq_e = ssim_index_torch(
            img1F, img2F, win_size=win_size, sigma=sigma, data_range=data_range
        )

        buffer_e = sigma1_sq_e + sigma2_sq_e
        zero_mask_e = (buffer_e == 0)
        sigma1_sq_e = sigma1_sq_e + zero_mask_e * 0.5
        sigma2_sq_e = sigma2_sq_e + zero_mask_e * 0.5

        buffer_e = sigma1_sq_e + sigma2_sq_e
        lam_e = sigma1_sq_e / (buffer_e + eps_t)

        _, ssim_map1_e, _, _ = ssim_index_torch(
            fuseF, img1F, win_size=win_size, sigma=sigma, data_range=data_range
        )
        _, ssim_map2_e, _, _ = ssim_index_torch(
            fuseF, img2F, win_size=win_size, sigma=sigma, data_range=data_range
        )

        Q_map_e = lam_e * ssim_map1_e + (1.0 - lam_e) * ssim_map2_e

        Cw_e = torch.maximum(sigma1_sq_e, sigma2_sq_e)
        cw_e = Cw_e / (Cw_e.sum() + eps_t)
        Qw = (cw_e * Q_map_e).sum()

        Qe = Q_intensity * (Qw ** alpha)
        return float(Qe.item())

    raise ValueError("sw must be 1, 2, or 3")


def compute_piella(
    mri_y: torch.Tensor,
    other_y: torch.Tensor,
    fusion_y: torch.Tensor,
    sw: int = 1,
    alpha: float = 1.0,
    *,
    win_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Wrapper for the Piella fusion quality metric,
    consistent with your existing compute_* function style.
    sw=1/2/3; when sw=3, alpha controls the edge contribution.
    """
    if mri_y.shape[1] > 1:
        mri_y = mri_y[:, 0:1, :, :]
    if other_y.shape[1] > 1:
        other_y = other_y[:, 0:1, :, :]
    if fusion_y.shape[1] > 1:
        fusion_y = fusion_y[:, 0:1, :, :]

    return metric_piella(
        mri_y, other_y, fusion_y,
        sw=sw, alpha=alpha,
        win_size=win_size, sigma=sigma, data_range=255.0
    )


__all__ = [
    "compute_qssim",
    "compute_qcv",
    "compute_qviff",
    "compute_qabf",
    "qcv_hvs_metric",
    "compute_piella"
]