"""
TR: DeepFusionNet düşük ışık görüntü iyileştirme modeli ve yükleme yardımcıları.
Bu dosyadaki sınıf/katman isimleri, eğitilmiş ağırlık dosyasıyla uyumluluk için korunmuştur.

EN: DeepFusionNet low-light image enhancement model and loader utilities.
Class/layer names are kept stable for checkpoint compatibility.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention bloğu.

    Amaç:
        Feature map içindeki kanal bazlı önemli bilgileri güçlendirmek.

    Giriş:
        x: [B, C, H, W]

    Çıkış:
        x ile aynı boyutta attention uygulanmış tensor.
    """

    def __init__(self, in_channels: int, oran: int = 16):
        super().__init__()

        hidden_channels = max(in_channels // oran, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return attention * x


class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention bloğu.

    Amaç:
        Görüntü üzerinde konumsal olarak önemli bölgeleri güçlendirmek.

    Giriş:
        x: [B, C, H, W]

    Çıkış:
        x ile aynı boyutta attention uygulanmış tensor.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size tek sayı olmalı. Örn: 3, 5, 7")

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(out))
        return attention * x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Sıra:
        1) Channel Attention
        2) Spatial Attention
    """

    def __init__(self, in_channels: int, ratio: int = 4, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DeepFusionNet(nn.Module):
    """
    CBAM attention kullanan U-Net benzeri görüntü iyileştirme modeli.

    Giriş:
        RGB tensor: [B, 3, H, W], değer aralığı [0, 1]

    Çıkış:
        RGB tensor: [B, 3, H, W], değer aralığı [0, 1]

    Not:
        Model 4 defa MaxPool2d kullandığı için H ve W değerlerinin 16'nın katı
        olması önerilir. Örn: 512x320, 640x384, 768x448, 1280x720.
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Encoder - Stage 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.cbam1 = CBAM(128)
        self.conv1_depth = nn.Conv2d(32, 32, kernel_size=3, groups=32, padding=1)
        self.conv1_kernel = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv1_kernel_depth = nn.Conv2d(32, 32, kernel_size=5, groups=32, padding=2)

        # Encoder - Stage 2
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAM(128)

        # Encoder - Stage 3
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)

        # Encoder - Stage 4
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.cbam4 = CBAM(128)

        # Bottleneck
        self.pointwise = nn.Conv2d(128, 32, 1)
        self.bottleneck = nn.Conv2d(32, 32, 3, padding=1)
        self.bottleneckb = nn.Conv2d(32, 32, 3, padding=1)

        # Bu iki katman checkpoint uyumluluğu için korunmuştur.
        # forward içinde kullanılmıyorlar ama modeller/dusuk_isik_modeli.pth içinde ağırlıkları var.
        self.bottleneckb2 = nn.Conv2d(32, 32, 3, padding=1)
        self.cbam_bottleneck = CBAM(128)
        self.bn_bottleneck = nn.BatchNorm2d(128)

        # Decoder
        self.dec1 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec1b = nn.Conv2d(128, 128, 3, padding=1)
        self.cbam_dec1 = CBAM(128)

        self.dec2 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec2b = nn.Conv2d(128, 128, 3, padding=1)
        self.cbam_dec2 = CBAM(128)

        self.dec3 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec3b = nn.Conv2d(128, 128, 3, padding=1)
        self.cbam_dec3 = CBAM(128)

        self.dec4 = nn.Conv2d(160, 64, 3, padding=1)
        self.dec4b = nn.Conv2d(64, 64, 3, padding=1)
        self.cbam_dec4 = CBAM(64)

        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder Stage 1: 3x3 + depthwise ve 5x5 + depthwise fusion
        e1 = self.relu(self.conv1(x))
        e1_depth = self.conv1_depth(e1)
        e1_ghost = torch.cat([e1, e1_depth], dim=1)  # 64 kanal

        e1_kernel = self.relu(self.conv1_kernel(x))
        e1_kernel_depth = self.relu(self.conv1_kernel_depth(e1_kernel))
        e1_kernel_ghost = torch.cat([e1_kernel, e1_kernel_depth], dim=1)  # 64 kanal

        e1_ghost_full = torch.cat([e1_ghost, e1_kernel_ghost], dim=1)  # 128 kanal
        e11 = self.relu(self.conv1b(e1_ghost_full))
        e11 = self.cbam1(self.bn1(e11))
        p1 = self.pool(e11)

        # Encoder Stage 2
        e2 = self.relu(self.conv2(p1))
        e2 = self.relu(self.conv2b(e2))
        e2 = self.cbam2(self.bn2(e2))
        p2 = self.pool(e2)

        # Encoder Stage 3
        e3 = self.relu(self.conv3(p2))
        e3 = self.relu(self.conv3b(e3))
        e3 = self.cbam3(self.bn3(e3))
        p3 = self.pool(e3)

        # Encoder Stage 4
        e4 = self.relu(self.conv4(p3))
        e4 = self.relu(self.conv4b(e4))
        e4 = self.cbam4(self.bn4(e4))
        p4 = self.pool(e4)

        # Bottleneck: 128 kanal -> 32 kanal -> 4 farklı feature -> 128 kanal concat
        p4 = self.pointwise(p4)
        p4_1 = self.relu(self.bottleneck(p4))
        p4_2 = self.relu(self.bottleneckb(p4_1))
        p4_3 = self.relu(self.bottleneckb(p4_2))
        p4_4 = self.relu(self.bottleneckb(p4_3))

        t = torch.cat([p4_1, p4_2, p4_3, p4_4], dim=1)
        t_plus = self.relu(self.cbam_bottleneck(t))
        b = self.cbam_bottleneck(t_plus)

        # Decoder Stage 1
        d1 = self.upsample(b)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.relu(self.dec1(d1))
        d1 = self.relu(self.dec1b(d1))
        d1 = self.cbam_dec1(d1)

        # Decoder Stage 2
        d2 = self.upsample(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.relu(self.dec2(d2))
        d2 = self.relu(self.dec2b(d2))
        d2 = self.cbam_dec2(d2)

        # Decoder Stage 3
        d3 = self.upsample(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.relu(self.dec3(d3))
        d3 = self.relu(self.dec3b(d3))
        d3 = self.cbam_dec3(d3)

        # Decoder Stage 4
        d4 = self.upsample(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.relu(self.dec4(d4))
        d4 = self.relu(self.dec4b(d4))
        d4 = self.cbam_dec4(d4)

        out = self.sigmoid(self.final_conv(d4))
        return out


def get_device(device_arg: str = "auto") -> torch.device:
    """auto/cpu/cuda/cuda:0 gibi stringlerden torch.device üretir."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def extract_state_dict(checkpoint: Any) -> OrderedDict:
    """Farklı checkpoint formatlarından state_dict çıkarır."""
    if isinstance(checkpoint, OrderedDict):
        return checkpoint

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in checkpoint and isinstance(checkpoint[key], (dict, OrderedDict)):
                return OrderedDict(checkpoint[key])
        return OrderedDict(checkpoint)

    raise TypeError(f"Desteklenmeyen checkpoint tipi: {type(checkpoint)}")


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    """DataParallel kaynaklı module. önekini temizler."""
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        cleaned[new_key] = value
    return cleaned


def load_deepfusion_model(
    model_path: str | Path,
    device: torch.device,
    half: bool = False,
    strict: bool = True,
) -> DeepFusionNet:
    """DeepFusionNet modelini modeller/dusuk_isik_modeli.pth ağırlıklarıyla yükler."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    model = DeepFusionNet().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = clean_state_dict(extract_state_dict(checkpoint))

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"[WARN] Eksik katman sayısı: {len(missing)}")
        print("[WARN] İlk eksikler:", missing[:10])
    if unexpected:
        print(f"[WARN] Fazla/uyumsuz katman sayısı: {len(unexpected)}")
        print("[WARN] İlk fazlalar:", unexpected[:10])

    model.eval()

    if half:
        if device.type != "cuda":
            print("[WARN] --half sadece CUDA üzerinde anlamlıdır. CPU'da float32 kullanılacak.")
        else:
            model.half()

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Toplam ve eğitilebilir parametre sayısını döndürür."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
