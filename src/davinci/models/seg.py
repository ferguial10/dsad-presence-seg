"""Segmentation models with one sigmoid channel per structure."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class DeepLabV3R50(nn.Module):
    """torchvision DeepLabV3-ResNet50; COCO weights as in Kolbinger et al. 2023."""

    def __init__(self, num_outputs: int, pretrained: str = "coco"):
        super().__init__()
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights, deeplabv3_resnet50)
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained == "coco" else None
        self.net = deeplabv3_resnet50(weights=weights, aux_loss=None if weights else False)
        self.net.aux_classifier = None
        self.net.classifier[4] = nn.Conv2d(256, num_outputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["out"]


class PresenceHead(nn.Module):
    """Image-level presence logits from globally pooled encoder features."""

    def __init__(self, channels: int, num_outputs: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(channels, num_outputs))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc(f.float().mean((2, 3)))


class DeepLabV3R50MultiHead(nn.Module):
    """Shared COCO ResNet50 backbone with one DeepLabv3 decoder per structure.

    The combined model of Kolbinger et al. 2023: every decoder starts from the
    COCO-pretrained DeepLabv3 head and ends in a single binary logit.
    """

    def __init__(self, num_outputs: int, presence_head: bool = False):
        super().__init__()
        import copy
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights, deeplabv3_resnet50)
        net = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        self.backbone = net.backbone
        net.classifier[4] = nn.Conv2d(256, 1, 1)
        self.heads = nn.ModuleList(copy.deepcopy(net.classifier) for _ in range(num_outputs))
        for head in self.heads:
            head[4].reset_parameters()
        self.presence = PresenceHead(2048, num_outputs) if presence_head else None

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)["out"]
        out = torch.cat([head(f) for head in self.heads], 1)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return (out, self.presence(f)) if self.presence is not None else out


def masked_bce_dice(logits: torch.Tensor, target: torch.Tensor,
                    weight: torch.Tensor, eps: float = 1.0,
                    dice_channels: torch.Tensor | None = None) -> torch.Tensor:
    """BCE over weighted pixels plus soft Dice per (image, channel).

    Dice is averaged over the (image, channel) pairs with any weight, or only
    over ``dice_channels`` (B, K) when given. Restricting it to channels that
    carry a mask avoids the degenerate Dice of empty targets, which is ~1 for
    any predicted mass and pushes towards predicting nothing (amendment 3).
    """
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denom = weight.sum().clamp_min(1.0)
    bce = (bce * weight).sum() / denom
    prob = torch.sigmoid(logits) * weight
    t = target * weight
    inter = (prob * t).sum((2, 3))
    union = prob.sum((2, 3)) + t.sum((2, 3))
    active = weight.sum((2, 3)) > 0
    if dice_channels is not None:
        active &= dice_channels.bool()
    dice = 1 - (2 * inter + eps) / (union + eps)
    dice = dice[active].mean() if active.any() else logits.sum() * 0
    return bce + dice


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen and B initialised to zero."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, r, bias=False)
        self.b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


def _conv_bn_relu(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


SURGENETXL_FILES = {"base": "DINOv2_ViTb14_size336_SurgeNetXL.pth",
                    "large": "DINOv2_ViTl14_size336_SurgeNetXL.pth"}


def load_surgenetxl(encoder: nn.Module, size: str) -> None:
    """Load SurgeNetXL DINOv2 weights (de Jong et al., MIDL 2026; CC BY-NC-SA)
    from the original DINOv2 layout into a HF ``Dinov2Model``.

    The checkpoint is read with ``weights_only=True``. Its 24x24 position grid
    (336 px pretraining) replaces the HF grid and is interpolated at run time.
    """
    import os
    root = Path(os.environ.get("SURGENETXL_DIR", Path.home() / "models" / "surgenetxl"))
    src = torch.load(root / SURGENETXL_FILES[size], map_location="cpu", weights_only=True)
    sd = {"embeddings.cls_token": src["cls_token"], "embeddings.mask_token": src["mask_token"],
          "embeddings.position_embeddings": src["pos_embed"],
          "embeddings.patch_embeddings.projection.weight": src["patch_embed.proj.weight"],
          "embeddings.patch_embeddings.projection.bias": src["patch_embed.proj.bias"],
          "layernorm.weight": src["norm.weight"], "layernorm.bias": src["norm.bias"]}
    for i in range(encoder.config.num_hidden_layers):
        a, b = f"blocks.{i}.", f"encoder.layer.{i}."
        for t in ("weight", "bias"):
            q, k, v = src[a + f"attn.qkv.{t}"].chunk(3)
            sd |= {b + f"attention.q_proj.{t}": q, b + f"attention.k_proj.{t}": k,
                   b + f"attention.v_proj.{t}": v, b + f"attention.o_proj.{t}": src[a + f"attn.proj.{t}"],
                   b + f"norm1.{t}": src[a + f"norm1.{t}"], b + f"norm2.{t}": src[a + f"norm2.{t}"],
                   b + f"mlp.fc1.{t}": src[a + f"mlp.fc1.{t}"], b + f"mlp.fc2.{t}": src[a + f"mlp.fc2.{t}"]}
        sd[b + "layer_scale1.lambda1"] = src[a + "ls1.gamma"]
        sd[b + "layer_scale2.lambda1"] = src[a + "ls2.gamma"]
    encoder.embeddings.position_embeddings = nn.Parameter(torch.zeros_like(sd["embeddings.position_embeddings"]))
    encoder.load_state_dict(sd, strict=True)


class DINOv2Seg(nn.Module):
    """DINOv2 encoder (frozen or LoRA) + light multi-level decoder.

    Four evenly spaced transformer layers are projected to 256 channels, fused at
    patch resolution (1/14), upsampled and refined with a shallow convolutional
    image stem at 1/4 and 1/2 resolution so that thin structures keep detail.
    """

    def __init__(self, num_outputs: int, size: str = "base", lora_rank: int = 0,
                 presence_head: bool = False, weights: str = "lvd142m"):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(f"facebook/dinov2-{size}",
                                                 attn_implementation="sdpa")
        if weights == "surgenetxl":
            load_surgenetxl(self.encoder, size)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if lora_rank:
            for layer in self.encoder.encoder.layer:
                att = layer.attention
                att.q_proj = LoRALinear(att.q_proj, lora_rank, lora_rank)
                att.v_proj = LoRALinear(att.v_proj, lora_rank, lora_rank)
        n = self.encoder.config.num_hidden_layers
        self.layers = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
        dim = self.encoder.config.hidden_size
        self.proj = nn.ModuleList(nn.Conv2d(dim, 256, 1) for _ in self.layers)
        self.fuse = nn.Sequential(_conv_bn_relu(1024, 256), _conv_bn_relu(256, 256))
        self.stem4 = nn.Sequential(_conv_bn_relu(3, 32), nn.MaxPool2d(2),
                                   _conv_bn_relu(32, 64), nn.MaxPool2d(2))
        self.stem2 = nn.Sequential(_conv_bn_relu(3, 32), nn.MaxPool2d(2))
        self.up4 = nn.Sequential(_conv_bn_relu(256 + 64, 128), _conv_bn_relu(128, 128))
        self.up2 = nn.Sequential(_conv_bn_relu(128 + 32, 64), _conv_bn_relu(64, 64))
        self.head = nn.Conv2d(64, num_outputs, 1)
        self.presence = PresenceHead(256, num_outputs) if presence_head else None

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        ph, pw = h // 14, w // 14
        out = self.encoder(pixel_values=x, output_hidden_states=True)
        feats = []
        for proj, i in zip(self.proj, self.layers):
            tokens = out.hidden_states[i + 1][:, 1:, :]  # drop CLS; index 0 = embeddings
            tokens = tokens[:, -ph * pw:, :]             # drop register tokens if any
            feats.append(proj(tokens.transpose(1, 2).reshape(x.shape[0], -1, ph, pw)))
        f = self.fuse(torch.cat(feats, 1))
        presence = self.presence(f) if self.presence is not None else None
        s4 = self.stem4(x)
        f = F.interpolate(f, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        f = self.up4(torch.cat([f, s4], 1))
        s2 = self.stem2(x)
        f = F.interpolate(f, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        f = self.up2(torch.cat([f, s2], 1))
        out = F.interpolate(self.head(f), size=(h, w), mode="bilinear", align_corners=False)
        return (out, presence) if presence is not None else out


class SAM2Seg(nn.Module):
    """SAM2.1 Hiera image encoder (FPN neck at 1/4, 1/8, 1/16) + light decoder.

    LoRA adapts the fused ``qkv`` projection of every Hiera block; the decoder
    fuses the three FPN levels at 1/4 and refines at 1/2 with an image stem, as
    in ``DINOv2Seg``.
    """

    def __init__(self, num_outputs: int, size: str = "base-plus", lora_rank: int = 16,
                 presence_head: bool = False):
        super().__init__()
        from transformers import Sam2Model
        self.encoder = Sam2Model.from_pretrained(f"facebook/sam2.1-hiera-{size}").vision_encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if lora_rank:
            for block in self.encoder.backbone.blocks:
                block.attn.qkv = LoRALinear(block.attn.qkv, lora_rank, lora_rank)
        self.fuse = nn.Sequential(_conv_bn_relu(3 * 256, 256), _conv_bn_relu(256, 128))
        self.stem2 = nn.Sequential(_conv_bn_relu(3, 32), nn.MaxPool2d(2))
        self.up2 = nn.Sequential(_conv_bn_relu(128 + 32, 64), _conv_bn_relu(64, 64))
        self.head = nn.Conv2d(64, num_outputs, 1)
        self.presence = PresenceHead(256, num_outputs) if presence_head else None

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        fpn = self.encoder(pixel_values=x).fpn_hidden_states  # strides 4, 8, 16
        size = fpn[0].shape[-2:]
        f = self.fuse(torch.cat([fpn[0]] + [F.interpolate(t, size=size, mode="bilinear",
                                                          align_corners=False) for t in fpn[1:]], 1))
        s2 = self.stem2(x)
        f = F.interpolate(f, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        f = self.up2(torch.cat([f, s2], 1))
        out = F.interpolate(self.head(f), size=(h, w), mode="bilinear", align_corners=False)
        return (out, self.presence(fpn[-1])) if self.presence is not None else out


def build_model(name: str, num_outputs: int) -> nn.Module:
    """Model names: deeplabv3_r50_coco, deeplabv3_r50_coco_multihead,
    dinov2_{base,large}_{frozen,lora16}[_surgenetxl], sam2_baseplus_lora16;
    the suffix ``_ph``
    adds an image-level presence head (the model then returns two outputs)."""
    presence = name.endswith("_ph")
    name = name.removesuffix("_ph")
    if name == "deeplabv3_r50_coco" and not presence:
        return DeepLabV3R50(num_outputs, "coco")
    if name == "deeplabv3_r50_coco_multihead":
        return DeepLabV3R50MultiHead(num_outputs, presence)
    if name.startswith("dinov2_"):  # dinov2_<size>_<frozen|loraR>[_surgenetxl]
        _, size, mode, *weights = name.split("_")
        rank = int(mode.removeprefix("lora")) if mode.startswith("lora") else 0
        return DINOv2Seg(num_outputs, size, rank, presence, *(weights or ["lvd142m"]))
    if name == "sam2_baseplus_lora16":
        return SAM2Seg(num_outputs, "base-plus", 16, presence)
    raise ValueError(f"unknown model {name}")
