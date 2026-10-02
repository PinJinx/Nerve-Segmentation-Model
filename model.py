import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_ROWS, IMG_COLS = 80, 112


def dice_coef(pred, target, smooth=1.0):
    p = pred.reshape(pred.size(0), -1)
    t = target.reshape(target.size(0), -1)
    intersection = (p * t).sum(dim=1)
    return ((2.0 * intersection + smooth) / (p.sum(dim=1) + t.sum(dim=1) + smooth)).mean()


def dice_loss(pred, target):
    return -dice_coef(pred, target)


class ConvBnElu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding='same'):
        super().__init__()
        if padding == 'same':
            pad = tuple(k // 2 for k in kernel) if isinstance(kernel, tuple) else kernel // 2
        else:
            pad = 0
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, depth):
        super().__init__()
        d = depth
        self.branch1 = nn.Conv2d(in_ch, d // 4, 1, padding=0, bias=False)

        self.branch2 = nn.Sequential(
            ConvBnElu(in_ch, d * 3 // 8, 1),
            ConvBnElu(d * 3 // 8, d // 2, (1, 3)),
            nn.Conv2d(d // 2, d // 2, (3, 1), padding=(1, 0), bias=False),
        )

        self.branch3 = nn.Sequential(
            ConvBnElu(in_ch, d // 16, 1),
            ConvBnElu(d // 16, d // 8, (1, 5)),
            nn.Conv2d(d // 8, d // 8, (5, 1), padding=(2, 0), bias=False),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, d // 8, 1, bias=False),
        )

        self.bn  = nn.BatchNorm2d(d)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return self.elu(self.bn(out))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale=0.1):
        super().__init__()
        self.scale = scale
        self.conv  = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.elu   = nn.ELU(inplace=True)

    def forward(self, x):
        residual = self.bn(self.conv(x)) * self.scale
        return self.elu(self.skip(x) + residual)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, depth):
        super().__init__()
        self.inception = InceptionBlock(in_ch, depth)
        self.downsample = ConvBnElu(depth, depth, 3, stride=2)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        skip = self.inception(x)
        down = self.drop(self.downsample(skip))
        return skip, down


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, depth):
        super().__init__()
        self.skip_refine = ResidualBlock(skip_ch, skip_ch)
        self.inception   = InceptionBlock(in_ch + skip_ch, depth)
        self.drop        = nn.Dropout2d(0.5)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, self.skip_refine(skip)], dim=1)
        return self.drop(self.inception(x))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(1,   32)
        self.enc2 = EncoderBlock(32,  64)
        self.enc3 = EncoderBlock(64,  128)
        self.enc4 = EncoderBlock(128, 256)

        self.bottleneck = nn.Sequential(
            InceptionBlock(256, 512),
            nn.Dropout2d(0.5),
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(IMG_ROWS // 16 * IMG_COLS // 16, 1),
            nn.Sigmoid(),
        )

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64,  64)
        self.dec1 = DecoderBlock(64,  32,  32)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)

        # Detach so aux BCE gradients don't flow into decoder
        aux = self.aux_head(x.detach()).reshape(x.size(0))

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        seg = self.seg_head(x)
        return seg, aux


class PretrainedUNet(nn.Module):
    """
    Segmentation model using pretrained encoder from segmentation_models_pytorch.
    Supports ImageNet backbones (resnet34, se_resnext50_32x4d, efficientnet-b3, mobilenet_v2, resnet50)
    with dual output: segmentation mask (seg) + existence probability (aux).
    """
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', net_type='unet'):
        super().__init__()
        import segmentation_models_pytorch as smp
        
        if net_type == 'unetplusplus':
            model_cls = smp.UnetPlusPlus
        elif net_type == 'fpn':
            model_cls = smp.FPN
        elif net_type == 'manet':
            model_cls = smp.MAnet
        else:
            model_cls = smp.Unet

        self.model = model_cls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=1,
            activation='sigmoid'
        )

        in_ch = self.model.encoder.out_channels[-1]
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pad input so spatial H, W are exact multiples of 32
        H, W = x.shape[-2], x.shape[-1]
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32

        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_pad = x

        features = self.model.encoder(x_pad)
        bottleneck = features[-1].detach()
        aux = self.aux_head(bottleneck).reshape(x.size(0))
        try:
            decoder_output = self.model.decoder(*features)
        except TypeError:
            decoder_output = self.model.decoder(features)
        seg = self.model.segmentation_head(decoder_output)

        if pad_h > 0 or pad_w > 0:
            seg = seg[:, :, :H, :W].contiguous()

        return seg, aux


MODEL_REGISTRY = {
    'unet_inception': lambda: UNet(),
    'resnet34_plus': lambda: PretrainedUNet(encoder_name='resnet34', encoder_weights='imagenet', net_type='unetplusplus'),
    'se_resnext50': lambda: PretrainedUNet(encoder_name='se_resnext50_32x4d', encoder_weights='imagenet', net_type='unet'),
    'efficientnet_b3': lambda: PretrainedUNet(encoder_name='efficientnet-b3', encoder_weights='imagenet', net_type='unet'),
    'fpn_mobilenet': lambda: PretrainedUNet(encoder_name='mobilenet_v2', encoder_weights='imagenet', net_type='fpn'),
    'manet_resnet18': lambda: PretrainedUNet(encoder_name='resnet18', encoder_weights='imagenet', net_type='manet'),
}

MODEL_ALIASES = {
    'unet': 'unet_inception',
    'unet_inception': 'unet_inception',
    'resnet34': 'resnet34_plus',
    'resnet34_plus': 'resnet34_plus',
    'resnet34+': 'resnet34_plus',
    'resnext50': 'se_resnext50',
    'se_resnext50': 'se_resnext50',
    'efficientnet': 'efficientnet_b3',
    'efficientnet_b3': 'efficientnet_b3',
    'effnet_b3': 'efficientnet_b3',
    'mobilenet': 'fpn_mobilenet',
    'fpn_mobilenet': 'fpn_mobilenet',
    'mobilenet_fpn': 'fpn_mobilenet',
    'fpn': 'fpn_mobilenet',
    'manet': 'manet_resnet18',
    'manet_resnet18': 'manet_resnet18',
    'resnet18_manet': 'manet_resnet18',
    'manet_resnet50': 'manet_resnet18',
}


def resolve_model_name(name: str) -> str:
    cleaned = name.lower().strip()
    if cleaned in MODEL_ALIASES:
        return MODEL_ALIASES[cleaned]
    valid_keys = list(MODEL_REGISTRY.keys())
    valid_aliases = list(MODEL_ALIASES.keys())
    raise ValueError(f"Unknown model '{name}'. Valid options: {valid_keys} (or aliases: {valid_aliases})")


def get_model(model_name: str = 'resnet34_plus') -> nn.Module:
    canonical = resolve_model_name(model_name)
    return MODEL_REGISTRY[canonical]()


def get_model_spec(model_name: str = 'resnet34_plus') -> dict:
    canonical = resolve_model_name(model_name)
    m = get_model(canonical)
    total_params = sum(p.numel() for p in m.parameters())
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

    specs = {
        'unet_inception': {
            'display_name': 'Custom Inception UNet',
            'backbone': 'Custom Inception Encoders (Scratch)',
            'decoder': 'Inception Decoder with Residual Skip Refinements',
            'description': 'Custom multi-scale UNet architecture with multi-kernel Inception blocks and dual classification head.',
        },
        'resnet34_plus': {
            'display_name': 'ResNet-34 UNet++',
            'backbone': 'ResNet-34 (Pretrained ImageNet)',
            'decoder': 'UNet++ (Nested Dense Skip Pathways)',
            'description': 'Segmentation model using pretrained ResNet-34 encoder with UNet++ nested decoder and dual aux head.',
        },
        'se_resnext50': {
            'display_name': 'SE-ResNeXt-50 UNet',
            'backbone': 'SE-ResNeXt-50 32x4d (Pretrained ImageNet)',
            'decoder': 'UNet Decoder with Attention / Squeeze-and-Excitation',
            'description': 'High-capacity segmentation model with Squeeze-and-Excitation ResNeXt-50 encoder and dual aux head.',
        },
        'efficientnet_b3': {
            'display_name': 'EfficientNet-B3 UNet',
            'backbone': 'EfficientNet-B3 (Pretrained ImageNet)',
            'decoder': 'UNet Decoder',
            'description': 'High-efficiency segmentation model with compound-scaled EfficientNet-B3 encoder and dual aux head.',
        },
        'fpn_mobilenet': {
            'display_name': 'MobileNet-V2 FPN',
            'backbone': 'MobileNet-V2 (Pretrained ImageNet)',
            'decoder': 'Feature Pyramid Network (FPN)',
            'description': 'Lightweight & fast segmentation model using MobileNet-V2 with Feature Pyramid Network and dual aux head.',
        },
        'manet_resnet18': {
            'display_name': 'ResNet-18 MAnet',
            'backbone': 'ResNet-18 (Pretrained ImageNet)',
            'decoder': 'MAnet (Multi-scale Attention Network)',
            'description': 'Lightweight Multi-scale Attention Network with ResNet-18 encoder optimized for laptop training and fast inference.',
        },
    }

    info = specs[canonical]
    info['canonical_name'] = canonical
    info['total_params'] = total_params
    info['total_params_fmt'] = f"{total_params / 1e6:.2f}M"
    info['trainable_params'] = trainable_params
    info['input_shape'] = f"(1, {IMG_ROWS}, {IMG_COLS})"
    info['outputs'] = f"Mask Seg (1, {IMG_ROWS}, {IMG_COLS}) + Nerve Aux Prob (1,)"

    return info


if __name__ == '__main__':
    print("Testing Model Registry & Specifications:\n" + "=" * 50)
    for name in list(MODEL_REGISTRY.keys()):
        spec = get_model_spec(name)
        print(f"Model: {spec['display_name']} [{spec['canonical_name']}]")
        print(f"  Backbone: {spec['backbone']}")
        print(f"  Decoder:  {spec['decoder']}")
        print(f"  Params:   {spec['total_params_fmt']} ({spec['total_params']:,} total)")
        print(f"  Inputs:   {spec['input_shape']}")
        
        m = get_model(name)
        dummy = torch.randn(2, 1, IMG_ROWS, IMG_COLS)
        seg, aux = m(dummy)
        print(f"  Output shapes -> Seg: {seg.shape}, Aux: {aux.shape}\n")



