"""Print parameter counts for the four non-DINO SenseCV two-head backbones.

Run in Kaggle or locally after installing torch and timm:
    python scripts/show_two_head_parameter_counts.py

This only constructs the architectures with randomly initialized weights; it
does not load data, train, or recreate any dataset.
"""
import torch
import torch.nn as nn

try:
    import timm
except ImportError as error:
    raise SystemExit("Install timm first: python -m pip install timm") from error


MODELS = {
    "MobileNetV1": "mobilenetv1_100.ra4_e3600_r224_in1k",
    "EfficientNet-B0": "efficientnet_b0.ra_in1k",
    "MobileNetV4": "mobilenetv4_conv_small.e2400_r224_in1k",
    "ResNet-18": "resnet18.a1_in1k",
    "ResNet-34": "resnet34.a1_in1k",
    "ResNet-50": "resnet50.a1_in1k",
    "ResNet-101": "resnet101.a1_in1k",
    "ResNet-152": "resnet152.a1_in1k",
}


class TwoHeadClassifier(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0
        )
        feature_dim = self.backbone.num_features
        self.obstacle_head = nn.Linear(feature_dim, 2)
        self.lateral_head = nn.Linear(feature_dim, 3)

    def forward(self, images):
        features = self.backbone(images)
        return self.obstacle_head(features), self.lateral_head(features)


def count(parameters):
    return sum(parameter.numel() for parameter in parameters)


def main():
    print(f"{'Model':<18} {'Total parameters':>18} {'Trainable':>18}")
    print("-" * 58)
    for name, backbone_name in MODELS.items():
        model = TwoHeadClassifier(backbone_name)
        total = count(model.parameters())
        trainable = count(p for p in model.parameters() if p.requires_grad)
        print(f"{name:<18} {total:>18,} {trainable:>18,}")


if __name__ == "__main__":
    main()
