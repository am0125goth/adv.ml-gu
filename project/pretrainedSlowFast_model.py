import torch
import torch.nn as nn

class PretrainedSlowFastBackbone(nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()

        self.slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        self.feature_dim = 2304
        self.slowfast.blocks = self.slowfast.blocks[:-1]

        self._fix_pooling_layers()
        
        if freeze_backbone:
            for param in self.slowfast.parameters():
                param.requires_grad = False

    def _fix_pooling_layers(self):
        "Replace problematic pooling layer"
        for name, module in self.slowfast.named_modules():
            if isinstance(module, nn.AvgPool3d) and module.kernel_size == (8, 7, 7):
                parent = self._get_parent_module(name)
                attr_name = attr_name = name.split('.')[-1]
                # Replace with adaptive pooling
                setattr(parent, attr_name, nn.AdaptiveAvgPool3d((1, 1, 1)))

    def _get_parent_module(self, module_path):
        """Helper to get the parent module of a given module path"""
        parts = module_path.split('.')
        parent = self.slowfast
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent
        
    
    def forward(self, x):
        """
        Args:
            x: tensor of shape [B, C, T, H, W]
        Returns:
            features: features of shape (B, 2304)
        """
        x = x.permute(0, 2, 1, 3, 4)
        batch_size, channels, num_frames, height, width = x.shape
        
        slow = x[:, :, ::4, :, :]
        fast = x

        features = self.slowfast([slow, fast])
        
        if len(features.shape) > 2:
            features = features.mean(dim=[-3, -2, -1])

        return features

class MultiHeadSlowFastPretrained(nn.Module):
    """
    Pre-trained SlowFast with multi-head classification.
    Designed to be comparable with MultiHeadSlowFast.
    """
    def __init__(self, num_verbs, num_objects, num_actions, freeze_backbone=False):
        super().__init__()

        self.backbone = PretrainedSlowFastBackbone(freeze_backbone=freeze_backbone)
        feature_dim = self.backbone.feature_dim

        self.feature_projection = nn.Linear(feature_dim, 512)

        # IMPORTANT: Simple heads to match from-scratch model exactly
        self.verb_head = nn.Linear(512, num_verbs)
        self.object_head = nn.Linear(512, num_objects)
        self.action_head = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        Args:
            x: Video tensor of shape (B, C, T, H, W)
        Returns:
            verb_logits: (B, num_verbs)
            object_logits: (B, num_objects)
            action_logits: (B, num_actions)
        """
        features = self.backbone(x)
        features = self.feature_projection(features)

        verb_logits = self.verb_head(features)
        object_logits = self.object_head(features)
        action_logits = self.action_head(features)

        return verb_logits, object_logits, action_logits
        