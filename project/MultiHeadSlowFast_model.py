import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiHeadSlowFast(nn.Module):
    def __init__(self, 
                 num_verbs, 
                 num_objects, 
                 num_actions, 
                 slow_temporal_stride=16, 
                 fast_temporal_stride=2, 
                 slow_channel_multiplier=1.0, 
                 fast_channel_multiplier=0.125, 
                 fusion_method='concat',
                 feature_dim=512):

        super().__init__()
        #share backbone with SlowFast
        self.backbone = SlowFast(
            slow_temporal_stride=slow_temporal_stride,
            fast_temporal_stride=fast_temporal_stride,
            slow_channel_multiplier=slow_channel_multiplier,
            fast_channel_multiplier=fast_channel_multiplier,
            fusion_method=fusion_method,
            feature_dim=feature_dim
        )
 
        self.verb_head = nn.Linear(feature_dim, num_verbs)
        self.object_head = nn.Linear(feature_dim, num_objects)
        self.action_head = nn.Linear(feature_dim, num_actions)
        
    def forward(self, x):
        #extract shared features
        features = self.backbone(x)
        
        #seperate predictions
        verb_logits = self.verb_head(features)
        object_logits = self.object_head(features)
        action_logits = self.action_head(features)
        
        return verb_logits, object_logits, action_logits

class SlowFast(nn.Module):
    def __init__(self, slow_temporal_stride, fast_temporal_stride, slow_channel_multiplier, fast_channel_multiplier, fusion_method='concat', feature_dim=512):
        super().__init__()

        self.slow_temporal_stride = slow_temporal_stride
        self.fast_temporal_stride = fast_temporal_stride
        self.fusion_method = fusion_method

        #slow pathway
        self.slow_path = ResNet3D(
            stem_temporal_stride=slow_temporal_stride,
            channel_multiplier=slow_channel_multiplier
        )

        #fast pathway
        self.fast_path = ResNet3D(
            stem_temporal_stride=fast_temporal_stride,
            channel_multiplier=fast_channel_multiplier
        )

        self.lateral_connections = nn.ModuleList([
            LateralConnections(64, 8),
            LateralConnections(128, 16),
            LateralConnections(256, 32),
            LateralConnections(512, 64)
        ])

        in_features = 512 + 64 if fusion_method=='concat' else 512
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        #shape of x : [B, T, C, H, W]
        B, T, C, H, W = x.shape

        #prepare slow and fast inputs
        slow_idx = torch.arange(0, T, self.slow_temporal_stride)
        fast_idx = torch.arange(0, T, self.fast_temporal_stride)

        slow_input = x[:, slow_idx] # shape of x : [B, T_slow, C, H, W]
        fast_input = x[:, fast_idx] # shape of x : [B, T_fast, C, H, W]

        #permute to [B, C, T, H, W] for 3D conv
        slow_input = slow_input.permute(0, 2, 1, 3, 4)
        fast_input = fast_input.permute(0, 2, 1, 3, 4)

        slow_feat = self.slow_path.stem(slow_input)
        fast_feat = self.fast_path.stem(fast_input)

        #ResNet for stages with lateral connections
        
        slow_feat = self.slow_path.res2(slow_feat)
        fast_feat = self.fast_path.res2(fast_feat)
        
        lateral_feat = self.lateral_connections[0](fast_feat)
        T_slow = slow_feat.size(2)
        lateral_feat = F.interpolate(lateral_feat, size=(T_slow, lateral_feat.size(3), lateral_feat.size(4)), mode='trilinear', align_corners=False)
        slow_feat = slow_feat + lateral_feat
                       
        slow_feat = self.slow_path.res3(slow_feat)
        fast_feat = self.fast_path.res3(fast_feat)
        
        lateral_feat = self.lateral_connections[1](fast_feat)
        T_slow = slow_feat.size(2)
        lateral_feat = F.interpolate(lateral_feat, size=(T_slow, lateral_feat.size(3), lateral_feat.size(4)), mode='trilinear', align_corners=False)
        slow_feat = slow_feat + lateral_feat
        
        slow_feat = self.slow_path.res4(slow_feat) 
        fast_feat = self.fast_path.res4(fast_feat)
        
        lateral_feat = self.lateral_connections[2](fast_feat)
        T_slow = slow_feat.size(2)
        lateral_feat = F.interpolate(lateral_feat, size=(T_slow, lateral_feat.size(3), lateral_feat.size(4)), mode='trilinear', align_corners=False)
        slow_feat = slow_feat + lateral_feat
        
        slow_feat = self.slow_path.res5(slow_feat)
        fast_feat = self.fast_path.res5(fast_feat)
                                        
        lateral_feat = self.lateral_connections[3](fast_feat)
        T_slow = slow_feat.size(2)
        lateral_feat = F.interpolate(lateral_feat, size=(T_slow, lateral_feat.size(3), lateral_feat.size(4)), mode='trilinear', align_corners=False)
        slow_feat = slow_feat + lateral_feat
        
        #fusion
        if self.fusion_method=='concat':
            T_slow = slow_feat.size(2)
            fast_feat = F.interpolate(fast_feat, size=(T_slow, lateral_feat.size(3), lateral_feat.size(4)), mode='trilinear', align_corners=False)
            fused_feat = torch.cat([slow_feat, fast_feat], dim=1)
        else:
            fused_feat = slow_feat + fast_feat
        
        features = self.feature_extractor(fused_feat)
        return features


class LateralConnections(nn.Module):
    def __init__(self, slow_channels, fast_channels):
        super().__init__()
        self.conv = nn.Conv3d(fast_channels, slow_channels, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.bn = nn.BatchNorm3d(slow_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResNet3D(nn.Module):
    def __init__(self, stem_temporal_stride, channel_multiplier):
        super().__init__()
    
        self.stem = nn.Sequential(
            nn.Conv3d(3, int(64*channel_multiplier), kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(int(64*channel_multiplier)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0,1, 1))
        )

        self.res2 = self._make_layer(int(64*channel_multiplier), int(64*channel_multiplier), 3)
        self.res3 = self._make_layer(int(64*channel_multiplier), int(128*channel_multiplier), 4, stride=2)
        self.res4 = self._make_layer(int(128*channel_multiplier), int(256*channel_multiplier), 6, stride=2)
        self.res5 = self._make_layer(int(256*channel_multiplier), int(512*channel_multiplier), 3, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(Bottleneck3D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Bottleneck3D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        return x


class Bottleneck3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels //4)

        self.conv2 = nn.Conv3d(out_channels // 4, out_channels // 4, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels // 4)

        self.conv3 = nn.Conv3d(out_channels // 4, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm3d(out_channels))
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
