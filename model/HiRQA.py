import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Fom OpenAI CLIP file
class AttentionPool2d(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        #Commenting the positional embedding
        # self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove the final FC layer
        self.out_channels = 2048
    
    def forward(self, x):
        return self.features(x)

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Backbone, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  
        self.out_channels = 512
    
    def forward(self, x):
        return self.features(x)

class HiRQA(nn.Module):
    def __init__(self, model = 'HiRQA'):
        super().__init__()

        if model == "HiRQA":          
            self.backbone = ResNetBackbone(pretrained=True)
        elif model == "HiRQA-S":          
            self.backbone = ResNet18Backbone(pretrained=True)
        
        self.global_pool_quality = AttentionPool2d(self.backbone.out_channels, 16, 1024)

        self.fc = nn.Sequential(
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        features_backbone = self.backbone(x) 
        pooled_quality = self.global_pool_quality(features_backbone).flatten(1)
        score = (self.fc(pooled_quality)).squeeze(-1)
        score = torch.sigmoid(score)
        return score