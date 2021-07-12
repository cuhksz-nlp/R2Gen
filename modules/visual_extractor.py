import torch
import torch.nn as nn
import torchvision.models as models
import argparse
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import trunc_normal_

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

# class VisualExtractor(nn.Module):
#     def __init__(self, args):
#         super(VisualExtractor, self).__init__()
#         self.visual_extractor = args.visual_extractor
#         self.pretrained = args.visual_extractor_pretrained
#         model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
#         modules = list(model.children())[:-2]
#         self.model = nn.Sequential(*modules)
#         self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

#     def forward(self, images):
#         patch_feats = self.model(images)
#         avg_feats = self.avg_fnt(patch_feats)
#         avg_feats = avg_feats.squeeze()
#         avg_feats = avg_feats.reshape(-1, patch_feats.size(1))
#         batch_size, feat_size, _, _ = patch_feats.shape
#         patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
#         return patch_feats, avg_feats


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        # model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        # modules = list(model.children())[:-2]
        # self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
        
        # if self.visual_extractor == 'deit_base_patch16_384':
        #     self.visual_encoder = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_384', pretrained = self.pretrained)
        # if self.visual_extractor == 'deit_base_patch16_224':
        #     self.visual_encoder = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained = self.pretrained)
        
        # self.visual_encoder = torch.hub.load('C:/Users/91951/.cache/torch/hub/facebookresearch_deit_main', 'deit_base_patch16_224',
        #                                      source = 'local', pretrained = self.pretrained)
        
        self.visual_encoder = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        self.visual_encoder.default_cfg = _cfg()
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        self.visual_encoder.load_state_dict(checkpoint["model"])

    def forward(self, images):
        patch_feats = self.visual_encoder.patch_embed(images)
        patch_feats = self.visual_encoder.pos_drop(patch_feats)
        for i in self.visual_encoder.blocks:
            patch_feats = i(patch_feats)
        
        patch_feats = patch_feats.squeeze(0)
        # patch_feats = patch_feats.unflatten(1, (14, 14))
        # print(patch_feats.shape)
        
        # patch_feats = patch_feats.permute(0, 3, 1, 2)
        # print(patch_feats.shape)
        # avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        # batch_size, feat_size, _, _ = patch_feats.shape
        # patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # return patch_feats, avg_feats
        return patch_feats

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    # parser.add_argument('--visual_extractor', type=str, default='deit_base_patch16_224', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default = False, help='whether to load the pretrained visual extractor')

    args = parser.parse_args()
    
    return args
    

# if __name__ == "__main__":
    
#     img = torch.randn((1, 3, 224, 224))
#     args = parse_agrs()
#     vis = VisualExtractor(args)
#     patch_feats = vis(img)
#     patch_feats, avg_feats = vis(img)
#     print('patch_feats out ', patch_feats.shape)
#     print('avg_features out ', avg_feats.shape)
    
    
    
    
