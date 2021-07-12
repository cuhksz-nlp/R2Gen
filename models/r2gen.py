import torch
import torch.nn as nn
import numpy as np
import argparse 

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

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
#     targets = torch.randint(1, 100, (1, 224))
#     args = parse_agrs()
#     vis = VisualExtractor(args)
#     patch_feats, avg_feats = vis(img)
#     print('patch_feats out ', patch_feats.shape)
#     print('avg_features out ', avg_feats.shape)

