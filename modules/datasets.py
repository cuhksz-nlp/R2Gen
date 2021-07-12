import os
import json
import torch
import torch.nn as nn
from PIL import Image
from functools import partial
from torch.utils.data import Dataset
from timm.models.vision_transformer import VisionTransformer, _cfg

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tensorizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        self.visual_encoder = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer = partial(nn.LayerNorm, eps=1e-6))
        
        self.visual_encoder.default_cfg = _cfg()
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        self.visual_encoder.load_state_dict(checkpoint["model"])
    
    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        # image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            # image_2 = self.transform(image_2)
        # image = torch.stack((image_1, image_2), 0)
        patch_feats = self.visual_encoder.patch_embed(image_1)
        patch_feats = self.visual_encoder.pos_drop(patch_feats)
        for i in self.visual_encoder.blocks:
            patch_feats = i(patch_feats)
        
        patch_feats = patch_feats.squeeze(0)
        
        example = self.tensorizer.tensorize_example(self.examples[idx]['report'], patch_feats, 
                                                    text = self.examples[idx]['tags'])
        return patch_feats, example


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample




# import os
# import json
# import torch
# from PIL import Image
# from torch.utils.data import Dataset


# class BaseDataset(Dataset):
#     def __init__(self, args, tokenizer, split, transform=None):
#         self.image_dir = args.image_dir
#         self.ann_path = args.ann_path
#         self.max_seq_length = args.max_seq_length
#         self.split = split
#         self.tokenizer = tokenizer
#         self.transform = transform
#         self.ann = json.loads(open(self.ann_path, 'r').read())

#         self.examples = self.ann[self.split]
#         for i in range(len(self.examples)):
#             self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
#             self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

#     def __len__(self):
#         return len(self.examples)


# class IuxrayMultiImageDataset(BaseDataset):
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         image_id = example['id']
#         image_path = example['image_path']
#         image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
#         image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
#         if self.transform is not None:
#             image_1 = self.transform(image_1)
#             image_2 = self.transform(image_2)
#         image = torch.stack((image_1, image_2), 0)
#         report_ids = example['ids']
#         report_masks = example['mask']
#         seq_length = len(report_ids)
#         sample = (image_id, image, report_ids, report_masks, seq_length)
#         return sample


# class MimiccxrSingleImageDataset(BaseDataset):
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         image_id = example['id']
#         image_path = example['image_path']
#         image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)
#         report_ids = example['ids']
#         report_masks = example['mask']
#         seq_length = len(report_ids)
#         sample = (image_id, image, report_ids, report_masks, seq_length)
#         return sample
