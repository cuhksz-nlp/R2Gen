import os
import json
import random
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, data_processor, tokenizer, split, transform=None):
        # exp setup
        self.exp = args.exp
        self.data_processor = data_processor
        ####################################
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        # exp setup
        self.examples = self.ann[self.split]
        # random sample for smaller dataset
        # if args.dataset_name == 'iu_xray':
        #     self.examples = random.sample(self.ann[self.split], 10)
        # else:
        #     if self.split == 'train':
        #         self.examples = random.sample(self.ann[self.split], 5000)
        #     elif self.split == 'val':
        #         self.examples = random.sample(self.ann[self.split], 1000)
        #     elif self.split == 'test':
        #         self.examples = random.sample(self.ann[self.split], 2000)
        for i in range(len(self.examples)):
            self.examples[i]['annotated_report'] = \
                data_processor.get_reports_by_exp(self.exp, self.split, self.examples[i]['id'],
                                                  tokenizer.clean_report(self.examples[i]['report']))
            self.examples[i]['ids'] = \
                tokenizer(self.data_processor, self.exp, self.split, self.examples[i]['id'],
                          self.examples[i]['report'])[:self.max_seq_length]
            ##################################################################
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


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
