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
        self.train_sample = args.train_sample
        self.val_sample = args.val_sample
        self.test_sample = args.test_sample
        ####################################
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        # exp setup
        # random sample for smaller dataset
        if self.split == 'train' and self.train_sample > 0:
            self.examples = random.sample(self.ann[self.split], self.train_sample)
        elif self.split == 'val' and self.val_sample > 0:
            self.examples = random.sample(self.ann[self.split], self.val_sample)
        elif self.split == 'test' and self.test_sample > 0:
            self.examples = random.sample(self.ann[self.split], self.test_sample)
        else:
            self.examples = self.ann[self.split]
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
