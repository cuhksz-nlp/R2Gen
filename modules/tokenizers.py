import torch
import random

class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70, 
            max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3,
            is_train=True):
        
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))

    def tensorize_example(self, text_a, img_feat, text_b=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len)) # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1 
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                        (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention 
        # for caption as caption will have full attention on image. 
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption
        attention_mask[c_start : c_end, c_start : c_end].copy_(self._triangle_mask[0 : seq_a_len, 0 : seq_a_len])
        # full attention for L-L, R-R
        attention_mask[l_start : l_end, l_start : l_end] = 1
        attention_mask[r_start : r_end, r_start : r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start : c_end, l_start : l_end] = 1
        attention_mask[c_start : c_end, r_start : r_end] = 1
        # full attention for L-R:
        attention_mask[l_start : l_end, r_start : r_end] = 1
        attention_mask[r_start : r_end, l_start : l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids)
        return (input_ids, attention_mask, segment_ids, img_feat, masked_pos)





# import json
# import re
# from collections import Counter


# class Tokenizer(object):
#     def __init__(self, args):
#         self.ann_path = args.ann_path
#         self.threshold = args.threshold
#         self.dataset_name = args.dataset_name
#         if self.dataset_name == 'iu_xray':
#             self.clean_report = self.clean_report_iu_xray
#         else:
#             self.clean_report = self.clean_report_mimic_cxr
#         self.ann = json.loads(open(self.ann_path, 'r').read())
#         self.token2idx, self.idx2token = self.create_vocabulary()

#     def create_vocabulary(self):
#         total_tokens = []
#         for example in self.ann['train']:
#             tokens = self.clean_report(example['report']).split()
#             for token in tokens:
#                 total_tokens.append(token)

#         counter = Counter(total_tokens)
#         vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
#         vocab.sort()
#         token2idx, idx2token = {}, {}
#         for idx, token in enumerate(vocab):
#             token2idx[token] = idx + 1
#             idx2token[idx + 1] = token
#         return token2idx, idx2token

#     def clean_report_iu_xray(self, report):
#         report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
#             .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
#             .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
#                                         replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report

#     def clean_report_mimic_cxr(self, report):
#         report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
#             .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
#             .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
#             .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
#             .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
#                                         .replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report

#     def get_token_by_id(self, id):
#         return self.idx2token[id]

#     def get_id_by_token(self, token):
#         if token not in self.token2idx:
#             return self.token2idx['<unk>']
#         return self.token2idx[token]

#     def get_vocab_size(self):
#         return len(self.token2idx)

#     def __call__(self, report):
#         tokens = self.clean_report(report).split()
#         ids = []
#         for token in tokens:
#             ids.append(self.get_id_by_token(token))
#         ids = [0] + ids + [0]
#         return ids

#     def decode(self, ids):
#         txt = ''
#         for i, idx in enumerate(ids):
#             if idx > 0:
#                 if i >= 1:
#                     txt += ' '
#                 txt += self.idx2token[idx]
#             else:
#                 break
#         return txt

#     def decode_batch(self, ids_batch):
#         out = []
#         for ids in ids_batch:
#             out.append(self.decode(ids))
#         return out
