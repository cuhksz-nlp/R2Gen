import json
import re
from collections import Counter


class Tokenizer(object):
    def __init__(self, args, data_processor):
        # exp setup
        self.exp = args.exp
        self.data_processor = data_processor
        self.val_test_partial_data = args.val_test_partial_data
        #############################################################
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        # exp setup
        # partial dataset for test and val
        if self.val_test_partial_data == 1:
            normal_ann = [asample
                          for ids, sample in self.data_processor.iu_mesh_impression_split['val'].items()
                          if sample['iu_mesh'] == 'normal'
                          for asample in self.ann['val'] if asample['id'] == ids]
            self.ann['val'] = normal_ann
            normal_ann = [asample
                          for ids, sample in self.data_processor.iu_mesh_impression_split['test'].items()
                          if sample['iu_mesh'] == 'normal'
                          for asample in self.ann['test'] if asample['id'] == ids]
            self.ann['test'] = normal_ann
        elif self.val_test_partial_data == 2:
            normal_ann = [asample
                          for ids, sample in self.data_processor.iu_mesh_impression_split['val'].items()
                          if sample['iu_mesh'] == 'No Indexing'
                          for asample in self.ann['val'] if asample['id'] == ids]
            self.ann['val'] = normal_ann
            normal_ann = [asample
                          for ids, sample in self.data_processor.iu_mesh_impression_split['test'].items()
                          if sample['iu_mesh'] == 'No Indexing'
                          for asample in self.ann['test'] if asample['id'] == ids]
            self.ann['test'] = normal_ann
        elif self.val_test_partial_data == 3:
            normal_ann = [asample
                          for ids, sample in self.data_processor.iu_mesh_impression_split['val'].items()
                          if sample['iu_mesh'] != 'normal' and sample['iu_mesh'] != 'No Indexing'
                          for asample in self.ann['val'] if asample['id'] == ids]
            self.ann['val'] = normal_ann
            normal_ann = [asample
                          for ids, sample in self.data_processor.iu_mesh_impression_split['test'].items()
                          if sample['iu_mesh'] != 'normal' and sample['iu_mesh'] != 'No Indexing'
                          for asample in self.ann['test'] if asample['id'] == ids]
            self.ann['test'] = normal_ann
        elif self.val_test_partial_data == 4:
            normal_ann = [asample
                          for ids, sample in self.data_processor.iu_mesh_impression_split['val'].items()
                          if sample['iu_mesh'] != 'normal'
                          for asample in self.ann['val'] if asample['id'] == ids]
            self.ann['val'] = normal_ann
            normal_ann = [asample
                          for ids, sample in self.data_processor.iu_mesh_impression_split['test'].items()
                          if sample['iu_mesh'] != 'normal'
                          for asample in self.ann['test'] if asample['id'] == ids]
            self.ann['test'] = normal_ann
        ###################################################################################################################
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            # exp setup
            self.data_processor.iu_mesh_impression_split['train'][example['id']]['impression'] = \
                self.clean_report(self.data_processor.iu_mesh_impression_split['train'][example['id']]['impression'])
            reports_with_additional_info = self.data_processor.get_reports_by_exp(self.exp, 'train', example['id'],
                                                                                  self.clean_report(example['report']))
            tokens = reports_with_additional_info.split()
            #################################################
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        # exp setup
        # vocab = [k for k, v in counter.items() if v >= self.threshold or "<" in k] + ['<unk>']
        #######################################################################################
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, data_processor, exp, split, r2gen_id, report):
        # exp setup
        if 3 < exp < 7:
            data_processor.iu_mesh_impression_split[split][r2gen_id]['impression'] = \
                self.clean_report(data_processor.iu_mesh_impression_split[split][r2gen_id]['impression'])
        tokens = data_processor.get_reports_by_exp(exp, split, r2gen_id, self.clean_report(report)).split()
        ###################################################################################################
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids, remove_annotation):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                    # exp setup
                    # remove MeSH annotization
                    tkn = self.idx2token[idx]
                    if remove_annotation == 1 and '<sep>' in tkn:
                        break
                    #################################
                    txt += tkn
            else:
                break
        return txt

    def decode_batch(self, ids_batch, remove_annotation):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids, remove_annotation))
        return out
