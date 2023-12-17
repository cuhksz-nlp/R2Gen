import json
import os
import openai
from modules.reference_sentences_loader import reference_sentences_loader
from modules.sentence_refiner import sentence_refiner


class TextEmbeddingModel():
    def __init__(self, ann_path, data_path, record_dir, record_file, api_key):
        self.ann_path = ann_path
        self.data_path = data_path
        self.record_dir = record_dir
        self.record_file = record_file
        self.api_key = api_key

        self.reference_sentences = []
        self.__load_reference_sentences()

        self.ground_truth_reports = {'train': [], 'val': [], 'test': []}
        self.target_reports = {'train': [], 'val': [], 'test': []}
        self.__load_data()

        self.refined_reports = {'train': [], 'val': [], 'test': []}
        self.start_idx = {'train': 0, 'val': 0, 'test': 0}
        self.__load_refined_reports()

        self.client = None
        self.__create_client()

    def __load_reference_sentences(self):
        self.reference_sentences = reference_sentences_loader(self.ann_path)
        print()
        print('about reference sentences ...')
        print('length:', len(self.reference_sentences))
        print('type:', type(self.reference_sentences))
        print()

    def __load_data(self):
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        for t in ['train', 'val', 'test']:
            self.ground_truth_reports[t] = [item['ground_truth'] for item in data[t]]
            self.target_reports[t] = [item['report'] for item in data[t]]

    def __load_refined_reports(self):
        if os.path.exists(self.record_dir + '/' + self.record_file):
            with open(self.record_dir + '/' + self.record_file, 'r') as f:
                self.refined_reports = json.load(f)
            for t in ['train', 'val', 'test']:
                self.start_idx[t] = len(self.refined_reports[t])

    def __create_client(self):
        self.client = openai.OpenAI(
            api_key=self.api_key
        )

    def __report_to_sentences(self, report):
        sentences = report.split(' . ')
        sentences[-1] = sentences[-1].replace(' .', '')
        return sentences

    def __refine_sentence(self, sentence):
        return sentence_refiner(self.client, sentence, self.reference_sentences)

    def __sentences_to_report(self, sentences):
        report = ' . '.join(sentences)
        report += ' .'
        return report

    def __store_refined_reports(self):
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
        with open(self.record_dir + '/' + self.record_file, 'w') as f:
            json.dump(self.refined_reports, f)

    def refine(self, dataset_type):
        result = {'train': [], 'val': [], 'test': []}
        for t in dataset_type:
            for i, target_report in enumerate(self.target_reports[t]):
                if i < self.start_idx[t]:
                    continue
                target_sentences = self.__report_to_sentences(target_report)
                refined_sentences = []
                for target_sentence in target_sentences:
                    refined_sentence = self.__refine_sentence(target_sentence)
                    refined_sentences.append(refined_sentence)
                refined_report = self.__sentences_to_report(refined_sentences)
                self.refined_reports[t].append(refined_report)
                self.__store_refined_reports()
                print()
                print('total:', len(self.target_reports[t]))
                print('current:', i)
                print('ground_truth:', self.ground_truth_reports[t][i])
                print('target_report:', target_report)
                print('refined_report:', refined_report)
            result[t] = [{'ground_truth': gt, 'report': re} for gt, re in zip(self.ground_truth_reports[t], self.refined_reports[t])]
        return result
