import json

def contains_number(string):
    return any(char.isdigit() for char in string)

def sanity_check(data):
    rtn_data = []
    tmp_data = []
    data = list(set(data))
    for item in data:
        if len(item)==0: continue
        if item[0]==".": continue
        if item[0]==" ": continue
        else: tmp_data.append(item)
    if len(tmp_data) > 2000:
      for report in tmp_data:
         if len(report) < 100: continue
         if contains_number(report): continue
         else: rtn_data.append(report)
    else:rtn_data = tmp_data
    return rtn_data

def report_to_sentences(report):
    sentences = report.split(' . ')
    sentences[-1] = sentences[-1][:-2]
    return sentences


def report_set_to_sentences(report_set):
    sentences = []
    for report in report_set:
        sentences += report_to_sentences(report)
    return sentences


def create_reference_report_set(data_path):
    with open(data_path) as f:
        data = json.load(f)
    training_data_list = data["train"]
    reference_report_set = []
    for item in training_data_list:
        reference_report = item["report"]
        reference_report = reference_report.lower().replace('.', ' .')
        reference_report = reference_report.replace("/", " ")
        if "XXXX" in reference_report: continue
        elif "xxxx" in reference_report: continue
        elif "___" in reference_report: continue
        elif "\n" in reference_report:
            reference_report = reference_report.replace("\n", "")
            reference_report_set.append(reference_report)
        else: reference_report_set.append(reference_report)
    return reference_report_set


def reference_sentences_loader(data_path):
    reference_report_set = create_reference_report_set(data_path)
    reference_sentences = report_set_to_sentences(reference_report_set)
    reference_sentences = sanity_check(reference_sentences)
    return reference_sentences
