import json

forbidden_words = [
    'previous', 'again', 'change', 'remain', 'past', 'prior', 'old', 'new', 'now', 'current',
    'pa ', 'ap ', 'lateral',
    ' and ', ' or ', ' but ', ' with ', ' without ', ' which ', ' where ', ' that ', ' although ',
    'left', 'right', 'upper', 'lower',
    'this',
    'svc', 'bb', 'chf',
    ' ize '
]

def reference_sentences_loader(ann_path):
    with open(ann_path, 'r') as f:
        data = json.load(f)
    reference_reports = [item["report"] for item in data["train"]]
    
    reference_sentences = []
    for report in reference_reports:
        reference_sentences.extend(report.split('.'))

    # 1st unique
    reference_sentences = list(set(reference_sentences))

    # adjust
    reference_sentences = [sentence.replace('\n', '') for sentence in reference_sentences]
    reference_sentences = [sentence.lower() for sentence in reference_sentences]
    reference_sentences = [' '.join(sentence.split()) for sentence in reference_sentences]

    # 2nd unique
    reference_sentences = list(set(reference_sentences))

    # sample
    reference_sentences = reference_sentences[:int(len(reference_sentences)/10)]

    # check
    reference_sentences = [sentence for sentence in reference_sentences if all(char.isalpha() or char.isspace() for char in sentence)]
    reference_sentences = [sentence for sentence in reference_sentences if sentence != '']
    for word in forbidden_words:
        reference_sentences = [sentence for sentence in reference_sentences if word not in sentence]
    reference_sentences = [sentence for sentence in reference_sentences if len(sentence) > 15]
    reference_sentences = [sentence for sentence in reference_sentences if len(sentence) < 50]

    return reference_sentences
