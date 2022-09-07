# exp setup
import spacy
from negspacy.termsets import termset


class NegationDetection(object):
    def __init__(self, args):
        self.args = args
        self.nlp0 = spacy.load("en_core_sci_sm")
        self.nlp1 = spacy.load("en_ner_bc5cdr_md")
        self.entities = ["DISEASE", "ORGAN", "NEG_ENTITY"]

    def get_lemmatize_doc_object(self, report):
        termset("en_clinical").add_patterns({"pseudo_negations": ["normal", "stable"],})
        lem_report = self.lemmatize(report, self.nlp0)
        return self.nlp1(lem_report)

    def lemmatize(self, report, nlp):
        doc = nlp(report)
        lemNote = [wd.lemma_ for wd in doc]
        return " ".join(lemNote)

    def print_negation(self, doc):
        for entity in doc.ents:
            print(entity.text, entity._.negex)
