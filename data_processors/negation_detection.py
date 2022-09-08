# exp setup
import spacy
from negspacy.termsets import termset
from negspacy.negation import Negex


class NegationDetection(object):
    def __init__(self, args):
        self.args = args
        self.nlp0 = spacy.load("en_core_sci_sm")
        self.termset = termset("en_clinical")
        self.nlp1 = spacy.load("en_ner_bc5cdr_md")
        # self.entities = ["DISEASE", "TEST", "TREATMENT", "NEG_ENTITY"]
        self.nlp0.add_pipe("negex", config={"ent_types": ["DISEASE", "TEST", "TREATMENT", "ORGAN", "NEG_ENTITY"]})
        self.nlp1.add_pipe("negex", config={"ent_types": ["DISEASE", "TEST", "TREATMENT", "ORGAN", "NEG_ENTITY"]})

    def get_lemmatize_doc_object(self, report):
        self.termset.add_patterns({
            "pseudo_negations": ["normal", "stable"],
            # "preceding_negations": ["normal", "stable"],
            # "following_negations": ["normal", "stable"],
        })
        # print(self.termset.get_patterns())
        for sentence in report["report"].split('.'):
            lem_sentence = self.lemmatize(sentence, self.nlp0)
            self.print_negation(self.nlp1(lem_sentence))

    def lemmatize(self, report, nlp):
        doc = nlp(report)
        lemNote = [wd.lemma_ for wd in doc]
        return " ".join(lemNote)

    def print_negation(self, doc):
        for entity in doc.ents:
            print(entity.text, entity._.negex)
