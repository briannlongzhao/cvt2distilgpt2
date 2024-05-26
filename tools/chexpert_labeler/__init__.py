from .loader import Loader
from .stages import Extractor, Classifier, Aggregator
from .constants import *
import bioc
import re
from negbio.pipeline import text2bioc, ssplit, section_split

class ChexpertLabeler:
    def __init__(self):
        self.loader = Loader(
            reports_path=None,
            sections_to_extract=[],
            extract_strict=False,
        )
        self.extractor = Extractor(
            mention_phrases_dir=Path(__file__).parent/"phrases/mention",
            unmention_phrases_dir=Path(__file__).parent/"phrases/unmention",
            verbose=False,
        )
        self.classifier = Classifier(
            pre_negation_uncertainty_path=Path(__file__).parent/"patterns/pre_negation_uncertainty.txt",
            negation_path=Path(__file__).parent/"patterns/negation.txt",
            post_negation_uncertainty_path=Path(__file__).parent/"patterns/post_negation_uncertainty.txt",
            verbose=False
        )
        self.aggregator = Aggregator(
            categories=CATEGORIES,
            verbose=False
        )
        self.splitter = ssplit.NegBioSSplitter(newline=False)

    def clean(self, report):
        """Clean the report text."""
        lower_report = report.lower()
        # Change `and/or` to `or`.
        corrected_report = re.sub('and/or',
                                  'or',
                                  lower_report)
        # Change any `XXX/YYY` to `XXX or YYY`.
        corrected_report = re.sub('(?<=[a-zA-Z])/(?=[a-zA-Z])',
                                  ' or ',
                                  corrected_report)
        # Clean double periods
        clean_report = corrected_report.replace("..", ".")
        # Insert space after commas and periods.
        clean_report = clean_report.translate(self.punctuation_spacer)
        # Convert any multi white spaces to single white spaces.
        clean_report = ' '.join(clean_report.split())
        # Remove empty sentences
        clean_report = re.sub(r'\.\s+\.', '.', clean_report)

        return clean_report

    def get_label(self, report):
        # Load reports in place.
        self.loader.load_single_report(report)

        # collection = bioc.BioCCollection()
        # clean_report = self.clean(report)
        # document = text2bioc.text2document(str(i), clean_report)
        # split_document = self.splitter.split_doc(document)
        # assert len(split_document.passages) == 1, \
        #     ('Each document must be given as a single passage.')
        # collection.add_document(split_document)
        # self.extractor.extract(collection)
        # self.classifier.classify(collection)
        # labels = self.aggregator.aggregate_dict(collection)

        # Extract observation mentions in place.
        self.extractor.extract(self.loader.collection)
        # Classify mentions in place.
        self.classifier.classify(self.loader.collection)
        # Aggregate mentions to obtain one set of labels for each report.
        labels = self.aggregator.aggregate_dict(self.loader.collection)

        assert len(labels) == 1
        return labels[0]