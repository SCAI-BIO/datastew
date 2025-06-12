import os

from datastew.embedding import Vectorizer
from datastew.repository import Concept, Mapping, Terminology


class BaseRepositoryTestSetup:
    @classmethod
    def setUpClass(cls):
        cls.TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        cls.vectorizer1 = Vectorizer("sentence-transformers/all-mpnet-base-v2")
        cls.vectorizer2 = Vectorizer("FremyCompany/BioLORD-2023")
        cls.model_name1 = cls.vectorizer1.model_name
        cls.model_name2 = cls.vectorizer2.model_name

        cls.terminology1 = Terminology("snomed CT", "SNOMED")
        cls.terminology2 = Terminology("NCI Thesaurus OBO Edition", "NCIT")

        cls.concepts_mappings = [
            cls._create_mapping(
                cls.terminology1, "Diabetes mellitus (disorder)", "Concept ID: 11893007", cls.vectorizer1
            ),
            cls._create_mapping(cls.terminology1, "Hypertension (disorder)", "Concept ID: 73211009", cls.vectorizer2),
            cls._create_mapping(cls.terminology1, "Asthma", "Concept ID: 195967001", cls.vectorizer1),
            cls._create_mapping(cls.terminology1, "Heart attack", "Concept ID: 22298006", cls.vectorizer2),
            cls._create_mapping(
                cls.terminology1, "Complex General Surgical Oncology", "Concept ID: 45756764", cls.vectorizer1
            ),
            cls._create_mapping(cls.terminology1, "Cancer", "Concept ID: 45877275", cls.vectorizer1),
            cls._create_mapping(cls.terminology2, "Common cold", "Concept ID: 13260007", cls.vectorizer1),
            cls._create_mapping(cls.terminology2, "Stroke", "Concept ID: 422504002", cls.vectorizer2),
            cls._create_mapping(cls.terminology2, "Migraine", "Concept ID: 386098009", cls.vectorizer1),
            cls._create_mapping(cls.terminology2, "Influenza", "Concept ID: 57386000", cls.vectorizer2),
            cls._create_mapping(cls.terminology2, "Osteoarthritis", "Concept ID: 399206004", cls.vectorizer1),
        ]

        cls.test_text = "The flu"

    @staticmethod
    def _create_mapping(terminology: Terminology, text: str, concept_id: str, vectorizer: Vectorizer):
        concept = Concept(terminology, text, concept_id)
        mapping = Mapping(
            concept,
            text,
            vectorizer.get_embedding(text),
            vectorizer.model_name,
        )
        return concept, mapping
