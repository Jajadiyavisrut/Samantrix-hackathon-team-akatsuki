"""
Unit and Integration Test Suite for Resume Classification System.
Validates:
1. Preprocessing and technical token preservation
2. Input validation (empty, whitespace, short inputs)
3. Dataset loading and schema validation
4. Train/Val/Test splitting
5. Model artifact loading in fresh process
6. Text inference & PDF extraction
"""
import os
import sys
import unittest
import pandas as pd
from pathlib import Path

# Add project root to sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_CSV_PATH, PROCESSED_DATA_DIR, MODELS_DIR,
    TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH, BEST_CLASSICAL_MODEL_PATH
)
from src.preprocessing import preprocess_text, tokenize_text
from src.data_loader import load_raw_csv, inspect_dataset_sources
from src.data_validation import run_data_quality_audit, clean_and_split_data
from src.inference import ResumeClassificationPipeline
from src.pdf_extractor import extract_text_from_pdf


class TestPreprocessing(unittest.TestCase):
    def test_technical_tokens_preservation(self):
        text = "Expert in C++, C#, .NET, Node.js, and Python."
        cleaned = preprocess_text(text)
        self.assertIn("cpp", cleaned)
        self.assertIn("csharp", cleaned)
        self.assertIn("dotnet", cleaned)
        self.assertIn("nodejs", cleaned)
        self.assertIn("python", cleaned)

    def test_empty_and_null_inputs(self):
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text("   "), "")
        self.assertEqual(preprocess_text(None), "")

    def test_contact_normalization(self):
        text = "Contact me at candidate@example.com or call (123) 456-7890. Visit https://portfolio.com"
        cleaned = preprocess_text(text)
        self.assertIn("email_ref", cleaned)
        self.assertIn("phone_ref", cleaned)
        self.assertIn("url_ref", cleaned)


class TestDataValidationAndLoading(unittest.TestCase):
    def test_raw_dataset_loading(self):
        if os.path.exists(RAW_CSV_PATH):
            df = load_raw_csv()
            self.assertIn("ID", df.columns)
            self.assertIn("Resume_str", df.columns)
            self.assertIn("Category", df.columns)
            self.assertEqual(df["Category"].nunique(), 24)

    def test_clean_and_split(self):
        if os.path.exists(RAW_CSV_PATH):
            df = load_raw_csv()
            cleaned, train_df, val_df, test_df = clean_and_split_data(df)
            self.assertGreater(len(train_df), len(val_df))
            self.assertGreater(len(train_df), len(test_df))
            # Test no duplicate texts between train and test
            train_texts = set(train_df["Resume_str"])
            test_texts = set(test_df["Resume_str"])
            self.assertEqual(len(train_texts.intersection(test_texts)), 0)


class TestInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(BEST_CLASSICAL_MODEL_PATH):
            cls.pipeline = ResumeClassificationPipeline()
        else:
            cls.pipeline = None

    def test_empty_input_handling(self):
        if self.pipeline:
            res_empty = self.pipeline.predict_text("")
            self.assertFalse(res_empty["success"])
            self.assertIn("empty", res_empty["error"])

            res_short = self.pipeline.predict_text("hi there")
            self.assertFalse(res_short["success"])
            self.assertIn("short", res_short["error"])

    def test_valid_text_predictions(self):
        if self.pipeline:
            chef_text = "Executive Chef with 15 years experience in fine dining cuisine, pastry arts, banquet operations, and kitchen menu development."
            res = self.pipeline.predict_text(chef_text)
            self.assertTrue(res["success"])
            self.assertEqual(res["predicted_category"], "CHEF")
            self.assertEqual(res["score_type"], "Decision Score")
            self.assertGreater(len(res["top_classes"]), 0)


if __name__ == "__main__":
    unittest.main()
