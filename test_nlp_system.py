"""
Test Suite and Comprehensive Examples

This module provides unit tests and practical examples demonstrating
all features of the NLP system.

Run tests with: python -m pytest test_nlp_system.py -v
Run examples with: python test_nlp_system.py --examples
"""

import unittest
from typing import List, Dict
import logging

import torch
from nlp_system import (
    QuestionAnsweringSystem,
    TextSummarizationSystem,
    EvaluationMetrics,
    QAExample,
    SummaryExample,
    create_train_val_split,
)
from config import QAPresets, SummarizationPresets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestQuestionAnsweringSystem(unittest.TestCase):
    """Test suite for Question Answering System."""

    @classmethod
    def setUpClass(cls):
        """Initialize QA system once for all tests."""
        cls.qa_system = QuestionAnsweringSystem(
            model_name="distilbert-base-uncased",
            device="cpu",  # Use CPU for testing
        )

    def test_initialization(self):
        """Test that QA system initializes correctly."""
        self.assertIsNotNone(self.qa_system.model)
        self.assertIsNotNone(self.qa_system.tokenizer)
        self.assertEqual(
            self.qa_system.max_seq_length,
            384,
        )

    def test_simple_prediction(self):
        """Test basic QA prediction."""
        context = "Paris is the capital of France."
        question = "What is the capital of France?"

        predictions = self.qa_system.predict(question, context, top_k=1)

        self.assertGreater(len(predictions), 0)
        self.assertIn('text', predictions[0])
        self.assertIn('score', predictions[0])
        self.assertGreater(predictions[0]['score'], 0.0)
        self.assertLess(predictions[0]['score'], 1.0)

    def test_multiple_predictions(self):
        """Test getting multiple answer candidates."""
        context = "The Eiffel Tower is in Paris, France."
        question = "Where is the Eiffel Tower?"

        predictions = self.qa_system.predict(question, context, top_k=5)

        self.assertEqual(len(predictions), 5)
        # Scores should be in descending order
        scores = [p['score'] for p in predictions]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_long_context_handling(self):
        """Test handling of long contexts with truncation."""
        long_context = (
            "This is a very long context. " * 100
        ) + "The answer is here."
        question = "Where is the answer?"

        predictions = self.qa_system.predict(question, long_context)

        self.assertGreater(len(predictions), 0)

    def test_qa_example_creation(self):
        """Test QAExample data structure."""
        example = QAExample(
            question_id="q1",
            question="What is AI?",
            context="AI stands for Artificial Intelligence.",
            answer_text="Artificial Intelligence",
            answer_start=16,
            title="AI Basics",
        )

        self.assertEqual(example.question_id, "q1")
        self.assertEqual(example.answer_text, "Artificial Intelligence")

    def test_config_presets(self):
        """Test configuration presets."""
        light_config = QAPresets.LIGHT
        balanced_config = QAPresets.BALANCED
        powerful_config = QAPresets.POWERFUL

        self.assertEqual(light_config.model_type, "distilbert")
        self.assertEqual(balanced_config.model_type, "bert")
        self.assertEqual(powerful_config.model_type, "bert")


class TestTextSummarizationSystem(unittest.TestCase):
    """Test suite for Text Summarization System."""

    @classmethod
    def setUpClass(cls):
        """Initialize summarization system once for all tests."""
        cls.sum_system = TextSummarizationSystem(
            model_name="facebook/bart-base",
            device="cpu",  # Use CPU for testing
        )

    def test_initialization(self):
        """Test that summarization system initializes correctly."""
        self.assertIsNotNone(self.sum_system.model)
        self.assertIsNotNone(self.sum_system.tokenizer)
        self.assertEqual(
            self.sum_system.max_input_length,
            1024,
        )

    def test_simple_summarization(self):
        """Test basic summarization."""
        article = (
            "Artificial Intelligence is transforming industries. "
            "Machine learning enables systems to learn from data. "
            "Deep learning uses neural networks for complex tasks. "
            "These technologies power modern applications."
        )

        summary = self.sum_system.summarize(
            article,
            max_length=50,
            num_beams=2,
        )

        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLess(len(summary.split()), 50)

    def test_variable_length_summaries(self):
        """Test controlling summary length."""
        article = "The quick brown fox jumps over the lazy dog. " * 20

        short_summary = self.sum_system.summarize(
            article,
            max_length=20,
            num_beams=2,
        )

        long_summary = self.sum_system.summarize(
            article,
            max_length=100,
            num_beams=2,
        )

        short_len = len(short_summary.split())
        long_len = len(long_summary.split())

        self.assertLess(short_len, long_len)

    def test_long_document_truncation(self):
        """Test handling of documents longer than max_input_length."""
        # Create document much longer than max_input_length
        long_article = "This is a test sentence. " * 500

        summary = self.sum_system.summarize(long_article)

        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

    def test_summary_example_creation(self):
        """Test SummaryExample data structure."""
        example = SummaryExample(
            id="s1",
            source="This is the source article.",
            target="Source article.",
        )

        self.assertEqual(example.id, "s1")
        self.assertEqual(example.source, "This is the source article.")


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for Evaluation Metrics."""

    def test_exact_match_perfect(self):
        """Test exact match with perfect predictions."""
        predictions = ["Paris", "France", "Eiffel Tower"]
        references = ["Paris", "France", "Eiffel Tower"]

        metrics = EvaluationMetrics.compute_qa_metrics(
            predictions,
            references,
        )

        self.assertAlmostEqual(metrics['exact_match'], 100.0, places=1)
        self.assertAlmostEqual(metrics['f1_score'], 100.0, places=1)

    def test_exact_match_none(self):
        """Test exact match with no correct predictions."""
        predictions = ["London", "Spain", "Big Ben"]
        references = ["Paris", "France", "Eiffel Tower"]

        metrics = EvaluationMetrics.compute_qa_metrics(
            predictions,
            references,
        )

        self.assertAlmostEqual(metrics['exact_match'], 0.0, places=1)

    def test_f1_partial_match(self):
        """Test F1 score with partial token overlap."""
        predictions = ["Great Wall"]
        references = ["The Great Wall of China"]

        metrics = EvaluationMetrics.compute_qa_metrics(
            predictions,
            references,
        )

        # F1 should be between 0 and 100
        self.assertGreater(metrics['f1_score'], 0)
        self.assertLess(metrics['f1_score'], 100)

    def test_rouge_scores(self):
        """Test ROUGE score computation."""
        predictions = [
            "Artificial Intelligence is transforming the world.",
            "Machine learning enables computers to learn.",
        ]
        references = [
            "AI is transforming industries and sectors.",
            "ML allows systems to learn from data.",
        ]

        scores = EvaluationMetrics.compute_rouge_scores(
            predictions,
            references,
        )

        # Check that all ROUGE scores are in valid range
        for key in ['rouge1', 'rouge2', 'rougeL']:
            self.assertIn(key, scores)
            self.assertGreaterEqual(scores[key], 0)
            self.assertLessEqual(scores[key], 100)

    def test_rouge_perfect_match(self):
        """Test ROUGE scores with identical texts."""
        predictions = ["The quick brown fox"]
        references = ["The quick brown fox"]

        scores = EvaluationMetrics.compute_rouge_scores(
            predictions,
            references,
        )

        # Should have very high scores
        for key in ['rouge1', 'rouge2', 'rougeL']:
            self.assertGreater(scores[key], 80)


class TestDataManagement(unittest.TestCase):
    """Test suite for data handling and splitting."""

    def test_train_val_split(self):
        """Test train/validation split."""
        data = list(range(100))

        train, val = create_train_val_split(data, train_ratio=0.8)

        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 20)
        self.assertEqual(len(train) + len(val), 100)

    def test_split_different_ratios(self):
        """Test split with different ratios."""
        data = list(range(1000))

        for ratio in [0.6, 0.7, 0.8, 0.9]:
            train, val = create_train_val_split(data, train_ratio=ratio)
            expected_train = int(1000 * ratio)
            self.assertEqual(len(train), expected_train)

    def test_split_preserves_order(self):
        """Test that split preserves original order."""
        data = list(range(100))

        train, val = create_train_val_split(data, train_ratio=0.8)

        # First 80 should be in train
        self.assertEqual(train, list(range(80)))
        # Last 20 should be in val
        self.assertEqual(val, list(range(80, 100)))


# ============================================================================
# COMPREHENSIVE EXAMPLES
# ============================================================================

class ComprehensiveExamples:
    """Comprehensive examples demonstrating all system features."""

    @staticmethod
    def example_1_basic_qa():
        """Example 1: Basic Question Answering."""
        logger.info("\n" + "=" * 70)
        logger.info("EXAMPLE 1: Basic Question Answering")
        logger.info("=" * 70)

        qa_system = QuestionAnsweringSystem(
            model_name="distilbert-base-uncased",
        )

        context = """
        The Statue of Liberty is a colossal neoclassical sculpture located
        on Liberty Island in New York Harbor. It was a gift from the people
        of France to the United States. The statue was designed by Frédéric
        Auguste Bartholdi and dedicated on October 28, 1886.
        """

        questions = [
            "What is the Statue of Liberty?",
            "Where is the Statue of Liberty located?",
            "Who designed the Statue of Liberty?",
        ]

        for question in questions:
            logger.info(f"\nQuestion: {question}")
            answers = qa_system.predict(question, context, top_k=2)
            for i, ans in enumerate(answers, 1):
                logger.info(f"  Answer {i}: '{ans['text']}' "
                          f"(score: {ans['score']:.4f})")

    @staticmethod
    def example_2_summarization():
        """Example 2: Text Summarization."""
        logger.info("\n" + "=" * 70)
        logger.info("EXAMPLE 2: Text Summarization")
        logger.info("=" * 70)

        sum_system = TextSummarizationSystem(
            model_name="facebook/bart-base",
        )

        article = """
        The COVID-19 pandemic has dramatically accelerated digital transformation
        across all sectors of the global economy. Remote work adoption, online
        shopping, telemedicine, and digital payment systems have seen unprecedented
        growth. Companies that were slow to digitalize have faced significant
        challenges, while those with strong digital infrastructure have thrived.
        
        Governments worldwide have recognized the importance of digital infrastructure
        and are investing heavily in broadband expansion and digital literacy programs.
        The pandemic has also highlighted the digital divide, with rural and low-income
        communities facing barriers to access.
        
        Looking ahead, experts predict that hybrid work models will become the norm,
        online shopping will continue to grow, and digital technologies will play an
        increasingly central role in healthcare delivery. The key challenge will be
        ensuring that digital transformation benefits are distributed equitably across
        all segments of society.
        """

        logger.info("Original article length:", len(article.split()), "words")

        summary = sum_system.summarize(article)
        logger.info(f"\nGenerated Summary ({len(summary.split())} words):")
        logger.info(summary)

    @staticmethod
    def example_3_config_management():
        """Example 3: Configuration Management."""
        logger.info("\n" + "=" * 70)
        logger.info("EXAMPLE 3: Configuration Management")
        logger.info("=" * 70)

        # Show available presets
        presets = {
            "QA Light": QAPresets.LIGHT,
            "QA Balanced": QAPresets.BALANCED,
            "QA Powerful": QAPresets.POWERFUL,
            "Sum Light": SummarizationPresets.LIGHT,
            "Sum Balanced": SummarizationPresets.BALANCED,
            "Sum Pegasus": SummarizationPresets.PEGASUS_BALANCED,
        }

        for name, preset in presets.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Model: {preset.model_name}")
            logger.info(f"  Max sequence: {preset.max_seq_length if hasattr(preset, 'max_seq_length') else preset.max_input_length}")
            logger.info(f"  Batch size: {preset.batch_size}")

    @staticmethod
    def example_4_evaluation():
        """Example 4: Evaluation Metrics."""
        logger.info("\n" + "=" * 70)
        logger.info("EXAMPLE 4: Evaluation Metrics")
        logger.info("=" * 70)

        # QA Evaluation
        qa_predictions = ["Paris", "12,000 miles", "The Great Wall"]
        qa_references = ["Paris", "13,000 miles", "Great Wall of China"]

        qa_metrics = EvaluationMetrics.compute_qa_metrics(
            qa_predictions,
            qa_references,
        )

        logger.info("\nQuestion Answering Metrics:")
        logger.info(f"  Exact Match: {qa_metrics['exact_match']:.2f}%")
        logger.info(f"  F1 Score: {qa_metrics['f1_score']:.2f}%")

        # Summarization Evaluation
        sum_predictions = [
            "AI transforms industries.",
            "Climate change is urgent.",
        ]
        sum_references = [
            "Artificial Intelligence transforms sectors.",
            "Climate change requires action.",
        ]

        sum_metrics = EvaluationMetrics.compute_rouge_scores(
            sum_predictions,
            sum_references,
        )

        logger.info("\nSummarization Metrics (ROUGE):")
        logger.info(f"  ROUGE-1: {sum_metrics['rouge1']:.2f}")
        logger.info(f"  ROUGE-2: {sum_metrics['rouge2']:.2f}")
        logger.info(f"  ROUGE-L: {sum_metrics['rougeL']:.2f}")

    @staticmethod
    def example_5_batch_processing():
        """Example 5: Batch Processing."""
        logger.info("\n" + "=" * 70)
        logger.info("EXAMPLE 5: Batch Processing")
        logger.info("=" * 70)

        qa_system = QuestionAnsweringSystem(device="cpu")

        # Batch QA examples
        context = "Marie Curie won two Nobel Prizes. Albert Einstein won one."
        questions = [
            "How many Nobel Prizes did Marie Curie win?",
            "How many Nobel Prizes did Albert Einstein win?",
        ]

        logger.info("\nBatch Question Answering:")
        for q in questions:
            ans = qa_system.predict(q, context, top_k=1)[0]
            logger.info(f"  Q: {q}")
            logger.info(f"  A: {ans['text']}")

    @staticmethod
    def run_all_examples():
        """Run all comprehensive examples."""
        logger.info("\n" + "=" * 70)
        logger.info("COMPREHENSIVE EXAMPLES")
        logger.info("=" * 70)

        ComprehensiveExamples.example_1_basic_qa()
        ComprehensiveExamples.example_2_summarization()
        ComprehensiveExamples.example_3_config_management()
        ComprehensiveExamples.example_4_evaluation()
        ComprehensiveExamples.example_5_batch_processing()

        logger.info("\n" + "=" * 70)
        logger.info("ALL EXAMPLES COMPLETED")
        logger.info("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    if "--examples" in sys.argv:
        # Run comprehensive examples
        ComprehensiveExamples.run_all_examples()
    else:
        # Run unit tests
        unittest.main(argv=[''], verbosity=2, exit=False)
