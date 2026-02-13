"""
End-to-End NLP System Demo

This script demonstrates the complete pipeline for both question answering
and text summarization tasks, including:
- Data loading and preprocessing
- Model training
- Inference and predictions
- Evaluation with metrics

Usage:
    python demo_nlp_system.py --task qa
    python demo_nlp_system.py --task summarization
    python demo_nlp_system.py --task all
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from nlp_system import (
    QuestionAnsweringSystem,
    TextSummarizationSystem,
    EvaluationMetrics,
    create_train_val_split,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# QUESTION ANSWERING DEMO
# ============================================================================

def demo_question_answering(
    num_train_samples: int = 100,
    num_val_samples: int = 20,
    output_dir: str = "./results/qa",
):
    """
    Complete QA pipeline demonstration.
    
    Args:
        num_train_samples: Number of training samples to use
        num_val_samples: Number of validation samples to use
        output_dir: Directory to save results
    """
    logger.info("=" * 70)
    logger.info("QUESTION ANSWERING SYSTEM DEMO")
    logger.info("=" * 70)

    # Initialize system
    qa_system = QuestionAnsweringSystem(
        model_name="distilbert-base-uncased",
        max_seq_length=384,
    )

    # Load dataset
    logger.info("\n1. Loading SQuAD v1.1 Dataset...")
    train_examples = qa_system.load_squad_dataset(
        split="train",
        max_samples=num_train_samples,
    )
    val_examples = qa_system.load_squad_dataset(
        split="validation",
        max_samples=num_val_samples,
    )

    # Display sample
    logger.info("\nSample QA Example:")
    sample = train_examples[0]
    logger.info(f"  Question: {sample.question}")
    logger.info(f"  Context: {sample.context[:200]}...")
    logger.info(f"  Answer: {sample.answer_text}")

    # Training (optional - comment out for faster demo)
    logger.info("\n2. Fine-tuning Model...")
    logger.info("Note: Skipping training for this demo. In production, use:")
    logger.info("  qa_system.train(train_examples, val_examples)")

    # Instead, we'll demonstrate inference with pre-trained model
    logger.info("\n3. Demonstrating Inference with Pre-trained Model...")

    # Test examples
    test_contexts = [
        "Python is a high-level programming language. It emphasizes code readability "
        "and allows developers to express concepts in fewer lines of code than would be "
        "possible in languages such as C++ or Java.",

        "The Great Wall of China is a series of fortifications made of stone, brick, "
        "tamped earth, and wood, built along the historical northern borders of China. "
        "The wall stretches more than 13,000 miles.",
    ]

    test_questions = [
        "What does Python emphasize?",
        "How long is the Great Wall of China?",
    ]

    results = []

    for context, question in zip(test_contexts, test_questions):
        logger.info(f"\nQuestion: {question}")
        logger.info(f"Context: {context[:100]}...")

        predictions = qa_system.predict(question, context, top_k=3)

        for i, pred in enumerate(predictions, 1):
            logger.info(
                f"  Answer {i}: '{pred['text']}' (confidence: {pred['score']:.4f})"
            )

        results.append({
            "question": question,
            "context": context,
            "predictions": [
                {
                    "text": p["text"],
                    "score": float(p["score"]),
                }
                for p in predictions
            ],
        })

    # Evaluation metrics (demonstration)
    logger.info("\n4. Evaluation Metrics (Demonstration)...")

    sample_predictions = ["Python", "13,000 miles"]
    sample_references = ["code readability", "more than 13,000 miles"]

    metrics = EvaluationMetrics.compute_qa_metrics(
        sample_predictions,
        sample_references,
    )

    logger.info(f"  Exact Match: {metrics['exact_match']:.2f}%")
    logger.info(f"  F1 Score: {metrics['f1_score']:.2f}%")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_results(
        {
            "task": "question_answering",
            "model": "distilbert-base-uncased",
            "predictions": results,
            "metrics": metrics,
        },
        f"{output_dir}/qa_results.json",
    )

    logger.info(f"\nResults saved to {output_dir}/qa_results.json")

    return results


# ============================================================================
# SUMMARIZATION DEMO
# ============================================================================

def demo_summarization(
    num_train_samples: int = 50,
    num_val_samples: int = 10,
    output_dir: str = "./results/summarization",
):
    """
    Complete summarization pipeline demonstration.
    
    Args:
        num_train_samples: Number of training samples to use
        num_val_samples: Number of validation samples to use
        output_dir: Directory to save results
    """
    logger.info("=" * 70)
    logger.info("TEXT SUMMARIZATION SYSTEM DEMO")
    logger.info("=" * 70)

    # Initialize system
    sum_system = TextSummarizationSystem(
        model_name="facebook/bart-base",
        max_input_length=1024,
        max_target_length=128,
    )

    # Load dataset
    logger.info("\n1. Loading CNN-DailyMail Dataset...")
    train_examples = sum_system.load_cnn_dailymail_dataset(
        split="train",
        max_samples=num_train_samples,
    )
    val_examples = sum_system.load_cnn_dailymail_dataset(
        split="validation",
        max_samples=num_val_samples,
    )

    # Display sample
    logger.info("\nSample Summarization Example:")
    sample = train_examples[0]
    logger.info(f"Article (first 300 chars):\n  {sample.source[:300]}...")
    logger.info(f"\nReference Summary:\n  {sample.target}")

    # Training (optional - comment out for faster demo)
    logger.info("\n2. Fine-tuning Model...")
    logger.info("Note: Skipping training for this demo. In production, use:")
    logger.info("  sum_system.train(train_examples, val_examples)")

    # Demonstration with custom texts
    logger.info("\n3. Demonstrating Inference with Custom Texts...")

    sample_articles = [
        """
        Artificial Intelligence (AI) is transforming industries around the world.
        Machine learning algorithms are now used to power recommendation systems,
        autonomous vehicles, and medical diagnostics. Companies like Google, Microsoft,
        and Amazon are investing billions in AI research. The technology has the potential
        to solve complex problems and improve human productivity. However, experts also
        raise concerns about job displacement and ethical implications. Governments are
        beginning to develop regulations to ensure AI is developed responsibly.
        """,

        """
        Climate change is one of the most pressing challenges of our time. Rising global
        temperatures are causing glaciers to melt, sea levels to rise, and extreme weather
        events to become more frequent. Scientists agree that human activities, particularly
        the burning of fossil fuels, are the primary cause of global warming. The transition
        to renewable energy sources like solar and wind power is essential. International
        agreements like the Paris Climate Accord represent global commitment to addressing
        this crisis. However, many countries are not on track to meet their targets.
        Individual actions and policy changes are both necessary to combat climate change.
        """,
    ]

    results = []

    for i, article in enumerate(sample_articles, 1):
        logger.info(f"\nArticle {i}:")
        logger.info(f"  Original length: {len(article.split())} words")

        summary = sum_system.summarize(article, num_beams=4)

        logger.info(f"  Summary length: {len(summary.split())} words")
        logger.info(f"  Summary:\n  {summary}")

        results.append({
            "article": article.strip(),
            "generated_summary": summary,
        })

    # Evaluation metrics (demonstration)
    logger.info("\n4. Evaluation Metrics (Demonstration)...")

    sample_predictions = [
        "AI is transforming industries with applications in recommendations, autonomous vehicles, and medical diagnostics.",
        "Climate change requires transition to renewable energy and international cooperation.",
    ]

    sample_references = [
        "Artificial Intelligence is transforming industries with applications in multiple domains.",
        "Climate change is a pressing challenge requiring renewable energy and policy changes.",
    ]

    metrics = EvaluationMetrics.compute_rouge_scores(
        sample_predictions,
        sample_references,
    )

    logger.info(f"  ROUGE-1: {metrics['rouge1']:.2f}")
    logger.info(f"  ROUGE-2: {metrics['rouge2']:.2f}")
    logger.info(f"  ROUGE-L: {metrics['rougeL']:.2f}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_results(
        {
            "task": "summarization",
            "model": "facebook/bart-base",
            "samples": results,
            "metrics": metrics,
        },
        f"{output_dir}/summarization_results.json",
    )

    logger.info(f"\nResults saved to {output_dir}/summarization_results.json")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(
        description="NLP System Demo - QA and Summarization",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["qa", "summarization", "all"],
        default="all",
        help="Which task to run",
    )
    parser.add_argument(
        "--qa-samples",
        type=int,
        default=100,
        help="Number of QA training samples",
    )
    parser.add_argument(
        "--sum-samples",
        type=int,
        default=50,
        help="Number of summarization training samples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    try:
        if args.task in ["qa", "all"]:
            qa_results = demo_question_answering(
                num_train_samples=args.qa_samples,
                num_val_samples=args.qa_samples // 5,
                output_dir=f"{args.output_dir}/qa",
            )

        if args.task in ["summarization", "all"]:
            sum_results = demo_summarization(
                num_train_samples=args.sum_samples,
                num_val_samples=args.sum_samples // 5,
                output_dir=f"{args.output_dir}/summarization",
            )

        logger.info("\n" + "=" * 70)
        logger.info("DEMO COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
