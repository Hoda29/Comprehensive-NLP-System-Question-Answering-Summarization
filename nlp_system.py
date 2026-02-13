"""
Comprehensive NLP System: Question Answering & Text Summarization

This module implements production-ready transformer-based pipelines for:
1. Extractive Question Answering (using BERT/DistilBERT on SQuAD v1.1)
2. Abstractive Text Summarization (using T5/BART/Pegasus on CNN-DailyMail)

Architecture:
- Uses Hugging Face Transformers library for pre-trained models
- Implements proper data preprocessing with tokenization and attention masks
- Includes comprehensive evaluation metrics (EM, F1, ROUGE)
- Provides train/validation/test splits for both datasets
- Contains clear separation of concerns with modular functions

Author: NLP Systems Team
Date: 2026
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader

# Evaluation metrics
from sklearn.metrics import f1_score, accuracy_score
import rouge_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QAExample:
    """Data structure for a question-answering example"""
    question_id: str
    question: str
    context: str
    answer_text: str
    answer_start: int
    title: str = ""


@dataclass
class SummaryExample:
    """Data structure for a summarization example"""
    id: str
    source: str
    target: str


# ============================================================================
# QUESTION ANSWERING SYSTEM
# ============================================================================

class QuestionAnsweringSystem:
    """
    Extractive Question Answering System using transformer models.
    
    Processes the SQuAD v1.1 dataset and fine-tunes a pre-trained model
    (BERT/DistilBERT) to extract answer spans from context passages.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_seq_length: int = 384,
        doc_stride: int = 128,
    ):
        """
        Initialize the QA system.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
            max_seq_length: Maximum sequence length for tokenization
            doc_stride: Stride for splitting long contexts
        """
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name
        ).to(device)

        logger.info(f"Initialized QA system with {model_name}")

    def load_squad_dataset(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[QAExample]:
        """
        Load SQuAD v1.1 dataset from Hugging Face.
        
        Args:
            split: 'train' or 'validation'
            max_samples: Limit dataset size for testing (None = use all)
            
        Returns:
            List of QAExample objects
        """
        logger.info(f"Loading SQuAD {split} split...")

        dataset = load_dataset("squad", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        qa_examples = []

        for example in dataset:
            # Handle multiple answers by using the first one
            answer = example["answers"]
            answer_text = answer["text"][0] if answer["text"] else ""
            answer_start = answer["answer_start"][0] if answer["answer_start"] else 0

            qa_examples.append(
                QAExample(
                    question_id=example["id"],
                    question=example["question"],
                    context=example["context"],
                    answer_text=answer_text,
                    answer_start=answer_start,
                    title=example["title"],
                )
            )

        logger.info(f"Loaded {len(qa_examples)} QA examples from {split} split")
        return qa_examples

    def prepare_squad_features(
        self,
        examples: List[QAExample],
        split: str = "train",
    ) -> Dataset:
        """
        Preprocess QA examples into tokenized features.
        
        Handles:
        - Tokenization of questions and contexts
        - Finding answer start/end positions in tokenized sequences
        - Creating attention masks and token type IDs
        - Handling overflow tokens for long contexts
        
        Args:
            examples: List of QAExample objects
            split: 'train', 'validation', or 'test'
            
        Returns:
            HuggingFace Dataset with tokenized features
        """
        logger.info(f"Preparing features for {len(examples)} examples...")

        # Convert examples to dict format
        examples_dict = {
            "id": [ex.question_id for ex in examples],
            "title": [ex.title for ex in examples],
            "context": [ex.context for ex in examples],
            "question": [ex.question for ex in examples],
            "answers": [
                {"text": [ex.answer_text], "answer_start": [ex.answer_start]}
                for ex in examples
            ],
        }

        def preprocess_function(examples):
            """Tokenize questions and contexts, identify answer positions."""

            # Tokenize questions and contexts
            tokenized = self.tokenizer(
                examples["question"],
                examples["context"],
                truncation="only_second",  # Truncate context only
                max_length=self.max_seq_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            sample_mapping = tokenized.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized.pop("offset_mapping")

            # Initialize answer positions
            tokenized["start_positions"] = []
            tokenized["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized["input_ids"][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)

                # Get the sample this example maps to
                sample_idx = sample_mapping[i]
                answers = examples["answers"][sample_idx]

                # If no answer (for validation), set to cls_index
                if len(answers["answer_start"]) == 0:
                    tokenized["start_positions"].append(cls_index)
                    tokenized["end_positions"].append(cls_index)
                else:
                    # Find token indices corresponding to answer span
                    answer_start_char = answers["answer_start"][0]
                    answer_end_char = answer_start_char + len(answers["text"][0])

                    # Find start and end token positions
                    start_pos = cls_index
                    end_pos = cls_index

                    for idx, (offset_start, offset_end) in enumerate(offsets):
                        if offset_start <= answer_start_char < offset_end:
                            start_pos = idx
                        if offset_start < answer_end_char <= offset_end:
                            end_pos = idx

                    tokenized["start_positions"].append(start_pos)
                    tokenized["end_positions"].append(end_pos)

            return tokenized

        # Apply preprocessing
        dataset = Dataset.from_dict(examples_dict)
        features = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Preparing {split} features",
        )

        logger.info(f"Prepared {len(features)} features")
        return features

    def train(
        self,
        train_examples: List[QAExample],
        val_examples: List[QAExample],
        output_dir: str = "./qa_model",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> Dict[str, float]:
        """
        Fine-tune the QA model on SQuAD dataset.
        
        Args:
            train_examples: Training QA examples
            val_examples: Validation QA examples
            output_dir: Directory to save model checkpoints
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting QA model training...")

        # Prepare features
        train_features = self.prepare_squad_features(train_examples, "train")
        val_features = self.prepare_squad_features(val_examples, "validation")

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=100,
            push_to_hub=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_features,
            eval_dataset=val_features,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )

        # Train
        train_result = trainer.train()

        logger.info(f"Training completed. Loss: {train_result.training_loss:.4f}")

        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return {
            "training_loss": train_result.training_loss,
            "model_saved_to": output_dir,
        }

    def predict(
        self,
        question: str,
        context: str,
        top_k: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Extract answer span(s) from context for a given question.
        
        Process:
        1. Tokenize question and context
        2. Forward pass through model to get start/end logits
        3. Extract top-k answer spans based on probability scores
        
        Args:
            question: The question to answer
            context: The context passage to extract answer from
            top_k: Number of top answer spans to return
            
        Returns:
            List of dicts with 'text', 'score', 'start', 'end'
        """
        self.model.eval()

        # Tokenize
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_seq_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get start and end logits
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Get top-k start and end positions
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)

        start_indices = torch.argsort(start_probs, descending=True)[:top_k]
        end_indices = torch.argsort(end_probs, descending=True)[:top_k]

        # Generate candidate answers
        answers = []
        input_ids = inputs["input_ids"][0]
        offset_mapping = self.tokenizer(
            question, context, return_offsets_mapping=True
        )["offset_mapping"]

        for start_idx in start_indices:
            for end_idx in end_indices:
                if start_idx < end_idx and end_idx - start_idx <= 15:
                    # Extract answer text
                    answer_tokens = input_ids[start_idx : end_idx + 1]
                    answer_text = self.tokenizer.decode(
                        answer_tokens, skip_special_tokens=True
                    )

                    # Calculate confidence score
                    score = (
                        start_probs[start_idx].item()
                        + end_probs[end_idx].item()
                    ) / 2

                    answers.append(
                        {
                            "text": answer_text,
                            "score": score,
                            "start": start_idx.item(),
                            "end": end_idx.item(),
                        }
                    )

        # Sort by score and return top-k
        answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:top_k]

        return answers


# ============================================================================
# TEXT SUMMARIZATION SYSTEM
# ============================================================================

class TextSummarizationSystem:
    """
    Abstractive Text Summarization System using transformer models.
    
    Handles long documents by truncating to model limits and
    fine-tunes pre-trained models (T5, BART, or Pegasus) on CNN-DailyMail.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_input_length: int = 1024,
        max_target_length: int = 128,
    ):
        """
        Initialize the summarization system.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
            max_input_length: Max tokens for input documents
            max_target_length: Max tokens for summary output
        """
        self.model_name = model_name
        self.device = device
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        ).to(device)

        logger.info(f"Initialized Summarization system with {model_name}")

    def load_cnn_dailymail_dataset(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[SummaryExample]:
        """
        Load CNN-DailyMail dataset from Hugging Face.
        
        Args:
            split: 'train', 'validation', or 'test'
            max_samples: Limit dataset size for testing
            
        Returns:
            List of SummaryExample objects
        """
        logger.info(f"Loading CNN-DailyMail {split} split...")

        dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        summary_examples = []

        for i, example in enumerate(dataset):
            summary_examples.append(
                SummaryExample(
                    id=str(i),
                    source=example["article"],
                    target=example["highlights"],
                )
            )

        logger.info(f"Loaded {len(summary_examples)} examples from {split} split")
        return summary_examples

    def prepare_summarization_features(
        self,
        examples: List[SummaryExample],
        split: str = "train",
    ) -> Dataset:
        """
        Preprocess summarization examples with proper truncation.
        
        Handles:
        - Truncating long documents to max_input_length
        - Tokenizing source articles and target summaries
        - Creating attention masks
        - Padding sequences
        
        Args:
            examples: List of SummaryExample objects
            split: 'train', 'validation', or 'test'
            
        Returns:
            HuggingFace Dataset with tokenized features
        """
        logger.info(f"Preparing features for {len(examples)} examples...")

        # Convert to dict
        examples_dict = {
            "id": [ex.id for ex in examples],
            "article": [ex.source for ex in examples],
            "highlights": [ex.target for ex in examples],
        }

        def preprocess_function(examples):
            """Tokenize articles and highlights with truncation."""

            # Tokenize source articles
            inputs = self.tokenizer(
                examples["article"],
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="np",
            )

            # Tokenize target summaries
            labels = self.tokenizer(
                examples["highlights"],
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="np",
            )

            inputs["labels"] = labels["input_ids"]

            return inputs

        dataset = Dataset.from_dict(examples_dict)
        features = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Preparing {split} features",
        )

        logger.info(f"Prepared {len(features)} features")
        return features

    def train(
        self,
        train_examples: List[SummaryExample],
        val_examples: List[SummaryExample],
        output_dir: str = "./summarization_model",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ) -> Dict[str, float]:
        """
        Fine-tune the summarization model on CNN-DailyMail.
        
        Args:
            train_examples: Training summary examples
            val_examples: Validation summary examples
            output_dir: Directory to save model checkpoints
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting summarization model training...")

        # Prepare features
        train_features = self.prepare_summarization_features(
            train_examples, "train"
        )
        val_features = self.prepare_summarization_features(
            val_examples, "validation"
        )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=100,
            push_to_hub=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_features,
            eval_dataset=val_features,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer, self.model
            ),
        )

        # Train
        train_result = trainer.train()

        logger.info(f"Training completed. Loss: {train_result.training_loss:.4f}")

        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return {
            "training_loss": train_result.training_loss,
            "model_saved_to": output_dir,
        }

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: int = 4,
    ) -> str:
        """
        Generate abstractive summary for input text.
        
        Process:
        1. Truncate text to max_input_length if needed
        2. Tokenize truncated text
        3. Generate summary using beam search
        4. Decode to text
        
        Args:
            text: Input document to summarize
            max_length: Max length of summary (None = use max_target_length)
            min_length: Min length of summary
            num_beams: Beam width for beam search
            
        Returns:
            Generated summary text
        """
        self.model.eval()

        max_length = max_length or self.max_target_length
        min_length = min_length or max_length // 2

        # Tokenize with truncation
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                early_stopping=True,
                attention_mask=inputs.get("attention_mask"),
            )

        # Decode
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class EvaluationMetrics:
    """
    Compute evaluation metrics for QA and summarization tasks.
    """

    @staticmethod
    def compute_qa_metrics(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute Exact Match (EM) and F1 Score for QA.
        
        Args:
            predictions: Model predictions
            references: Ground truth answers
            
        Returns:
            Dictionary with 'exact_match' and 'f1_score'
        """
        em_count = 0
        f1_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())

            # Exact match
            if pred.lower() == ref.lower():
                em_count += 1

            # F1 score
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1 = 1.0 if pred == ref else 0.0
            else:
                common = len(pred_tokens & ref_tokens)
                precision = common / len(pred_tokens) if pred_tokens else 0
                recall = common / len(ref_tokens) if ref_tokens else 0
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

            f1_scores.append(f1)

        return {
            "exact_match": (em_count / len(predictions)) * 100,
            "f1_score": (sum(f1_scores) / len(f1_scores)) * 100,
        }

    @staticmethod
    def compute_rouge_scores(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores for summarization.
        
        ROUGE-1: Unigram overlap
        ROUGE-2: Bigram overlap
        ROUGE-L: Longest common subsequence
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        scorer = rouge_score.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": np.mean(rouge1_scores) * 100,
            "rouge2": np.mean(rouge2_scores) * 100,
            "rougeL": np.mean(rougeL_scores) * 100,
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_train_val_split(
    examples: List[Any],
    train_ratio: float = 0.8,
) -> Tuple[List[Any], List[Any]]:
    """
    Split examples into train and validation sets.
    
    Args:
        examples: List of examples to split
        train_ratio: Fraction for training (rest for validation)
        
    Returns:
        Tuple of (train_examples, val_examples)
    """
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]


def save_results(
    results: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary of results
        output_path: Path to save JSON
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    logger.info("NLP System initialized successfully")
