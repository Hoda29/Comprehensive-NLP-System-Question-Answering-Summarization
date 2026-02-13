"""
Configuration and Hyperparameter Management

Provides centralized configuration for both QA and summarization systems,
including model selection, training parameters, and evaluation settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# QUESTION ANSWERING CONFIGURATION
# ============================================================================

@dataclass
class QAConfig:
    """Configuration for Question Answering system."""

    # Model selection
    model_name: str = "distilbert-base-uncased"
    model_type: str = "bert"  # 'bert' or 'distilbert'

    # Tokenization
    max_seq_length: int = 384
    doc_stride: int = 128
    max_query_length: int = 64

    # Training
    output_dir: str = "./models/qa"
    num_epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1

    # Inference
    top_k: int = 1
    max_answer_length: int = 15
    n_best_size: int = 5

    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
        }

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"QA config saved to {path}")

    @classmethod
    def from_json(cls, path: str) -> "QAConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class SummarizationConfig:
    """Configuration for Text Summarization system."""

    # Model selection
    model_name: str = "facebook/bart-base"
    model_type: str = "bart"  # 'bart', 't5', or 'pegasus'

    # Tokenization
    max_input_length: int = 1024
    max_target_length: int = 128
    min_target_length: int = 50

    # Training
    output_dir: str = "./models/summarization"
    num_epochs: int = 3
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 2

    # Inference
    num_beams: int = 4
    early_stopping: bool = True
    length_penalty: float = 2.0

    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
        }

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Summarization config saved to {path}")

    @classmethod
    def from_json(cls, path: str) -> "SummarizationConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class QAPresets:
    """Preset configurations for QA systems."""

    LIGHT = QAConfig(
        model_name="distilbert-base-uncased",
        model_type="distilbert",
        max_seq_length=256,
        batch_size=32,
        num_epochs=2,
    )

    BALANCED = QAConfig(
        model_name="bert-base-uncased",
        model_type="bert",
        max_seq_length=384,
        batch_size=16,
        num_epochs=3,
    )

    POWERFUL = QAConfig(
        model_name="bert-large-uncased",
        model_type="bert",
        max_seq_length=512,
        batch_size=8,
        num_epochs=3,
        learning_rate=1e-5,
    )


class SummarizationPresets:
    """Preset configurations for summarization systems."""

    LIGHT = SummarizationConfig(
        model_name="facebook/bart-base",
        model_type="bart",
        max_input_length=512,
        max_target_length=64,
        batch_size=16,
        num_epochs=2,
    )

    BALANCED = SummarizationConfig(
        model_name="facebook/bart-base",
        model_type="bart",
        max_input_length=1024,
        max_target_length=128,
        batch_size=8,
        num_epochs=3,
    )

    POWERFUL = SummarizationConfig(
        model_name="facebook/bart-large",
        model_type="bart",
        max_input_length=1024,
        max_target_length=150,
        batch_size=4,
        num_epochs=3,
    )

    T5_BALANCED = SummarizationConfig(
        model_name="t5-base",
        model_type="t5",
        max_input_length=512,
        max_target_length=128,
        batch_size=8,
        num_epochs=3,
    )

    PEGASUS_BALANCED = SummarizationConfig(
        model_name="google/pegasus-cnn_dailymail",
        model_type="pegasus",
        max_input_length=1024,
        max_target_length=128,
        batch_size=8,
        num_epochs=2,  # Pre-trained on CNN-DailyMail, needs fewer epochs
    )


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""

    # QA Evaluation
    qa_metrics: list = field(
        default_factory=lambda: ["exact_match", "f1_score"]
    )
    qa_threshold: float = 0.0  # Minimum confidence threshold

    # Summarization Evaluation
    sum_metrics: list = field(
        default_factory=lambda: ["rouge1", "rouge2", "rougeL"]
    )
    use_stemmer: bool = True

    # General
    num_samples_for_eval: Optional[int] = None  # None = use all
    bootstrap_samples: int = 0  # 0 = no bootstrap, else run N iterations


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for complete NLP pipeline."""

    qa_config: QAConfig = field(default_factory=QAConfig)
    sum_config: SummarizationConfig = field(default_factory=SummarizationConfig)
    eval_config: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Data
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def validate(self) -> bool:
        """Validate configuration consistency."""
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        if not abs(sum(ratios) - 1.0) < 1e-6:
            logger.warning(
                f"Data split ratios don't sum to 1.0: {ratios}"
            )
            return False
        return True

    def to_json(self, path: str) -> None:
        """Save pipeline config to JSON."""
        config_dict = {
            "qa": self.qa_config.to_dict(),
            "summarization": self.sum_config.to_dict(),
            "evaluation": self.eval_config.__dict__,
            "data": {
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
            },
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Pipeline config saved to {path}")


if __name__ == "__main__":
    # Example usage
    qa_config = QAPresets.BALANCED
    print(f"QA Config: {qa_config}")

    sum_config = SummarizationPresets.BALANCED
    print(f"Summarization Config: {sum_config}")
