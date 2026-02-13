"""
Advanced Training Utilities and Analysis

Provides tools for:
- Custom training callbacks and early stopping
- Model performance analysis
- Error analysis and debugging
- Training visualization and reporting
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from datetime import datetime

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM TRAINER CALLBACKS
# ============================================================================

class MetricsCallback(TrainerCallback):
    """
    Custom callback to log and save detailed training metrics.
    
    Tracks:
    - Training and validation loss
    - Learning rate changes
    - Training speed (samples/sec)
    - Gradient norm
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = defaultdict(list)
        self.start_time = None

    def on_init_end(self, args, state, control, **kwargs):
        """Called at the end of initialization."""
        self.start_time = datetime.now()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the model logs metrics."""
        if logs:
            for key, value in logs.items():
                self.history[key].append({
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "value": value,
                })

    def on_train_end(self, args, state, control, **kwargs):
        """Save metrics history at the end of training."""
        metrics_file = self.log_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(dict(self.history), f, indent=2, default=str)
        logger.info(f"Metrics saved to {metrics_file}")


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback to halt training if validation metric plateaus.
    
    Monitors a validation metric and stops if it doesn't improve
    for a specified number of evaluations.
    """

    def __init__(
        self,
        metric_name: str = "eval_loss",
        patience: int = 3,
        threshold: float = 0.0,
    ):
        """
        Args:
            metric_name: Name of metric to monitor (e.g., 'eval_loss')
            patience: Number of evals with no improvement before stopping
            threshold: Minimum change to qualify as improvement
        """
        self.metric_name = metric_name
        self.patience = patience
        self.threshold = threshold

        self.best_value = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check metric after evaluation."""
        if metrics and self.metric_name in metrics:
            current_value = metrics[self.metric_name]

            if self.best_value is None:
                self.best_value = current_value
            elif current_value < self.best_value - self.threshold:
                self.best_value = current_value
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            logger.info(
                f"Early Stopping: {self.metric_name} = {current_value:.4f} "
                f"(patience: {self.patience_counter}/{self.patience})"
            )

            if self.patience_counter >= self.patience:
                logger.warning("Early stopping triggered!")
                control.should_training_stop = True


class ModelCheckpointCallback(TrainerCallback):
    """
    Custom callback to save best model based on a metric.
    
    Tracks best checkpoint and provides reporting.
    """

    def __init__(
        self,
        metric_name: str = "eval_loss",
        mode: str = "min",  # 'min' or 'max'
    ):
        self.metric_name = metric_name
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Update best checkpoint based on metric."""
        if metrics and self.metric_name in metrics:
            current_value = metrics[self.metric_name]

            if (
                (self.mode == "min" and current_value < self.best_value) or
                (self.mode == "max" and current_value > self.best_value)
            ):
                self.best_value = current_value
                self.best_checkpoint = state.best_model_checkpoint

                logger.info(
                    f"Best checkpoint updated: {self.metric_name} = "
                    f"{current_value:.4f} (saved in {self.best_checkpoint})"
                )


# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

class PerformanceAnalyzer:
    """
    Analyze model predictions and identify error patterns.
    """

    def __init__(self):
        self.error_analysis = defaultdict(list)
        self.performance_by_category = defaultdict(list)

    def analyze_qa_predictions(
        self,
        predictions: List[str],
        references: List[str],
        contexts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze QA predictions to identify error patterns.
        
        Args:
            predictions: Model predictions
            references: Ground truth answers
            contexts: Optional context passages
            
        Returns:
            Dictionary with error analysis
        """
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        total = len(predictions)

        # Classify errors
        errors_by_type = defaultdict(list)

        for i, (pred, ref, ctx) in enumerate(
            zip(predictions, references, contexts or [None] * len(predictions))
        ):
            if pred != ref:
                error_type = self._classify_qa_error(pred, ref, ctx)
                errors_by_type[error_type].append({
                    "prediction": pred,
                    "reference": ref,
                    "context": ctx,
                    "index": i,
                })

        return {
            "accuracy": (correct / total) * 100 if total > 0 else 0,
            "total": total,
            "correct": correct,
            "errors_by_type": dict(errors_by_type),
            "error_breakdown": {
                k: len(v) for k, v in errors_by_type.items()
            },
        }

    @staticmethod
    def _classify_qa_error(
        prediction: str,
        reference: str,
        context: Optional[str],
    ) -> str:
        """Classify type of QA error."""
        if not prediction:
            return "no_answer"
        elif prediction.lower() == reference.lower():
            return "case_difference"
        elif reference.lower() in prediction.lower():
            return "partial_answer"
        elif prediction.lower() in reference.lower():
            return "incomplete_answer"
        else:
            return "wrong_answer"

    def analyze_summarization_predictions(
        self,
        predictions: List[str],
        references: List[str],
        rouge_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Analyze summarization predictions.
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            rouge_scores: ROUGE metric scores
            
        Returns:
            Dictionary with analysis
        """
        length_predictions = [len(p.split()) for p in predictions]
        length_references = [len(r.split()) for r in references]

        return {
            "rouge_scores": rouge_scores,
            "avg_generated_length": np.mean(length_predictions),
            "avg_reference_length": np.mean(length_references),
            "length_ratio": (
                np.mean(length_predictions) / np.mean(length_references)
                if np.mean(length_references) > 0 else 0
            ),
            "predictions": predictions,
            "references": references,
        }


# ============================================================================
# TRAINING HISTORY AND REPORTING
# ============================================================================

class TrainingReporter:
    """Generate comprehensive training reports."""

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_qa_report(
        self,
        train_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
        predictions: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate QA training report."""
        report = f"""
        ========================================
        QUESTION ANSWERING TRAINING REPORT
        ========================================
        
        Training Metrics:
        {json.dumps(train_metrics, indent=2)}
        
        Evaluation Metrics:
        {json.dumps(eval_metrics, indent=2)}
        
        Model Configuration:
        {json.dumps(config, indent=2) if config else "Not provided"}
        
        Sample Predictions:
        """

        for i, pred in enumerate(predictions[:5], 1):
            report += f"\n\nPrediction {i}:\n"
            report += f"  Question: {pred.get('question', 'N/A')}\n"
            report += f"  Answer: {pred.get('text', 'N/A')}\n"

        # Save report
        report_path = self.output_dir / "qa_training_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"QA report saved to {report_path}")
        return report

    def generate_summarization_report(
        self,
        train_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
        samples: List[Dict[str, str]],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate summarization training report."""
        report = f"""
        ========================================
        SUMMARIZATION TRAINING REPORT
        ========================================
        
        Training Metrics:
        {json.dumps(train_metrics, indent=2)}
        
        Evaluation Metrics:
        {json.dumps(eval_metrics, indent=2)}
        
        Model Configuration:
        {json.dumps(config, indent=2) if config else "Not provided"}
        
        Sample Summaries:
        """

        for i, sample in enumerate(samples[:3], 1):
            report += f"\n\nSample {i}:\n"
            report += f"  Source (first 200 chars):\n  "
            report += f"{sample.get('article', 'N/A')[:200]}...\n"
            report += f"  Generated Summary:\n  "
            report += f"{sample.get('generated_summary', 'N/A')}\n"
            if 'reference_summary' in sample:
                report += f"  Reference Summary:\n  "
                report += f"{sample.get('reference_summary', 'N/A')}\n"

        # Save report
        report_path = self.output_dir / "summarization_training_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Summarization report saved to {report_path}")
        return report


# ============================================================================
# GRADIENT AND ACTIVATION MONITORING
# ============================================================================

class GradientMonitor:
    """Monitor gradients during training to detect issues."""

    @staticmethod
    def get_gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
        """
        Compute gradient statistics for monitoring training health.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with gradient statistics
        """
        total_norm = 0.0
        param_norms = []
        zero_grad_params = 0

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_norms.append(param_norm.item())
            else:
                zero_grad_params += 1

        total_norm = total_norm ** 0.5

        return {
            "total_norm": float(total_norm),
            "mean_norm": float(np.mean(param_norms)) if param_norms else 0.0,
            "max_norm": float(np.max(param_norms)) if param_norms else 0.0,
            "min_norm": float(np.min(param_norms)) if param_norms else 0.0,
            "zero_grad_params": int(zero_grad_params),
            "num_params_with_grads": len(param_norms),
        }

    @staticmethod
    def detect_gradient_issues(model: torch.nn.Module) -> List[str]:
        """
        Detect common gradient problems.
        
        Returns:
            List of warnings about gradient issues
        """
        warnings = []
        stats = GradientMonitor.get_gradient_stats(model)

        # Check for NaN or Inf
        if not np.isfinite(stats['total_norm']):
            warnings.append("⚠️  Non-finite gradient norm detected!")

        # Check for exploding gradients
        if stats['total_norm'] > 10.0:
            warnings.append("⚠️  Potentially exploding gradients detected!")

        # Check for vanishing gradients
        if stats['total_norm'] < 1e-6 and stats['num_params_with_grads'] > 0:
            warnings.append("⚠️  Potentially vanishing gradients detected!")

        return warnings


if __name__ == "__main__":
    # Example usage
    logger.info("Advanced training utilities loaded successfully")
