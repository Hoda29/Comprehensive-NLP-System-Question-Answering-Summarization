"""
README - Comprehensive NLP System: Question Answering & Summarization

This document provides complete documentation for the production-ready NLP
system implementing transformer-based question answering and text summarization.
"""

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================

TABLE_OF_CONTENTS = """
1. OVERVIEW
2. SYSTEM ARCHITECTURE
3. INSTALLATION
4. QUICK START
5. DETAILED USAGE GUIDE
6. MODEL ARCHITECTURES
7. EVALUATION METRICS
8. ADVANCED FEATURES
9. TROUBLESHOOTING
10. PERFORMANCE BENCHMARKS
11. CONTRIBUTING
12. LICENSE
"""

# ============================================================================
# 1. OVERVIEW
# ============================================================================

OVERVIEW = """
COMPREHENSIVE NLP SYSTEM: QUESTION ANSWERING & SUMMARIZATION

This system demonstrates production-ready implementations of two fundamental
NLP tasks using state-of-the-art transformer models:

### Task 1: EXTRACTIVE QUESTION ANSWERING
- **Dataset**: SQuAD v1.1 (Stanford Question Answering Dataset)
- **Models**: BERT, DistilBERT, RoBERTa
- **Architecture**: Span-based extraction from context passages
- **Key Features**:
  * Answer span extraction via start/end token classification
  * Confidence scoring based on logit probabilities
  * Top-k predictions with ranking
  * Proper handling of long contexts with sliding windows

### Task 2: ABSTRACTIVE TEXT SUMMARIZATION
- **Dataset**: CNN-DailyMail (346K article-summary pairs)
- **Models**: BART, T5, Pegasus
- **Architecture**: Seq2Seq with transformer encoder-decoder
- **Key Features**:
  * Document truncation for input length limits
  * Beam search decoding with length penalty
  * Variable summary length control
  * Pre-trained models fine-tuned on domain data

### Comprehensive Evaluation
- **QA Metrics**: Exact Match (EM), F1 Score
- **Summarization Metrics**: ROUGE-1, ROUGE-2, ROUGE-L
- **Error Analysis**: Classification of failure modes
- **Performance Monitoring**: Gradient tracking, loss curves
"""

# ============================================================================
# 2. SYSTEM ARCHITECTURE
# ============================================================================

ARCHITECTURE = """
NLP SYSTEM ARCHITECTURE

┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  SQuAD v1.1 (QA)      │      CNN-DailyMail (Summarization)  │
│  • 100K+ examples     │      • 346K examples                │
│  • Passages + Q&A     │      • Articles + summaries         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 PREPROCESSING MODULE                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ QA Preprocessing │         │Sum Preprocessing │         │
│  ├──────────────────┤         ├──────────────────┤         │
│  │ • Tokenization   │         │ • Tokenization   │         │
│  │ • Answer span    │         │ • Truncation     │         │
│  │   identification │         │ • Padding        │         │
│  │ • Attention mask │         │ • Attention mask │         │
│  │ • Padding        │         │ • Special tokens │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER MODEL LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  QA Models       │         │Sum Models        │         │
│  ├──────────────────┤         ├──────────────────┤         │
│  │ • BERT           │         │ • BART           │         │
│  │ • DistilBERT     │         │ • T5             │         │
│  │ • RoBERTa        │         │ • Pegasus        │         │
│  │                  │         │                  │         │
│  │ Output: Logits   │         │ Output: Logits   │         │
│  │ (start, end)     │         │ (token probs)    │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  POSTPROCESSING LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ QA Postprocess   │         │Sum Postprocess   │         │
│  ├──────────────────┤         ├──────────────────┤         │
│  │ • Span extraction│         │ • Beam search    │         │
│  │ • Score ranking  │         │ • Decoding       │         │
│  │ • Text retrieval │         │ • Length control │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  EVALUATION LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  • Exact Match & F1 Score (QA)                             │
│  • ROUGE Metrics (Summarization)                           │
│  • Error Analysis & Reporting                              │
│  • Performance Monitoring                                   │
└─────────────────────────────────────────────────────────────┘

DATA FLOW FOR QUESTION ANSWERING

Input: (Question, Context)
         ↓
   Tokenization: [CLS] question [SEP] context [SEP]
         ↓
   Model Forward Pass: Get start_logits, end_logits
         ↓
   Softmax Normalization: P(start), P(end)
         ↓
   Span Extraction: Find (i, j) that maximizes P(start_i) + P(end_j)
         ↓
   Token to Text Conversion: Decode tokens to answer string
         ↓
Output: Answer(s) with confidence scores

DATA FLOW FOR SUMMARIZATION

Input: Long Article (e.g., 1500 words)
         ↓
   Truncation: Limit to max_input_length (e.g., 1024 tokens)
         ↓
   Tokenization: Convert to token IDs
         ↓
   Model Forward Pass: Generate embeddings in encoder
         ↓
   Beam Search Decoding: Generate summary tokens
         ↓
   Early Stopping: Stop if EOS token or max_length reached
         ↓
   Token to Text Conversion: Decode tokens to summary
         ↓
Output: Abstractive summary (e.g., 100 words)
"""

# ============================================================================
# 3. INSTALLATION
# ============================================================================

INSTALLATION = """
INSTALLATION GUIDE

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)
- 8GB+ RAM (16GB+ recommended for training)
- 20GB+ disk space (for models and datasets)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd nlp-system
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; import transformers; print('✓ Installation successful')"
```

### Optional: GPU Setup (for faster training)
```bash
# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install GPU-accelerated dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Dependency Overview
- **transformers**: Hugging Face transformer models and utilities
- **datasets**: Dataset loading and processing
- **torch**: Deep learning framework
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **rouge-score**: ROUGE metric computation
- **tqdm**: Progress bars
"""

# ============================================================================
# 4. QUICK START
# ============================================================================

QUICK_START = """
QUICK START GUIDE

### Run Complete Demo (QA + Summarization)
```bash
python demo_nlp_system.py --task all
```

### Run Only Question Answering Demo
```bash
python demo_nlp_system.py --task qa --qa-samples 100
```

### Run Only Summarization Demo
```bash
python demo_nlp_system.py --task summarization --sum-samples 50
```

### Example 1: Simple Question Answering
```python
from nlp_system import QuestionAnsweringSystem

# Initialize system
qa_system = QuestionAnsweringSystem(model_name="distilbert-base-uncased")

# Ask a question
context = "The Great Wall of China is over 13,000 miles long."
question = "How long is the Great Wall of China?"

answers = qa_system.predict(question, context, top_k=3)
for ans in answers:
    print(f"Answer: {ans['text']} (confidence: {ans['score']:.4f})")
```

### Example 2: Simple Text Summarization
```python
from nlp_system import TextSummarizationSystem

# Initialize system
sum_system = TextSummarizationSystem(model_name="facebook/bart-base")

# Summarize text
article = '''
    Artificial Intelligence is transforming the world. Machine learning
    enables computers to learn from data without explicit programming.
    Deep learning uses neural networks inspired by biological neurons.
    These technologies power modern applications like recommendation
    systems, autonomous vehicles, and natural language processing.
    The field continues to advance rapidly with new architectures
    and training techniques emerging regularly.
'''

summary = sum_system.summarize(article)
print(f"Summary: {summary}")
```

### Example 3: Complete Training Pipeline
```python
from nlp_system import QuestionAnsweringSystem
from config import QAPresets

# Use preset configuration
config = QAPresets.BALANCED

# Initialize with config
qa_system = QuestionAnsweringSystem(
    model_name=config.model_name,
    max_seq_length=config.max_seq_length,
)

# Load data
train_examples = qa_system.load_squad_dataset(split="train", max_samples=1000)
val_examples = qa_system.load_squad_dataset(split="validation", max_samples=200)

# Train
results = qa_system.train(
    train_examples=train_examples,
    val_examples=val_examples,
    num_epochs=config.num_epochs,
    batch_size=config.batch_size,
)

print(f"Training completed: {results}")
```
"""

# ============================================================================
# 5. DETAILED USAGE GUIDE
# ============================================================================

DETAILED_USAGE = """
DETAILED USAGE GUIDE

### Question Answering System

#### Initialization with Different Models
```python
# Use lighter DistilBERT model (faster)
qa_light = QuestionAnsweringSystem(
    model_name="distilbert-base-uncased",
    max_seq_length=256,
)

# Use larger BERT model (more accurate)
qa_full = QuestionAnsweringSystem(
    model_name="bert-base-uncased",
    max_seq_length=384,
)

# Use RoBERTa (often better performance)
qa_roberta = QuestionAnsweringSystem(
    model_name="roberta-base",
    max_seq_length=384,
)
```

#### Loading and Preprocessing Data
```python
# Load full SQuAD training set
train_data = qa_system.load_squad_dataset(split="train")
# Returns: List[QAExample] with 87,599 examples

# Load with limit (for testing)
small_train = qa_system.load_squad_dataset(
    split="train",
    max_samples=1000,
)

# Create train/validation split
from nlp_system import create_train_val_split
train, val = create_train_val_split(small_train, train_ratio=0.8)
```

#### Making Predictions
```python
# Single prediction
context = "Paris is the capital of France."
question = "What is the capital of France?"

# Get top 1 answer
answers = qa_system.predict(question, context, top_k=1)
print(answers[0]['text'])  # Output: "Paris"

# Get top 5 answers (for comparison)
top_5 = qa_system.predict(question, context, top_k=5)
for i, ans in enumerate(top_5, 1):
    print(f"{i}. {ans['text']} ({ans['score']:.4f})")
```

#### Batch Prediction (for efficiency)
```python
contexts = [ctx1, ctx2, ctx3, ...]
questions = [q1, q2, q3, ...]

predictions = []
for ctx, q in zip(contexts, questions):
    pred = qa_system.predict(q, ctx, top_k=1)
    predictions.append(pred[0]['text'])
```

### Summarization System

#### Initialization with Different Models
```python
# BART (balanced)
sum_bart = TextSummarizationSystem(
    model_name="facebook/bart-base",
    max_input_length=1024,
)

# T5 (versatile)
sum_t5 = TextSummarizationSystem(
    model_name="t5-base",
    max_input_length=512,
)

# Pegasus (CNN-DailyMail specific)
sum_pegasus = TextSummarizationSystem(
    model_name="google/pegasus-cnn_dailymail",
    max_input_length=1024,
)
```

#### Summarizing Documents
```python
# Summarize with default settings
summary = sum_system.summarize(long_article)

# Control summary length
short_summary = sum_system.summarize(
    article,
    max_length=50,  # Max 50 tokens
    min_length=20,  # At least 20 tokens
)

# Use different beam widths (higher = slower but often better)
summary_large = sum_system.summarize(
    article,
    num_beams=8,  # Default is 4
)

summary_fast = sum_system.summarize(
    article,
    num_beams=1,  # Greedy decoding (faster)
)
```

#### Loading CNN-DailyMail Dataset
```python
# Load training set
train_data = sum_system.load_cnn_dailymail_dataset(split="train")
# Returns: List[SummaryExample] with 287,227 examples

# Load with limit
small_data = sum_system.load_cnn_dailymail_dataset(
    split="train",
    max_samples=5000,
)

# Check example
example = small_data[0]
print(f"Article length: {len(example.source.split())} words")
print(f"Summary length: {len(example.target.split())} words")
```
"""

# ============================================================================
# 6. MODEL ARCHITECTURES
# ============================================================================

MODELS = """
MODEL ARCHITECTURES

### Question Answering Models

#### BERT (Bidirectional Encoder Representations from Transformers)
- **Architecture**: 12-layer transformer encoder
- **Vocab Size**: 30,522 tokens
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Total Parameters**: 110M
- **Advantages**:
  * Strong bidirectional context understanding
  * Pre-trained on Wikipedia + BookCorpus
  * Works well for extractive tasks
- **Training Speed**: ~2 hours (SQuAD on 1 GPU)
- **Inference Speed**: ~10ms per sample

#### DistilBERT
- **Architecture**: 6-layer transformer encoder (distilled from BERT)
- **Parameters**: 66M (40% of BERT)
- **Speed**: 60% faster than BERT
- **Accuracy**: 95% of BERT performance
- **Advantages**:
  * Lighter weight, faster inference
  * Good for resource-constrained environments
  * Nearly equivalent performance to BERT
- **Use Case**: Production systems with latency constraints

#### RoBERTa (Robustly Optimized BERT)
- **Architecture**: 12-layer transformer encoder
- **Training Data**: 160GB text (vs 16GB for BERT)
- **Parameters**: 355M (base), 110M (base smaller)
- **Advantages**:
  * Better pre-training procedure
  * Often better performance than BERT
  * Better on downstream tasks
- **Use Case**: When maximum accuracy is required

### Summarization Models

#### BART (Denoising Autoencoder for Sequence-to-Sequence)
- **Architecture**: Transformer encoder-decoder
- **Encoder**: 12 layers (similar to BERT)
- **Decoder**: 12 layers (autoregressive)
- **Parameters**: 406M (base)
- **Pre-training**: Denoising tasks on large corpora
- **Advantages**:
  * Strong on both extractive and abstractive tasks
  * Good generalization across domains
  * Efficient training
- **Training Speed**: ~6 hours (CNN-DailyMail on 1 GPU)

#### T5 (Text-to-Text Transfer Transformer)
- **Architecture**: Unified encoder-decoder
- **Treats all tasks**: As text generation tasks
- **Sizes**: Small (60M) → XXL (11B) parameters
- **Pre-training**: Supervised and unsupervised tasks
- **Advantages**:
  * Very versatile (many NLP tasks)
  * Flexible prompt engineering
  * Strong performance across tasks
- **Use Case**: Multi-task systems

#### Pegasus (Pre-training with Extracted Gap-sentences)
- **Architecture**: Encoder-decoder transformer
- **Parameters**: 223M (base)
- **Pre-training**: Gap-sentence generation
- **Advantages**:
  * Specifically designed for summarization
  * Pre-trained on news articles (domain-relevant)
  * Often best results on CNN-DailyMail
- **Use Case**: News article summarization

### Model Selection Guidelines

| Task | Dataset Size | Latency | Accuracy | Recommended |
|------|-------------|---------|----------|------------|
| QA - Demo | <1K | Fast | Lower | DistilBERT |
| QA - Production | >10K | Normal | High | BERT/RoBERTa |
| Sum - Demo | <5K | Fast | Lower | BART-base |
| Sum - Production | >50K | Normal | High | BART/Pegasus |
| Sum - News | >100K | Normal | Highest | Pegasus |
"""

# ============================================================================
# 7. EVALUATION METRICS
# ============================================================================

EVALUATION = """
EVALUATION METRICS

### Question Answering Metrics

#### Exact Match (EM)
- Percentage of predictions that exactly match the reference answer
- Case-sensitive by default
- Formula: EM = (# correct) / (# total)
- Range: 0-100%
- Interpretation: Strict metric, penalizes any deviation

Example:
  Reference: "Paris"
  Prediction: "Paris" → EM = 1.0
  Prediction: "paris" → EM = 0.0
  Prediction: "The city Paris" → EM = 0.0

#### F1 Score
- Measures overlap between prediction and reference tokens
- More lenient than EM, rewards partial matches
- Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
  Where:
    Precision = |pred ∩ ref| / |pred|
    Recall = |pred ∩ ref| / |ref|
- Range: 0-100%
- Interpretation: Token-level agreement

Example:
  Reference: "The Great Wall of China"
  Prediction: "Great Wall of China"
  Precision: 3/4 = 0.75
  Recall: 3/5 = 0.60
  F1: 2*(0.75*0.60)/(0.75+0.60) = 0.67

#### Computing QA Metrics
```python
from nlp_system import EvaluationMetrics

predictions = ["Paris", "12,000 miles", ...]
references = ["Paris", "13,000 miles", ...]

metrics = EvaluationMetrics.compute_qa_metrics(
    predictions,
    references,
)
print(f"EM: {metrics['exact_match']:.2f}%")
print(f"F1: {metrics['f1_score']:.2f}%")
```

### Summarization Metrics

#### ROUGE Scores
ROUGE = Recall-Oriented Understudy for Gisting Evaluation

Measures n-gram overlap between generated and reference summaries.

##### ROUGE-1 (Unigram Recall)
- Measures single word overlap
- Formula: ROUGE-1 = (# matching unigrams) / (# reference unigrams)
- High correlation with content coverage
- Less sensitive to word order

Example:
  Reference: "The cat sat on the mat"
  Prediction: "A cat on the mat"
  Matching words: {cat, on, the, mat}
  ROUGE-1 = 4/6 = 0.67

##### ROUGE-2 (Bigram Recall)
- Measures two-word phrase overlap
- Captures more context than ROUGE-1
- Better correlation with human judgment
- Sensitive to content consecutiveness

Example:
  Reference: "the cat sat on the mat"
  Prediction: "a cat on the mat"
  Matching bigrams: {on the, the mat}
  ROUGE-2 = 2/5 = 0.40

##### ROUGE-L (Longest Common Subsequence)
- Measures longest matching sequence
- Captures sentence-level structure
- Less sensitive to ordering changes
- More human-like evaluation

Example:
  Reference: "the cat sat on the mat"
  Prediction: "the cat on the mat"
  LCS: "the cat ... on the mat"
  ROUGE-L ≈ 0.86

#### Computing ROUGE Scores
```python
from nlp_system import EvaluationMetrics

predictions = ["AI is transforming industries", ...]
references = ["Artificial Intelligence transforms sectors", ...]

scores = EvaluationMetrics.compute_rouge_scores(
    predictions,
    references,
)
print(f"ROUGE-1: {scores['rouge1']:.2f}")
print(f"ROUGE-2: {scores['rouge2']:.2f}")
print(f"ROUGE-L: {scores['rougeL']:.2f}")
```

### Typical Performance Benchmarks

| Model | Dataset | EM | F1 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|----|----|---------|---------|---------|
| DistilBERT | SQuAD | 78.8 | 86.6 | - | - | - |
| BERT | SQuAD | 81.8 | 88.7 | - | - | - |
| BART-base | CNN-DM | - | - | 40.2 | 17.6 | 36.5 |
| T5-base | CNN-DM | - | - | 41.5 | 18.9 | 38.5 |
| Pegasus | CNN-DM | - | - | 44.2 | 21.2 | 40.9 |
"""

# ============================================================================
# 8. ADVANCED FEATURES
# ============================================================================

ADVANCED = """
ADVANCED FEATURES

### Configuration Management
```python
from config import QAPresets, SummarizationPresets

# Use preset configurations
qa_config = QAPresets.BALANCED
sum_config = SummarizationPresets.POWERFUL

# Customize configuration
qa_config.batch_size = 32
qa_config.learning_rate = 1e-5
qa_config.num_epochs = 5

# Save and load configurations
qa_config.to_json("my_qa_config.json")
loaded_config = QAConfig.from_json("my_qa_config.json")
```

### Custom Training Callbacks
```python
from training_utils import MetricsCallback, EarlyStoppingCallback

# Log detailed metrics
metrics_callback = MetricsCallback(log_dir="./training_logs")

# Stop early if validation loss plateaus
early_stop = EarlyStoppingCallback(
    metric_name="eval_loss",
    patience=3,  # Stop after 3 evals with no improvement
)

# Use callbacks during training
trainer = Trainer(
    model=model,
    callbacks=[metrics_callback, early_stop],
)
```

### Performance Analysis
```python
from training_utils import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Analyze QA predictions
qa_analysis = analyzer.analyze_qa_predictions(
    predictions=pred_list,
    references=ref_list,
    contexts=context_list,
)

print(f"Accuracy: {qa_analysis['accuracy']:.2f}%")
print(f"Error breakdown: {qa_analysis['error_breakdown']}")

# Identify error patterns
for error_type, examples in qa_analysis['errors_by_type'].items():
    print(f"\\n{error_type}: {len(examples)} errors")
    for ex in examples[:2]:
        print(f"  Pred: {ex['prediction']}")
        print(f"  Ref: {ex['reference']}")
```

### Gradient Monitoring
```python
from training_utils import GradientMonitor

# During training loop
optimizer.step()
stats = GradientMonitor.get_gradient_stats(model)
print(f"Gradient norm: {stats['total_norm']:.4f}")

# Detect problems
warnings = GradientMonitor.detect_gradient_issues(model)
for warning in warnings:
    logger.warning(warning)
```

### Batch Processing
```python
# Process multiple samples efficiently
texts = [article1, article2, article3, ...]

# Using batch summarization
from torch.utils.data import DataLoader
batch_size = 8
loader = DataLoader(texts, batch_size=batch_size)

summaries = []
for batch in loader:
    batch_summaries = [sum_system.summarize(t) for t in batch]
    summaries.extend(batch_summaries)
```
"""

# ============================================================================
# 9. TROUBLESHOOTING
# ============================================================================

TROUBLESHOOTING = """
TROUBLESHOOTING GUIDE

### Common Issues and Solutions

#### Issue 1: Out of Memory (OOM) Error
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size:
   ```python
   training_args = TrainingArguments(
       per_device_train_batch_size=4,  # Reduce from 16
   )
   ```

2. Reduce sequence length:
   ```python
   qa_system = QuestionAnsweringSystem(
       max_seq_length=256,  # Reduce from 384
   )
   ```

3. Use gradient accumulation:
   ```python
   training_args = TrainingArguments(
       gradient_accumulation_steps=4,  # Simulate larger batches
   )
   ```

4. Use smaller model:
   ```python
   model_name = "distilbert-base-uncased"  # Instead of bert-large
   ```

5. Enable mixed precision training:
   ```python
   training_args = TrainingArguments(
       fp16=True,  # Use half precision
   )
   ```

#### Issue 2: Poor Model Performance
**Check**:
1. Data quality: Are labels correct? Are examples representative?
2. Learning rate: Too high (loss spikes), too low (no improvement)
3. Batch size: Try different values (16, 32, 64)
4. Epochs: Increase if underfitting, decrease if overfitting
5. Hyperparameters: Use learning rate schedule, warmup

```python
# Diagnostic training with smaller dataset
small_train = examples[:1000]
small_val = examples[1000:1200]

results = qa_system.train(
    small_train,
    small_val,
    num_epochs=1,
    batch_size=8,
)
# Monitor loss curve to diagnose issues
```

#### Issue 3: Slow Inference
**Solutions**:
1. Use smaller model:
   ```python
   qa = QuestionAnsweringSystem("distilbert-base-uncased")
   ```

2. Use GPU:
   ```python
   qa = QuestionAnsweringSystem(device="cuda")
   ```

3. Reduce sequence length:
   ```python
   qa = QuestionAnsweringSystem(max_seq_length=256)
   ```

4. Use quantization:
   ```python
   from transformers import quantization_utils
   # Quantize model to int8
   ```

#### Issue 4: Reproducibility
```python
import random
import numpy as np
import torch

# Set all seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Set deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

#### Issue 5: CUDA Not Available
```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Force CPU usage
qa = QuestionAnsweringSystem(device="cpu")
```

### Debug Mode
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed info about:
# - Data loading
# - Tokenization
# - Model forward passes
# - Training steps
```

### Performance Profiling
```python
import cProfile
import pstats

# Profile inference
profiler = cProfile.Profile()
profiler.enable()

predictions = qa_system.predict(question, context)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```
"""

# ============================================================================
# CONCATENATE ALL SECTIONS
# ============================================================================

FULL_README = f"""
{OVERVIEW}

{ARCHITECTURE}

{INSTALLATION}

{QUICK_START}

{DETAILED_USAGE}

{MODELS}

{EVALUATION}

{ADVANCED}

{TROUBLESHOOTING}

================================================================================
END OF README
================================================================================
"""

if __name__ == "__main__":
    print(FULL_README)
