"""
IMPLEMENTATION GUIDE - Comprehensive NLP System

Complete documentation covering system architecture, implementation details,
production deployment, and best practices.

Author: NLP Systems Team
Last Updated: 2026-02-13
"""

# ============================================================================
# SECTION 1: SYSTEM OVERVIEW
# ============================================================================

SYSTEM_OVERVIEW = """
COMPREHENSIVE NLP SYSTEM: PRODUCTION-READY TRANSFORMERS

This system implements two fundamental NLP tasks using state-of-the-art
transformer models from Hugging Face:

┌──────────────────────────────────────────────────────────────────┐
│                    NLP SYSTEM ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. QUESTION ANSWERING (Extractive)                             │
│     ├─ Models: BERT, DistilBERT, RoBERTa                        │
│     ├─ Dataset: SQuAD v1.1 (87.6K training examples)            │
│     ├─ Task: Extract answer spans from context                  │
│     └─ Output: Answer text + confidence scores                  │
│                                                                  │
│  2. TEXT SUMMARIZATION (Abstractive)                            │
│     ├─ Models: BART, T5, Pegasus                                │
│     ├─ Dataset: CNN-DailyMail (287K training examples)          │
│     ├─ Task: Generate concise summaries                         │
│     └─ Output: Summary text (abstractive)                       │
│                                                                  │
│  3. EVALUATION FRAMEWORK                                        │
│     ├─ QA Metrics: EM (Exact Match) & F1 Score                  │
│     ├─ Summary Metrics: ROUGE-1, ROUGE-2, ROUGE-L              │
│     ├─ Error Analysis & Pattern Detection                       │
│     └─ Performance Monitoring                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

KEY ARCHITECTURAL PRINCIPLES:

1. MODULAR DESIGN
   - Separate classes for QA and summarization
   - Independent data processing pipelines
   - Pluggable evaluation metrics

2. TRANSFORMER-CENTRIC
   - Leverage pre-trained models from Hugging Face
   - Fine-tuning support with custom training loops
   - Efficient tokenization and batching

3. PRODUCTION-READY
   - Error handling and validation
   - Configurable hyperparameters
   - Comprehensive logging
   - Model checkpointing

4. EVALUATION-FIRST
   - Built-in metric computation
   - Error analysis and debugging
   - Performance monitoring
   - Benchmark comparisons
"""

# ============================================================================
# SECTION 2: DETAILED IMPLEMENTATION
# ============================================================================

QA_IMPLEMENTATION = """
QUESTION ANSWERING IMPLEMENTATION DETAILS

TASK DEFINITION:
Given a question and a context passage, extract the answer span from the
passage that answers the question.

INPUT: (Question: str, Context: str)
OUTPUT: Answer(s) with start position, end position, and confidence score

ARCHITECTURE:

1. TOKENIZATION PHASE
   Input: Question + Context
         ↓
   Tokenization:
     [CLS] question_tokens [SEP] context_tokens [SEP]
                           ↓
   Token Embedding: Each token → 768D vector (for BERT-base)
   Attention Mask: Marks valid tokens (1) and padding (0)

2. MODEL FORWARD PASS
   Encoder Input: Token embeddings + positional embeddings
                  ↓
   12-Layer Transformer Encoder (for BERT-base)
                  ↓
   Output: Hidden states for each token (shape: [seq_len, 768])
                  ↓
   Start Logits: Dense layer projects to [seq_len]
   End Logits:   Dense layer projects to [seq_len]

3. ANSWER EXTRACTION
   Start Probabilities: softmax(start_logits)
   End Probabilities:   softmax(end_logits)
                  ↓
   Top-K Selection: Find top-k highest (start, end) pairs
                  ↓
   Validity Filtering:
     - start_idx < end_idx
     - end_idx - start_idx <= max_answer_length
                  ↓
   Score Calculation:
     score = (P(start_idx) + P(end_idx)) / 2
                  ↓
   Token-to-Text Conversion:
     answer_tokens = tokens[start_idx:end_idx+1]
     answer_text = tokenizer.decode(answer_tokens)

IMPLEMENTATION CODE STRUCTURE:

class QuestionAnsweringSystem:
    
    def __init__(model_name, max_seq_length=384):
        """Initialize tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    def prepare_squad_features(examples):
        """Convert QAExample objects to tokenized tensors."""
        # Key challenge: Finding answer span in tokenized sequence
        # Because tokenization changes positions, must map from
        # character offsets to token offsets
        
        use return_offsets_mapping=True in tokenizer
        to get char_start → token_idx mapping
    
    def train(train_examples, val_examples):
        """Fine-tune on SQuAD dataset."""
        # 1. Prepare features using prepare_squad_features
        # 2. Create DataLoader for batching
        # 3. Setup optimizer with learning rate schedule
        # 4. Training loop:
        #    - Forward pass: outputs start/end logits
        #    - Compute loss: Cross-entropy on token positions
        #    - Backward pass: Gradient computation
        #    - Update: Optimizer step
        # 5. Validation: Compute EM and F1 metrics
    
    def predict(question, context, top_k=1):
        """Extract answer span(s) from context."""
        # 1. Tokenize
        # 2. Forward pass → logits
        # 3. Softmax normalization
        # 4. Top-k span selection
        # 5. Decode tokens to text
        # 6. Return with scores

COMMON CHALLENGES AND SOLUTIONS:

Challenge 1: Answer Span Not in Context
Problem: Reference answer doesn't appear in context (rare in SQuAD)
Solution: Use CLS token position as default (indicates no answer)

Challenge 2: Long Context Handling
Problem: Context may exceed max_seq_length (384 tokens)
Solution: Sliding window with doc_stride (e.g., 128)
         - Create multiple features per example
         - Track which paragraph contains answer
         - Select best answer across all chunks

Challenge 3: Tokenization Mismatch
Problem: Character offsets don't align with token boundaries
Solution: Use return_offsets_mapping from tokenizer
         - Maps each token to (char_start, char_end)
         - Find tokens overlapping with answer span

EVALUATION METRICS:

Exact Match (EM):
    Measures: Percentage of predictions exactly matching reference
    Formula: EM = (num_exact_matches) / total × 100%
    Interpretation: Strict, zero tolerance for deviation
    
F1 Score:
    Measures: Token-level overlap between prediction and reference
    Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)
    Where:
        Precision = |pred_tokens ∩ ref_tokens| / |pred_tokens|
        Recall = |pred_tokens ∩ ref_tokens| / |ref_tokens|
    Range: 0-100%
    Interpretation: More lenient than EM, rewards partial matches

TYPICAL RESULTS:
    Model           | EM    | F1    | Speed
    DistilBERT      | 78.8% | 86.6% | ~10ms/sample
    BERT-base       | 81.8% | 88.7% | ~15ms/sample
    RoBERTa-base    | 85.0% | 91.2% | ~15ms/sample
"""

SUMMARIZATION_IMPLEMENTATION = """
TEXT SUMMARIZATION IMPLEMENTATION DETAILS

TASK DEFINITION:
Given a long document, generate a shorter, coherent summary that captures
the main ideas while respecting input/output length constraints.

INPUT: Document (variable length, often 500+ words)
OUTPUT: Summary (typically 50-150 words)

ARCHITECTURE:

1. DOCUMENT TRUNCATION
   Long Document (e.g., 1500 words)
                  ↓
   Convert to tokens: ~2000 tokens
                  ↓
   Truncate: Keep first max_input_length tokens (e.g., 1024)
             [This is necessary because transformers have fixed context length]
                  ↓
   Result: ~1024 tokens (information loss acceptable for summarization)

2. ENCODER PROCESSING (BART/T5/Pegasus)
   Token Embedding: Each token → hidden_dim (e.g., 768)
   + Positional Encoding
   + Segment Embeddings
                  ↓
   12-Layer Transformer Encoder
                  ↓
   Context Representation: [seq_len, hidden_dim]
   Attention: Each token attends to all other tokens
   
3. DECODER GENERATION (Autoregressive)
   Start: [BOS] token
                  ↓
   Generate Token by Token:
   for t in 1 to max_length:
       1. Embed previous token
       2. Self-attention over generated tokens so far
       3. Cross-attention over encoder context
       4. Feed-forward layers
       5. Output logits for next token
       6. Sample or select top-k token
       7. If [EOS] token → stop
                  ↓
   Result: Sequence of tokens [BOS] w1 w2 ... wn [EOS]
                  ↓
   Decode to Text: tokenizer.decode(token_ids)

4. BEAM SEARCH DECODING
   Instead of greedy selection (top-1), maintain top-k candidates:
   
   Step 1: Start with [BOS]
   Step 2: Generate top-k possible next tokens
           Maintain k hypotheses with scores
   Step 3: For each hypothesis, generate top-k continuations
           Select top-k overall (prune others)
   Step 4: Repeat until all reach [EOS] or max_length
   Step 5: Return best hypothesis by score
   
   Advantages: Better coverage of high-probability sequences
   Computational Cost: ~k× slower than greedy

IMPLEMENTATION CODE STRUCTURE:

class TextSummarizationSystem:
    
    def __init__(model_name, max_input_length=1024):
        """Initialize tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def prepare_summarization_features(examples):
        """Convert articles to tokenized features."""
        # Tokenize articles with truncation
        input_encodings = tokenizer(
            articles,
            max_length=max_input_length,
            truncation=True,  # KEY: Truncate long documents
            padding='max_length',
        )
        
        # Tokenize target summaries
        target_encodings = tokenizer(
            summaries,
            max_length=max_target_length,
            truncation=True,
            padding='max_length',
        )
        
        # For training, include target as labels
        input_encodings['labels'] = target_encodings['input_ids']
    
    def train(train_examples, val_examples):
        """Fine-tune on CNN-DailyMail dataset."""
        # 1. Prepare features
        # 2. Create DataLoader
        # 3. Training loop:
        #    - Forward pass: encoder processes input, decoder generates
        #    - Loss: Cross-entropy on target token predictions
        #    - Backward pass
        #    - Update
        # 4. Validation: Compute ROUGE scores
    
    def summarize(text, max_length=128, num_beams=4):
        """Generate summary using beam search."""
        # 1. Tokenize and truncate document
        # 2. Forward pass through encoder
        # 3. Decode using beam search:
        #    generate(
        #        input_ids,
        #        num_beams=num_beams,  # Beam width
        #        max_length=max_length,
        #        early_stopping=True,
        #        length_penalty=2.0,  # Encourage longer summaries
        #    )
        # 4. Decode token IDs to text
        # 5. Return summary

HANDLING LONG DOCUMENTS:

Key Challenge: Models have maximum sequence length (e.g., 512-1024)
Many documents exceed this limit

Solution 1: Truncate (Used in this system)
  Advantage: Simple, fast
  Disadvantage: Loss of information
  When to use: Abstractive summarization (early content often most important)

Solution 2: Hierarchical Summarization
  1. Split document into chunks
  2. Summarize each chunk
  3. Summarize chunk summaries
  Advantage: Captures full document
  Disadvantage: Slower, more error propagation

Solution 3: Sliding Window
  Similar to truncation but with overlap
  Advantage: Doesn't lose middle content
  Disadvantage: Computationally expensive

EVALUATION METRICS:

ROUGE-1 (Unigram):
    Measures: How many words in reference appear in prediction
    Formula: ROUGE-1 = (matching_unigrams) / (reference_unigrams)
    Example:
        Ref:  "The quick brown fox"
        Pred: "A quick brown dog"
        Matches: {quick, brown}
        ROUGE-1 = 2/4 = 0.50

ROUGE-2 (Bigram):
    Measures: Two-word phrase overlap
    Better correlation with human judgment
    Example:
        Ref:  "the quick brown fox"
        Pred: "a quick brown dog"
        Matches: {quick brown}
        ROUGE-2 = 1/3 = 0.33

ROUGE-L (Longest Common Subsequence):
    Measures: Longest matching word sequence
    Captures sentence structure
    Example:
        Ref:  "the cat sat on the mat"
        Pred: "the cat on the mat"
        LCS: "the cat ... on the mat"

TYPICAL RESULTS:
    Model           | ROUGE-1 | ROUGE-2 | ROUGE-L | Speed
    BART-base       | 40.2    | 17.6    | 36.5    | ~50ms
    T5-base         | 41.5    | 18.9    | 38.5    | ~60ms
    Pegasus         | 44.2    | 21.2    | 40.9    | ~80ms
"""

# ============================================================================
# SECTION 3: TRAINING AND DEPLOYMENT
# ============================================================================

TRAINING_GUIDE = """
TRAINING GUIDE

### Preparing Training Data

For Question Answering:
1. Use SQuAD v1.1 format:
   {
     "data": [
       {
         "title": "Document title",
         "paragraphs": [
           {
             "context": "Passage text...",
             "qas": [
               {
                 "question": "Question?",
                 "id": "q1",
                 "answers": [
                   {"text": "Answer", "answer_start": 15}
                 ]
               }
             ]
           }
         ]
       }
     ]
   }

2. Load with load_squad_dataset() method

For Summarization:
1. Use standard article-summary pairs
2. Load with load_cnn_dailymail_dataset() method
3. Or create custom dataset:
   articles = ["Long article 1...", "Long article 2...", ...]
   summaries = ["Summary 1...", "Summary 2...", ...]

### Training Configuration

QA Configuration:
    config = QAPresets.BALANCED
    - Model: bert-base-uncased
    - Max sequence length: 384
    - Batch size: 16
    - Learning rate: 2e-5
    - Epochs: 3
    - Optimizer: AdamW with weight decay

Summarization Configuration:
    config = SummarizationPresets.BALANCED
    - Model: facebook/bart-base
    - Max input length: 1024
    - Max target length: 128
    - Batch size: 8 (smaller due to larger model)
    - Learning rate: 2e-5
    - Epochs: 3

### Training Process

1. Load and prepare data
   qa_examples = qa_system.load_squad_dataset(split="train")
   train_examples, val_examples = create_train_val_split(
       qa_examples,
       train_ratio=0.8,
   )

2. Configure training
   training_args = TrainingArguments(
       output_dir="./models/qa",
       num_train_epochs=3,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=32,
       learning_rate=2e-5,
       evaluation_strategy="epoch",
       save_strategy="epoch",
       load_best_model_at_end=True,
   )

3. Create trainer
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_features,
       eval_dataset=val_features,
       tokenizer=tokenizer,
       data_collator=DataCollatorWithPadding(tokenizer),
   )

4. Train
   results = trainer.train()

5. Evaluate
   eval_results = trainer.evaluate()
   
   For QA:
       - Compute EM and F1 on validation set
   For Summarization:
       - Compute ROUGE scores on validation set

### Learning Rate Scheduling

Default: Linear schedule with warmup
- First 10% of steps: Linear warmup from 0 to learning_rate
- Remaining 90% of steps: Linear decay to 0

For longer training:
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps,
    )

### Early Stopping

Use EarlyStoppingCallback to prevent overfitting:
    callback = EarlyStoppingCallback(
        metric_name="eval_loss",
        patience=3,  # Stop after 3 evals with no improvement
    )
    trainer.add_callback(callback)

### Gradient Accumulation

Simulate larger batch size:
    training_args = TrainingArguments(
        gradient_accumulation_steps=4,
        per_device_train_batch_size=8,  # Effective: 32
    )

### Mixed Precision Training

Speed up training on modern GPUs:
    training_args = TrainingArguments(
        fp16=True,  # Use float16
        fp16_opt_level="O2",
    )

### Training Monitoring

Monitor in real-time:
    pip install tensorboard
    tensorboard --logdir ./logs

Or use Weights & Biases:
    pip install wandb
    wandb login
    
    training_args = TrainingArguments(
        report_to=["wandb"],
        run_name="qa_finetuning_v1",
    )
"""

DEPLOYMENT_GUIDE = """
PRODUCTION DEPLOYMENT GUIDE

### Model Export

Save trained model:
    qa_system.model.save_pretrained("./model/qa")
    qa_system.tokenizer.save_pretrained("./model/qa")

Load for inference:
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    
    model = AutoModelForQuestionAnswering.from_pretrained("./model/qa")
    tokenizer = AutoTokenizer.from_pretrained("./model/qa")

### Quantization (Size Reduction)

Convert to int8 for smaller models:
    from transformers import AutoModelForQuestionAnswering
    model = AutoModelForQuestionAnswering.from_pretrained("./model/qa")
    
    # Quantize
    from torch.quantization import quantize_dynamic
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    
    Benefits:
    - 4x smaller model size (~25MB → ~6MB for BERT)
    - ~4x faster inference
    - Minimal accuracy loss (usually <1%)

### Batch Inference

Process multiple samples efficiently:
    from torch.utils.data import DataLoader
    
    contexts = [ctx1, ctx2, ...]
    questions = [q1, q2, ...]
    
    # Create batch iterator
    batch_size = 32
    num_batches = (len(contexts) + batch_size - 1) // batch_size
    
    all_answers = []
    for i in range(num_batches):
        batch_ctx = contexts[i*batch_size:(i+1)*batch_size]
        batch_q = questions[i*batch_size:(i+1)*batch_size]
        
        # Process batch
        batch_answers = [
            qa_system.predict(q, c)[0]['text']
            for q, c in zip(batch_q, batch_ctx)
        ]
        all_answers.extend(batch_answers)

### API Deployment (FastAPI)

Create REST API:
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    
    class QARequest(BaseModel):
        question: str
        context: str
    
    @app.post("/predict")
    async def predict(request: QARequest):
        answers = qa_system.predict(
            request.question,
            request.context,
            top_k=1,
        )
        return {"answer": answers[0]['text']}
    
    # Run with:
    # uvicorn api:app --host 0.0.0.0 --port 8000

### Docker Deployment

Create Dockerfile:
    FROM python:3.9
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]

Build and run:
    docker build -t nlp-system:latest .
    docker run -p 8000:8000 nlp-system:latest

### Kubernetes Deployment

Deploy on Kubernetes cluster:
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: nlp-system
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: nlp-system
      template:
        metadata:
          labels:
            app: nlp-system
        spec:
          containers:
          - name: nlp-system
            image: nlp-system:latest
            ports:
            - containerPort: 8000
            resources:
              requests:
                memory: "2Gi"
                cpu: "1"
              limits:
                memory: "4Gi"
                cpu: "2"

### Monitoring and Logging

Track model performance in production:
    import logging
    
    logger = logging.getLogger(__name__)
    
    @app.post("/predict")
    async def predict(request: QARequest):
        start_time = time.time()
        
        answers = qa_system.predict(
            request.question,
            request.context,
        )
        
        inference_time = time.time() - start_time
        logger.info(f"Inference time: {inference_time:.3f}s")
        logger.info(f"Answer: {answers[0]['text']}")
        
        return {"answer": answers[0]['text']}

### A/B Testing

Compare model versions:
    import random
    
    @app.post("/predict")
    async def predict(request: QARequest):
        # Route to model v1 or v2 randomly
        if random.random() < 0.5:
            answers = qa_system_v1.predict(request.question, request.context)
            model_version = "v1"
        else:
            answers = qa_system_v2.predict(request.question, request.context)
            model_version = "v2"
        
        logger.info(f"Used model: {model_version}")
        return {"answer": answers[0]['text'], "model_version": model_version}
"""

# ============================================================================
# SECTION 4: PERFORMANCE OPTIMIZATION
# ============================================================================

OPTIMIZATION = """
PERFORMANCE OPTIMIZATION TECHNIQUES

### Inference Speed Optimization

1. Model Selection
   Latency (ms/sample):
   - DistilBERT: 10-15ms ⭐ Fastest
   - BERT: 15-20ms
   - RoBERTa: 20-25ms
   - ELECTRA: 15-20ms
   
   Choose DistilBERT for low-latency requirements

2. Sequence Length Reduction
   For QA:
   - Default: 384 tokens
   - Optimized: 256 tokens
   - Reduction: ~30% faster
   
   For Summarization:
   - Default: 1024 tokens
   - Optimized: 512 tokens
   - Reduction: ~40% faster

3. Batch Processing
   Single vs batch inference:
   - Single sample: 15ms
   - Batch of 32: 15ms per sample (~480ms total)
   - Speedup: 32× throughput
   - Latency per sample: Same

4. GPU Acceleration
   Without GPU: ~100-200ms per sample
   With GPU: ~15-30ms per sample
   Speedup: 5-10×

5. Quantization
   Full precision (float32): Baseline
   Half precision (float16): 2× faster
   Integer quantization (int8): 4× faster, ~1% accuracy loss

### Memory Optimization

1. Model Distillation
   Compress larger models to smaller ones:
   - DistilBERT: 66M parameters (vs 110M for BERT)
   - Size: ~250MB → ~100MB
   - Speed: ~60% faster
   - Accuracy: ~95% of original

2. Weight Pruning
   Remove less important connections:
   - Can reduce model size by 50%
   - Minimal impact on accuracy
   - Trade-off: More complex to implement

3. Knowledge Distillation
   Train small model to mimic large model:
   - Large model (teacher): Better accuracy
   - Small model (student): Faster inference
   - Use teacher logits as training signal

### Throughput Optimization

For batch processing:
    batch_size = 64  # Optimize based on GPU memory
    
    # Parallel processing with multiple GPUs
    model = torch.nn.DataParallel(model)
    
    # Effective throughput: ~1000 samples/second on single GPU

### Code Optimization

Avoid common bottlenecks:

1. Tokenization in loop (❌ Slow)
   for sample in samples:
       tokens = tokenizer(sample)

   In batch (✓ Fast)
   tokens = tokenizer(samples, batch_encode_plus=True)

2. Model.eval() before inference (✓ Required)
   model.eval()
   with torch.no_grad():
       outputs = model(inputs)

3. Avoid Python loops in inference
   Use vectorized operations with PyTorch

### Caching

Cache common queries:
    from functools import lru_cache
    
    @lru_cache(maxsize=1000)
    def cached_predict(question, context):
        return qa_system.predict(question, context)
    
    # Benefits: If same Q+C repeated, return cached answer
    # Typical cache hit rate: 20-40% for real applications
"""

# ============================================================================
# CONCATENATE ALL SECTIONS
# ============================================================================

COMPLETE_GUIDE = f"""
{SYSTEM_OVERVIEW}

{QA_IMPLEMENTATION}

{SUMMARIZATION_IMPLEMENTATION}

{TRAINING_GUIDE}

{DEPLOYMENT_GUIDE}

{OPTIMIZATION}

================================================================================
END OF IMPLEMENTATION GUIDE
================================================================================
"""

if __name__ == "__main__":
    print(COMPLETE_GUIDE)
