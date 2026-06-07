# FinBERT Fine-Tuning Guide

**What this is:** A step-by-step walkthrough for fine-tuning the `ProsusAI/finbert` model on Financial PhraseBank, then pushing your fine-tuned weights to the HuggingFace Hub so your VS Code project can call them via the Inference API — no local GPU needed after training.

**Where you run this:** In a free GPU notebook — [Google Colab](https://colab.research.google.com) (T4 GPU, ~12h free/day) or [Kaggle](https://kaggle.com/code) (T4 GPU, 30h free/week). **Not in VS Code.**

**When to do this:** After you have the baseline evaluation working (Phase 1 below). Fine-tuning is a "do it once offline" task, not part of the live pipeline.

---

## Background: What Is Fine-Tuning?

FinBERT (`ProsusAI/finbert`) is a version of BERT that was already pre-trained on a large corpus of financial text. It already understands financial language reasonably well.

**Fine-tuning** means taking that pre-trained model and training it a little more on *your specific task's* labeled data — in this case, the Financial PhraseBank sentiment labels — so it gets better at exactly the classification you care about.

Think of it like hiring a finance expert (pre-trained FinBERT) and giving them a short company-specific training course (fine-tuning on PhraseBank). They already know the field; you're just sharpening their focus.

**Honest expectation:** Because FinBERT was largely built on Financial PhraseBank already, your accuracy gain from fine-tuning will probably be small (1–3% F1). The value here is learning the process, not the accuracy jump.

---

## Phase 1 — Baseline Evaluation (No GPU Needed)

Do this first, from your normal VS Code environment. It records how good the off-the-shelf model is before any training.

### What you need
- Your `.venv` active
- `HF_TOKEN` set in your `.env`
- `pip install datasets scikit-learn` (add to `requirements.txt`)

### Steps

**1. Create the file `src/eval_baseline.py` in your VS Code project.**

**2. Paste this code:**

```python
from datasets import load_dataset
from huggingface_hub import InferenceClient
from sklearn.metrics import classification_report
import time
from src.config import HF_TOKEN

hf = InferenceClient(token=HF_TOKEN)

# Load Financial PhraseBank — "sentences_allagree" = only sentences all 3 annotators agreed on
# (highest-quality labels, best for evaluation)
ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train")

# PhraseBank has no official test split — we create one (80% train, 20% test)
# seed=42 makes it reproducible: same split every time you run it
ds = ds.train_test_split(test_size=0.2, seed=42)
test_set = ds["test"]

# FinBERT returns label names; map them to integers for sklearn
label_map = {"positive": 2, "negative": 0, "neutral": 1}

y_true = []   # ground-truth labels (integers)
y_pred = []   # model predictions (integers)

print(f"Evaluating {len(test_set)} sentences...")

for i, item in enumerate(test_set):
    # One Inference API call per sentence (FinBERT's 512-token limit means sentence-level anyway)
    result = hf.text_classification(item["sentence"], model="ProsusAI/finbert")
    pred_label = result[0]["label"].lower()   # top prediction

    y_true.append(item["label"])
    y_pred.append(label_map[pred_label])

    # Respect free-tier rate limits: ~10 requests/second max
    time.sleep(0.1)

    if (i + 1) % 50 == 0:
        print(f"  Done {i + 1}/{len(test_set)}")

print("\n=== Baseline FinBERT Results ===")
print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))
```

**3. Run it:** `python -m src.eval_baseline`

**4. Save the output.** You'll compare these numbers against your fine-tuned model later. Example of what to look for:

```
              precision    recall  f1-score   support
    negative       0.XX      0.XX      0.XX       NNN
     neutral       0.XX      0.XX      0.XX       NNN
    positive       0.XX      0.XX      0.XX       NNN
    accuracy                           0.XX      NNNN
```

Record the **weighted F1** number — that's your baseline.

---

## Phase 2 — Fine-Tuning (GPU Notebook)

### One-time setup: Create a HuggingFace account and get a token

1. Go to [huggingface.co](https://huggingface.co) and create a free account if you don't have one.
2. Go to **Settings → Access Tokens → New token**.
3. Name it something like `colab-training`, set role to **Write** (needed to push models).
4. Copy the token — you'll paste it into Colab.

---

### Step 1 — Open a GPU notebook

**Google Colab:**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → New notebook
3. Runtime → Change runtime type → **T4 GPU** → Save
4. Confirm you have a GPU: run `!nvidia-smi` in a cell — you should see `Tesla T4`

**Kaggle (alternative):**
1. Go to [kaggle.com/code](https://kaggle.com/code) → New Notebook
2. Settings (right sidebar) → Accelerator → **GPU T4 x2** → Save
3. Run `!nvidia-smi` to confirm

---

### Step 2 — Install dependencies

Run this in a cell:

```python
!pip install transformers datasets accelerate scikit-learn -q
```

- `transformers` — the HuggingFace library that contains BERT, FinBERT, and the `Trainer` class
- `datasets` — to download Financial PhraseBank
- `accelerate` — makes multi-GPU/mixed-precision training easier (Trainer uses it internally)
- `scikit-learn` — for the F1 metric during training

---

### Step 3 — Log in to HuggingFace Hub

Run this in a cell:

```python
from huggingface_hub import notebook_login
notebook_login()
```

A text box will appear. Paste your **Write** token from Step 0. This lets Colab push your trained model to your HF account.

---

### Step 4 — Load the dataset and tokenizer

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load FinBERT's tokenizer — it knows exactly how to convert text into the
# format FinBERT expects (WordPiece tokens, special [CLS] and [SEP] tokens, etc.)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# Load Financial PhraseBank
ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train")

# Same 80/20 split and seed as Phase 1 — critical so your test set is identical
ds = ds.train_test_split(test_size=0.2, seed=42)

# Map text labels to integers — the model works with numbers, not strings
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}

def tokenize(batch):
    # truncation=True: cut sentences longer than 512 tokens
    # padding="max_length": pad shorter sentences to exactly 512 tokens
    # (they all need to be the same length for batch training)
    enc = tokenizer(batch["sentence"], truncation=True, max_length=512, padding="max_length")
    enc["labels"] = [label2id[l] for l in batch["label"]]
    return enc

# Apply tokenization to every example; batched=True processes 1000 at a time (faster)
tokenized = ds.map(tokenize, batched=True)

# Remove the original text columns — the model only needs token IDs and labels
tokenized = tokenized.remove_columns(["sentence", "label"])

# Tell the dataset to return PyTorch tensors instead of plain Python lists
tokenized.set_format("torch")

print("Train size:", len(tokenized["train"]))
print("Test size:", len(tokenized["test"]))
```

---

### Step 5 — Load the model

```python
from transformers import AutoModelForSequenceClassification

# Load FinBERT with a classification head on top
# num_labels=3 because we have 3 classes: negative, neutral, positive
model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert",
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

print("Model loaded. Parameters:", sum(p.numel() for p in model.parameters()), "total")
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
```

You'll see ~110 million parameters — that's normal for BERT-base. All of them are trainable (we're doing full fine-tuning, not frozen-backbone fine-tuning).

---

### Step 6 — Define the evaluation metric

```python
import numpy as np
from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
    # eval_pred is a tuple of (logits, labels)
    # logits: raw model outputs before softmax, shape (batch_size, 3)
    # labels: ground truth integers
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)   # pick the class with highest score
    return {
        "f1": f1_score(labels, preds, average="weighted"),
        "accuracy": (preds == labels).mean(),
    }
```

---

### Step 7 — Configure training

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="finbert-finetuned",      # local folder to save checkpoints during training

    # --- How long to train ---
    num_train_epochs=3,                  # 3 passes over the training data; more = slower but more accurate
                                         # For PhraseBank, 2–3 is usually optimal

    # --- Batch sizes ---
    per_device_train_batch_size=16,      # 16 examples per GPU step during training
    per_device_eval_batch_size=32,       # 32 during evaluation (no gradients = can use more memory)

    # --- Learning rate ---
    learning_rate=2e-5,                  # Standard for fine-tuning BERT models; don't go higher
    weight_decay=0.01,                   # Regularization; prevents overfitting

    # --- When to evaluate and save ---
    eval_strategy="epoch",               # Evaluate after every epoch (not every N steps)
    save_strategy="best",                # Only save when the model improves
    load_best_model_at_end=True,         # After training, keep the best checkpoint (not just the last)
    metric_for_best_model="f1",          # "Best" means highest weighted F1

    # --- Speed ---
    fp16=True,                           # Mixed precision: uses float16 where safe, halves memory use
                                         # Only works on GPU — don't set this if testing locally

    # --- Hub ---
    push_to_hub=True,                    # Automatically push to HF Hub when done
    hub_model_id="your-username/finbert-financial-phrasebank",   # CHANGE THIS to your HF username
)
```

> **Important:** Replace `your-username` with your actual HuggingFace username before running.

---

### Step 8 — Train

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
```

You will see a progress bar with loss going down each epoch. Training takes roughly **5–10 minutes** on a T4 GPU for this dataset size. At the end of each epoch you'll see the eval F1 — watch it go up (or plateau) across the 3 epochs.

---

### Step 9 — Evaluate and compare against baseline

```python
results = trainer.evaluate()
print("Fine-tuned model results:")
print(f"  F1:       {results['eval_f1']:.4f}")
print(f"  Accuracy: {results['eval_accuracy']:.4f}")
print(f"  Loss:     {results['eval_loss']:.4f}")
```

Compare the F1 here against the baseline number you saved in Phase 1. Even a small improvement (e.g., 0.88 → 0.91) is worth noting.

---

### Step 10 — Push to HuggingFace Hub

```python
trainer.push_to_hub()
print("Model pushed! Find it at: https://huggingface.co/your-username/finbert-financial-phrasebank")
```

This uploads your model weights, tokenizer config, and a model card to your HF profile. It's now a private (or public) hosted model you can call via the Inference API.

---

## Phase 3 — Use Your Fine-Tuned Model in VS Code

Once the model is on the Hub, swap one line in [src/sentiment.py](src/sentiment.py):

```python
# Before — using the baseline off-the-shelf model:
result = hf.text_classification(sentence, model="ProsusAI/finbert")

# After — using your fine-tuned model:
result = hf.text_classification(sentence, model="your-username/finbert-financial-phrasebank")
```

Nothing else changes. The Inference API handles both identically.

> **Note:** By default, newly pushed models are **private** on HuggingFace — only your token can call them. If you want it public (so anyone can use it), go to your model page on huggingface.co → Settings → Make public.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large for T4 | Reduce `per_device_train_batch_size` to 8 |
| `RuntimeError: fp16` error | FP16 not supported on this runtime | Remove `fp16=True` from TrainingArguments |
| `notebook_login()` fails | Token doesn't have Write access | Create a new token with Write role |
| Model push fails | `hub_model_id` not set correctly | Make sure it's `"your-username/model-name"` |
| Eval F1 lower than baseline | Overfitting or LR too high | Try `num_train_epochs=2` or `learning_rate=1e-5` |
| Rate limit error in Phase 1 | Too many API calls too fast | Increase `time.sleep(0.1)` to `time.sleep(0.2)` |

---

## What to Write in Your Report

When documenting this work, include:

1. **Baseline F1** (from Phase 1 — off-the-shelf FinBERT on your test split)
2. **Fine-tuned F1** (from Phase 2 Step 9)
3. **Training config** (epochs, LR, batch size — paste the TrainingArguments)
4. **Why gains are modest**: FinBERT was pre-trained on Financial PhraseBank data, so the baseline is already strong
5. **What you would do with more data**: mention FiQA (aspect-based sentiment) or domain-specific news corpora as potential training additions

---

## Quick Reference: Key Concepts

| Term | Plain English |
|---|---|
| **Pre-training** | Training a model from scratch on a huge unlabeled corpus (done once by Hugging Face/researchers, not by you) |
| **Fine-tuning** | Taking a pre-trained model and training it a little more on your labeled data |
| **Tokenization** | Converting raw text → integer IDs the model can process |
| **Epoch** | One full pass through the entire training dataset |
| **Batch size** | How many examples the model sees at once before updating weights |
| **Learning rate** | How big a step the model takes when updating weights — too high = unstable, too low = slow |
| **F1 score** | Harmonic mean of precision and recall — a balanced accuracy metric, especially useful when classes are unequal in size |
| **fp16** | Half-precision floating point — faster and uses less GPU memory with minimal accuracy loss |
| **Model card** | The README page for a model on HuggingFace Hub |
