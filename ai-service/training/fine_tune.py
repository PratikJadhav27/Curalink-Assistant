"""
Curalink Custom LLM - Fine-tuning Script
=========================================
Base Model  : google/flan-t5-base  (250M params, instruction-tuned seq2seq)
Technique   : LoRA (Parameter-Efficient Fine-Tuning via PEFT)
Dataset     : medalpaca/medical_meadow_medqa + custom medical synthesis examples
Target Tasks:
  1. Medical Query Expansion  → "Expand: <query>" → optimized boolean query
  2. Medical Synthesis        → "Summarize research: <context>" → medical summary

HOW TO RUN:
  1. Open Google Colab (https://colab.research.google.com)
  2. Set Runtime → Change runtime type → T4 GPU
  3. Upload this file OR paste the contents into a Colab notebook cell
  4. Run every cell in order
  5. The trained model will be pushed to HuggingFace Hub at:
     https://huggingface.co/Pratik-027/curalink-medical-llm

REQUIREMENTS (install in Colab first):
  pip install transformers==4.40.0 peft==0.10.0 datasets==2.19.0 
              accelerate==0.29.3 bitsandbytes==0.43.1 huggingface_hub sentencepiece

ESTIMATED TIME: ~60-90 minutes on Colab T4 GPU
"""

# ─────────────────────────────────────────────
# CELL 1: Install dependencies
# ─────────────────────────────────────────────
# Run this in a Colab cell:
# !pip install -q transformers==4.40.0 peft==0.10.0 datasets==2.19.0 accelerate==0.29.3 bitsandbytes==0.43.1 sentencepiece huggingface_hub

# ─────────────────────────────────────────────
# CELL 2: HuggingFace Login
# ─────────────────────────────────────────────
# from huggingface_hub import login
# login(token="YOUR_HF_TOKEN_HERE")   # <-- paste your HF write token here

# ─────────────────────────────────────────────
# CELL 3: The full fine-tuning script
# ─────────────────────────────────────────────
import os
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType

# ── Config ──────────────────────────────────────────────────────────────
BASE_MODEL   = "google/flan-t5-base"
OUTPUT_DIR   = "./curalink-medical-llm"
HF_REPO_ID   = "Pratik-027/curalink-medical-llm"   # ← your HF Hub repo
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 256
BATCH_SIZE     = 8
NUM_EPOCHS     = 3
LEARNING_RATE  = 3e-4

# ── Step 1: Load + Format Medical Dataset ───────────────────────────────
print("📚 Loading medical QA dataset...")
raw = load_dataset("medalpaca/medical_meadow_medqa", split="train")

# We'll sample 8000 rows to keep training time reasonable on Colab
raw = raw.shuffle(seed=42).select(range(min(8000, len(raw))))

# Format into input-output pairs (flan-t5 text-to-text format)
def format_medqa(example):
    """Format MedQA entries into Curalink-style synthesis prompts."""
    instruction = example.get("instruction", "")
    inp         = example.get("input", "")
    output      = example.get("output", "")

    # Build a medical research synthesis prompt
    if inp:
        input_text = (
            f"Medical Research Assistant. Answer this clinical question based on evidence:\n"
            f"Question: {instruction}\nContext: {inp[:400]}"
        )
    else:
        input_text = (
            f"Medical Research Assistant. Answer this clinical question:\n"
            f"Question: {instruction}"
        )
    return {"input_text": input_text, "target_text": output}

print("🔧 Formatting dataset...")
formatted = raw.map(format_medqa, remove_columns=raw.column_names)

# ── Step 2: Add custom Query Expansion examples ──────────────────────────
# These teach the model the specific query expansion task Curalink uses
print("➕ Adding custom query expansion training examples...")
custom_examples = [
    {
        "input_text": "Expand medical search query for PubMed. Disease: diabetes. Query: latest treatments",
        "target_text": "diabetes AND (treatment OR therapy OR intervention OR management) AND (insulin resistance OR type 2 diabetes OR hyperglycemia) AND (2020 OR 2021 OR 2022 OR 2023 OR 2024)",
    },
    {
        "input_text": "Expand medical search query for PubMed. Disease: lung cancer. Query: immunotherapy options",
        "target_text": "lung cancer AND (immunotherapy OR checkpoint inhibitors OR pembrolizumab OR nivolumab OR atezolizumab) AND (clinical trial OR randomized controlled trial OR phase 3)",
    },
    {
        "input_text": "Expand medical search query for PubMed. Disease: Alzheimer's disease. Query: new drug research",
        "target_text": "Alzheimer's disease AND (drug therapy OR pharmacotherapy OR novel treatment OR disease modifying therapy) AND (amyloid beta OR tau protein OR neurodegeneration) AND (2021 OR 2022 OR 2023 OR 2024)",
    },
    {
        "input_text": "Expand medical search query for PubMed. Disease: hypertension. Query: clinical trials",
        "target_text": "hypertension AND (clinical trial OR randomized controlled trial OR RCT) AND (antihypertensive OR blood pressure reduction OR cardiovascular risk) AND (RECRUITING OR ACTIVE)",
    },
    {
        "input_text": "Expand medical search query for PubMed. Disease: breast cancer. Query: hormone therapy side effects",
        "target_text": "breast cancer AND (hormone therapy OR endocrine therapy OR tamoxifen OR aromatase inhibitor) AND (side effects OR adverse events OR toxicity OR quality of life)",
    },
    {
        "input_text": "Expand medical search query for PubMed. Disease: COVID-19. Query: long term effects",
        "target_text": "COVID-19 AND (long COVID OR post-acute sequelae OR PASC OR persistent symptoms) AND (fatigue OR cognitive impairment OR cardiovascular OR pulmonary) AND (2022 OR 2023 OR 2024)",
    },
    {
        "input_text": "Expand medical search query for PubMed. Disease: depression. Query: cognitive behavioral therapy",
        "target_text": "depression AND (cognitive behavioral therapy OR CBT OR psychotherapy OR psychological intervention) AND (efficacy OR randomized controlled trial OR meta-analysis OR systematic review)",
    },
    {
        "input_text": "Expand medical search query for PubMed. Disease: rheumatoid arthritis. Query: biologic drugs",
        "target_text": "rheumatoid arthritis AND (biologic therapy OR TNF inhibitor OR JAK inhibitor OR methotrexate OR tocilizumab OR adalimumab) AND (disease activity OR remission OR joint damage)",
    },
]
custom_ds = Dataset.from_list(custom_examples)
formatted = Dataset.from_list(
    list(formatted) + list(custom_ds)
)

print(f"✅ Total training examples: {len(formatted)}")

# ── Step 3: Train / Eval Split ───────────────────────────────────────────
split = formatted.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
eval_ds  = split["test"]

# ── Step 4: Load Tokenizer + Model ──────────────────────────────────────
print(f"🤖 Loading base model: {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# ── Step 5: Apply LoRA Adapters ─────────────────────────────────────────
print("🔗 Applying LoRA adapters...")
lora_config = LoraConfig(
    task_type     = TaskType.SEQ_2_SEQ_LM,
    r             = 16,         # LoRA rank (higher = more parameters = better quality)
    lora_alpha    = 32,         # LoRA scaling factor
    lora_dropout  = 0.05,       # Dropout to prevent overfitting
    target_modules= ["q", "v"], # Apply LoRA to attention query + value projections
    bias          = "none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~0.7% trainable params (only LoRA layers, not the full 250M)

# ── Step 6: Tokenize ────────────────────────────────────────────────────
def tokenize(batch):
    inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        batch["target_text"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = labels["input_ids"]
    # Replace padding token id with -100 so it's ignored in loss calculation
    inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in inputs["labels"]
    ]
    return inputs

print("🔢 Tokenizing dataset...")
train_tokenized = train_ds.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])
eval_tokenized  = eval_ds.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])

# ── Step 7: Training Arguments ──────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate               = LEARNING_RATE,
    warmup_steps                = 100,
    weight_decay                = 0.01,
    logging_dir                 = f"{OUTPUT_DIR}/logs",
    logging_steps               = 50,
    eval_strategy               = "epoch",   # renamed from evaluation_strategy in transformers>=4.41
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    predict_with_generate       = True,
    fp16                        = False,     # Set False to avoid GradScaler issues on some T4 setups
    dataloader_pin_memory       = False,     # Avoids OOM on Colab
    report_to                   = "none",    # Disable wandb
)

# ── Step 8: Trainer ─────────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_tokenized,
    eval_dataset    = eval_tokenized,
    processing_class= tokenizer,   # renamed from tokenizer= in transformers>=4.46
    data_collator   = data_collator,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("🚀 Starting fine-tuning...")
trainer.train()
print("✅ Training complete!")

# ── Step 9: Save + Merge Adapters + Push to HF Hub ─────────────────────
print("💾 Saving model...")
# Save LoRA adapter weights
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Merge LoRA adapters into the base model for efficient inference
print("🔗 Merging LoRA adapters into base model for inference...")
from peft import PeftModel
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{OUTPUT_DIR}-merged")
tokenizer.save_pretrained(f"{OUTPUT_DIR}-merged")

# Push merged model to HuggingFace Hub
print(f"📤 Pushing to HuggingFace Hub: {HF_REPO_ID}...")
merged_model.push_to_hub(HF_REPO_ID, private=False)
tokenizer.push_to_hub(HF_REPO_ID, private=False)

print(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅ TRAINING & UPLOAD COMPLETE!                          ║
║                                                          ║
║  Your model is live at:                                  ║
║  https://huggingface.co/{HF_REPO_ID}  ║
╚══════════════════════════════════════════════════════════╝
""")
