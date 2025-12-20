"""
Colab-Optimized LoRA Fine-tuning Training Script
Automatically mounts Google Drive and saves trained weights
Designed for Google Colab with T4/P100 GPU

This script should be run in Google Colab before evaluation.
After training, run colab_evaluation.py to evaluate the LoRA model.
"""

import os
import sys
import signal
import torch
import gc
import shutil
from datetime import datetime
from typing import Dict, List, Any

# Signal patching for compatibility
if not hasattr(signal, 'SIGALRM'):
    signal.SIGALRM = 14
if not hasattr(signal, 'SIGCHLD'):
    signal.SIGCHLD = 17
if not hasattr(signal, 'SIGUSR1'):
    signal.SIGUSR1 = 10
if not hasattr(signal, 'SIGUSR2'):
    signal.SIGUSR2 = 12

if not hasattr(signal, 'alarm'):
    def _dummy_alarm(seconds):
        return 0
    signal.alarm = _dummy_alarm

_original_signal_signal = signal.signal
def _safe_signal_handler(sig, handler):
    try:
        return _original_signal_signal(sig, handler)
    except (ValueError, OSError, AttributeError, TypeError):
        return None

signal.signal = _safe_signal_handler
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import Dataset
from data_loader import load_all_datasets
from config import BASE_MODEL_NAME, LORA_CONFIG


def mount_google_drive():
    """
    Mount Google Drive for saving checkpoints
    Only works in Google Colab
    """
    try:
        from google.colab import drive
        print("\n" + "="*80)
        print("MOUNTING GOOGLE DRIVE")
        print("="*80)
        print("\n📁 Attempting to mount Google Drive...")
        print("   (You may see a popup asking for permission)")
        
        drive.mount('/content/drive')
        print("\n✅ Google Drive mounted successfully!")
        print("   Location: /content/drive/MyDrive/")
        
        return True
    except ImportError:
        print("\n⚠️  WARNING: Not running in Google Colab!")
        print("   This script is optimized for Colab.")
        print("   Checkpoint will be saved locally.")
        return False
    except Exception as e:
        print(f"\n⚠️  WARNING: Could not mount Google Drive: {e}")
        print("   Checkpoint will be saved locally.")
        return False


def get_checkpoint_dir(use_drive: bool) -> str:
    """
    Determine checkpoint directory based on Google Drive availability
    """
    if use_drive:
        checkpoint_dir = "/content/drive/MyDrive/lora_checkpoint"
        print(f"\n💾 Checkpoints will be saved to Google Drive:")
        print(f"   {checkpoint_dir}")
    else:
        checkpoint_dir = "/content/lora_checkpoint"  # Colab local storage
        print(f"\n💾 Checkpoints will be saved locally to Colab:")
        print(f"   {checkpoint_dir}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def print_device_info():
    """Display GPU information"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"\n🔧 Device Info:")
        print(f"   - Device: GPU (CUDA)")
        print(f"   - GPU Name: {device_name}")
        print(f"   - GPU Count: {device_count}")
        print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   - CUDA Version: {torch.version.cuda}")
    else:
        print(f"\n🔧 Device Info:")
        print(f"   - Device: CPU")
        print(f"   - WARNING: Training on CPU will be extremely slow!")
    print()


def format_training_prompt(question: str, answer: str) -> str:
    """
    Format question-answer pairs into training prompt
    Uses Phi-3 instruction format
    """
    prompt = f"""<|user|>
{question}<|end|>
<|assistant|>
{answer}<|end_of_turn|>"""
    return prompt


def create_training_dataset(
    datasets_dict: Dict[str, Dict[str, List]],
    tokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Create training dataset from all datasets combined
    Combines GSM8K, CommonsenseQA, and SVAMP into single dataset
    """
    print("\n" + "="*80)
    print("CREATING TRAINING DATASET")
    print("="*80)
    
    all_texts = []
    total_samples = 0
    
    for dataset_name, dataset_content in datasets_dict.items():
        questions = dataset_content["questions_list"]
        answers = dataset_content["ground_truth_answers"]
        
        print(f"\n📚 Processing {dataset_name.upper()}:")
        print(f"   - {len(questions)} samples")
        
        for question, answer in zip(questions, answers):
            # Format the prompt-answer pair
            text = format_training_prompt(question, str(answer))
            all_texts.append(text)
            total_samples += 1
        
        print(f"   ✅ Added {len(questions)} samples to training data")
    
    print(f"\n✅ Total training samples: {total_samples}")
    print(f"   (Combined from all datasets)")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({"text": all_texts})
    
    print(f"\n✅ Dataset created: {len(dataset)} samples")
    
    return dataset


def get_quantization_config():
    """Create 4-bit quantization config"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )


def train_lora_model():
    """
    Main training function for LoRA fine-tuning in Colab
    """
    print("\n" + "="*80)
    print(" "*10 + "🚀 LoRA Fine-Tuning Training Script (Colab Edition)")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 0: Mount Google Drive
    print("\n" + "#"*80)
    print("# STEP 0: SETTING UP GOOGLE DRIVE")
    print("#"*80)
    use_drive = mount_google_drive()
    checkpoint_dir = get_checkpoint_dir(use_drive)
    
    try:
        # ====================================================================
        # STEP 1: Load Datasets
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 1: LOADING DATASETS FOR TRAINING")
        print("="*80)
        
        # Load 100 samples per dataset for comprehensive training
        print("\n📥 Loading datasets (100 samples each)...")
        datasets_dict = load_all_datasets(num_samples=100)
        
        total_samples = 0
        for dataset_name in datasets_dict.keys():
            num_questions = len(datasets_dict[dataset_name]["questions_list"])
            total_samples += num_questions
            print(f"✅ {dataset_name.upper()}: {num_questions} samples")
        
        print(f"\n📊 Total training samples: {total_samples}")
        print()
        
        # ====================================================================
        # STEP 2: Load Base Model with 4-bit Quantization
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 2: LOADING BASE MODEL WITH 4-BIT QUANTIZATION")
        print("="*80)
        
        quantization_config = get_quantization_config()
        print(f"\n🔍 Loading model: {BASE_MODEL_NAME}")
        print(f"📦 Using 4-bit quantization for memory efficiency...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"\n✅ Model loaded successfully")
        print(f"✅ Model dtype: {base_model.dtype}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded")
        
        print_device_info()
        
        # ====================================================================
        # STEP 3: Prepare Model for LoRA
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 3: PREPARING MODEL FOR LORA TRAINING")
        print("="*80)
        
        print("\n🔧 Preparing model for kbit training...")
        base_model = prepare_model_for_kbit_training(base_model)
        print("✅ Model prepared")
        
        # Create LoRA config
        print(f"\n🔧 Creating LoRA configuration...")
        print(f"   - Rank (r): {LORA_CONFIG['r']}")
        print(f"   - Alpha: {LORA_CONFIG['lora_alpha']}")
        print(f"   - Target modules: {LORA_CONFIG['target_modules']}")
        print(f"   - Dropout: {LORA_CONFIG['lora_dropout']}")
        
        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            target_modules=LORA_CONFIG["target_modules"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            bias=LORA_CONFIG["bias"],
            task_type=LORA_CONFIG["task_type"]
        )
        
        # Apply LoRA to model
        print(f"\n🔧 Applying LoRA to model...")
        lora_model = get_peft_model(base_model, lora_config)
        lora_model.print_trainable_parameters()
        print("✅ LoRA applied successfully")
        
        # ====================================================================
        # STEP 4: Create Training Dataset
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 4: PREPARING TRAINING DATA")
        print("="*80)
        
        train_dataset = create_training_dataset(datasets_dict, tokenizer)
        
        # ====================================================================
        # STEP 5: Configure Training Arguments
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 5: CONFIGURING TRAINING PARAMETERS")
        print("="*80)
        
        # Create log directory
        os.makedirs(os.path.join(checkpoint_dir, "logs"), exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            overwrite_output_dir=False,
            
            # Training parameters
            num_train_epochs=3,                    # 3 epochs
            per_device_train_batch_size=4,         # Larger batch for T4 GPU
            gradient_accumulation_steps=2,         # Simulate batch_size=8
            
            # Optimization parameters
            learning_rate=2e-4,                    # Lower LR for fine-tuning
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            
            # Memory optimization
            optim="paged_adamw_8bit",             # Memory-efficient optimizer
            max_grad_norm=0.3,
            
            # Logging and evaluation
            logging_dir=os.path.join(checkpoint_dir, "logs"),
            logging_steps=10,
            logging_strategy="steps",
            
            # Save strategy
            save_strategy="epoch",
            save_total_limit=3,
            
            # Performance
            fp16=True,                             # Mixed precision training
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            
            # Other
            seed=42,
            disable_tqdm=False,
            report_to=[]
        )
        
        print(f"\n🎯 Training Configuration:")
        print(f"   - Epochs: {training_args.num_train_epochs}")
        print(f"   - Batch size: {training_args.per_device_train_batch_size}")
        print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"   - Learning rate: {training_args.learning_rate}")
        print(f"   - FP16 mixed precision: {training_args.fp16}")
        print(f"   - Optimizer: {training_args.optim}")
        print(f"   - Checkpoint dir: {checkpoint_dir}")
        
        # ====================================================================
        # STEP 6: Create Data Collator and Trainer
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 6: INITIALIZING TRAINER")
        print("="*80)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not MLM
        )
        
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        print(f"\n✅ Trainer initialized")
        print(f"✅ Ready to start training...")
        
        # ====================================================================
        # STEP 7: Start Training
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 7: STARTING TRAINING")
        print("="*80)
        print(f"\n⏱️  Estimated training time: 15-30 minutes on Colab T4")
        print(f"📊 Checkpoint will be saved every epoch")
        print(f"💾 Saves will go to: {checkpoint_dir}")
        print(f"\n🚀 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Start training
        train_result = trainer.train()
        
        # ====================================================================
        # STEP 8: Save Trained Model
        # ====================================================================
        print("\n" + "="*80)
        print("STEP 8: SAVING TRAINED MODEL")
        print("="*80)
        
        print(f"\n💾 Saving trained LoRA weights...")
        lora_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"✅ Model saved to: {checkpoint_dir}")
        print(f"\n📁 Saved files:")
        print(f"   - adapter_model.bin (LoRA weights)")
        print(f"   - adapter_config.json (LoRA configuration)")
        print(f"   - training_args.bin (Training arguments)")
        
        # ====================================================================
        # Final Summary
        # ====================================================================
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n📊 Training Results:")
        print(f"   - Final Loss: {train_result.training_loss:.4f}")
        print(f"\n🎯 Next Steps:")
        
        if use_drive:
            print(f"\n   ✅ Your LoRA model is saved in Google Drive:")
            print(f"      /MyDrive/lora_checkpoint/")
            print(f"\n   1. Update config.py with the Drive path:")
            print(f"      LORA_CHECKPOINT_DIR = '/content/drive/MyDrive/lora_checkpoint'")
        else:
            print(f"\n   ✅ Your LoRA model is saved in Colab storage:")
            print(f"      /content/lora_checkpoint/")
            print(f"\n   1. Or download it to your computer")
        
        print(f"\n   2. Run: python colab_evaluation.py")
        print(f"\n✨ Your LoRA model is now trained and ready for evaluation!")
        print("="*80)
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# LoRA FINE-TUNING TRAINING SCRIPT (COLAB OPTIMIZED)")
    print("# This will train LoRA adapters and save to Google Drive")
    print("#"*80 + "\n")
    
    exit_code = train_lora_model()
    sys.exit(exit_code)
