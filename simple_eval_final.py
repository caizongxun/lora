#!/usr/bin/env python3
"""
Ultra-Simple LoRA Evaluation Script - FINAL WORKING VERSION
Using device_map='cuda' for 4-bit quantized models

Usage:
    python simple_eval_final.py
"""

import sys
import traceback
import os
from datetime import datetime
import torch

# Setup log file
log_file = '/content/eval_debug.log'
log_output = []

def log_print(msg=""):
    """Print to both console and log file"""
    print(msg)
    log_output.append(msg)

def save_log():
    """Save all output to log file"""
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_output))
    print(f"\n\n📋 Log saved to: {log_file}")

log_print("\n" + "="*80)
log_print("🚀 Simple LoRA Evaluation - FINAL VERSION (device_map='cuda')")
log_print("="*80)
log_print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Python version: {sys.version}")
log_print(f"Working directory: {os.getcwd()}")

# Test each import
log_print("\n[STEP 0] Testing imports...\n")

# Test 1: torch
log_print("[TEST 1] Importing torch...")
try:
    log_print(f"✅ Success! torch version: {torch.__version__}")
    log_print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_print(f"   GPU: {torch.cuda.get_device_name(0)}")
        log_print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except Exception as e:
    log_print(f"❌ FAILED: {type(e).__name__}: {e}")
    log_print(traceback.format_exc())
    save_log()
    sys.exit(1)

# Test 2: transformers
log_print("\n[TEST 2] Importing transformers...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import transformers
    log_print(f"✅ Success! transformers version: {transformers.__version__}")
except Exception as e:
    log_print(f"❌ FAILED: {type(e).__name__}: {e}")
    log_print(traceback.format_exc())
    save_log()
    sys.exit(1)

# Test 3: peft
log_print("\n[TEST 3] Importing peft...")
try:
    from peft import PeftModel
    import peft
    log_print(f"✅ Success! peft version: {peft.__version__}")
except Exception as e:
    log_print(f"❌ FAILED: {type(e).__name__}: {e}")
    log_print(traceback.format_exc())
    save_log()
    sys.exit(1)

log_print("\n" + "="*80)
log_print("All imports successful! Proceeding with model loading...")
log_print("="*80)

def main():
    # Configuration
    base_model = "microsoft/Phi-3-mini-4k-instruct"
    lora_model_id = "zongowo111/phi3-lora-gsm8k-commonsense"
    
    try:
        # Step 1: Setup quantization
        log_print("\n[STEP 1/5] Setting up 4-bit quantization...")
        log_print("   - load_in_4bit: True")
        log_print("   - bnb_4bit_compute_dtype: torch.float16")
        log_print("   - bnb_4bit_use_double_quant: True")
        log_print("   - bnb_4bit_quant_type: nf4")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        log_print("✅ Quantization config created successfully")
        
        # Step 2: Load base model
        # FIX: Use device_map="cuda" (not "cuda:0" or "auto")
        log_print("\n[STEP 2/5] Loading base model...")
        log_print(f"   Model: {base_model}")
        log_print(f"   device_map: 'cuda'  <-- Correct for 4-bit quantization")
        log_print(f"   Trust remote code: True")
        log_print(f"   (This may take 2-3 minutes on first run)")
        
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="cuda",  # <-- CORRECT: Use 'cuda' not 'cuda:0' or 'auto'
            trust_remote_code=True
        )
        log_print("✅ Base model loaded successfully")
        log_print(f"   Model type: {type(base_model_obj).__name__}")
        log_print(f"   Model dtype: {base_model_obj.dtype}")
        
        # Step 3: Load tokenizer
        log_print("\n[STEP 3/5] Loading tokenizer...")
        log_print(f"   Model: {base_model}")
        log_print(f"   Trust remote code: True")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        log_print("✅ Tokenizer loaded successfully")
        log_print(f"   Tokenizer type: {type(tokenizer).__name__}")
        log_print(f"   Vocab size: {len(tokenizer)}")
        
        # Step 4: Load LoRA weights
        log_print("\n[STEP 4/5] Loading LoRA weights...")
        log_print(f"   LoRA model ID: {lora_model_id}")
        log_print(f"   Is trainable: False")
        log_print(f"   (Downloading from HuggingFace Hub)")
        
        lora_model = PeftModel.from_pretrained(
            base_model_obj,
            lora_model_id,
            is_trainable=False
        )
        log_print("✅ LoRA weights loaded successfully")
        log_print(f"   Model type: {type(lora_model).__name__}")
        
        # Step 5: Test inference
        log_print("\n[STEP 5/5] Testing inference...")
        test_prompt = "What is 2+2?"
        prompt = f"<|user|>\n{test_prompt}<|end|>\n<|assistant|>\n"
        
        log_print(f"   Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        log_print(f"   ✅ Input IDs shape: {inputs['input_ids'].shape}")
        
        # Move inputs to GPU
        log_print(f"   Moving inputs to GPU...")
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        log_print(f"   ✅ Inputs moved to CUDA")
        
        log_print(f"   Running model.generate()...")
        with torch.no_grad():
            outputs = lora_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=128,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        log_print(f"   ✅ Output generated: shape {outputs.shape}")
        
        log_print(f"   Decoding output...")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log_print(f"   ✅ Response decoded")
        
        log_print("\n" + "="*80)
        log_print("🎉 INFERENCE RESULT")
        log_print("="*80)
        log_print(f"\nPrompt: {test_prompt}")
        log_print(f"\nResponse:\n{response[len(prompt):].strip()}")
        log_print("\n" + "="*80)
        log_print("✅ Evaluation completed successfully!")
        log_print("="*80)
        
        save_log()
        return 0
        
    except KeyboardInterrupt:
        log_print("\n⚠️  Interrupted by user")
        save_log()
        return 1
        
    except Exception as e:
        log_print(f"\n" + "="*80)
        log_print("❌ ERROR OCCURRED")
        log_print("="*80)
        log_print(f"\nError type: {type(e).__name__}")
        log_print(f"Error message: {e}")
        log_print(f"\nFull traceback:")
        log_print("-"*80)
        log_print(traceback.format_exc())
        log_print("-"*80)
        
        save_log()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
