#!/usr/bin/env python3
"""
Ultra-Simple LoRA Evaluation Script - VERBOSE ERROR PRINTING
Minimal dependencies, maximum error visibility

Usage:
    python simple_eval_verbose.py
"""

import sys
import traceback

print("\n" + "="*80)
print("🚀 Simple LoRA Evaluation - VERBOSE MODE")
print("="*80)

# Test each import separately with detailed error messages
print("\n[STEP 0] Testing imports...\n")

# Test 1: torch
print("[TEST 1] Importing torch...")
try:
    import torch
    print(f"✅ Success! torch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: transformers
print("\n[TEST 2] Importing transformers...")
try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"✅ Success! transformers version: {transformers.__version__}")
    print(f"   AutoModelForCausalLM: {AutoModelForCausalLM}")
    print(f"   AutoTokenizer: {AutoTokenizer}")
    print(f"   BitsAndBytesConfig: {BitsAndBytesConfig}")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: peft
print("\n[TEST 3] Importing peft...")
try:
    import peft
    from peft import PeftModel
    print(f"✅ Success! peft version: {peft.__version__}")
    print(f"   PeftModel: {PeftModel}")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: datasets
print("\n[TEST 4] Importing datasets...")
try:
    import datasets
    from datasets import load_dataset
    print(f"✅ Success! datasets version: {datasets.__version__}")
    print(f"   load_dataset: {load_dataset}")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("All imports successful! Proceeding with model loading...")
print("="*80)

def main():
    # Configuration
    base_model = "microsoft/Phi-3-mini-4k-instruct"
    lora_model_id = "zongowo111/phi3-lora-gsm8k-commonsense"
    num_samples = 3
    
    try:
        # Step 1: Setup quantization
        print("\n[STEP 1/5] Setting up 4-bit quantization...")
        print(f"   Creating BitsAndBytesConfig with:")
        print(f"   - load_in_4bit: True")
        print(f"   - bnb_4bit_compute_dtype: torch.float16")
        print(f"   - bnb_4bit_use_double_quant: True")
        print(f"   - bnb_4bit_quant_type: nf4")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("✅ Quantization config created successfully")
        
        # Step 2: Load base model
        print("\n[STEP 2/5] Loading base model...")
        print(f"   Model: {base_model}")
        print(f"   Device map: auto")
        print(f"   Trust remote code: True")
        print(f"   (This may take 2-3 minutes on first run)")
        
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Base model loaded successfully")
        print(f"   Model type: {type(base_model_obj).__name__}")
        print(f"   Model device: {next(base_model_obj.parameters()).device}")
        
        # Step 3: Load tokenizer
        print("\n[STEP 3/5] Loading tokenizer...")
        print(f"   Model: {base_model}")
        print(f"   Trust remote code: True")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
        print(f"   Tokenizer type: {type(tokenizer).__name__}")
        print(f"   Vocab size: {len(tokenizer)}")
        
        # Step 4: Load LoRA weights
        print("\n[STEP 4/5] Loading LoRA weights...")
        print(f"   LoRA model ID: {lora_model_id}")
        print(f"   Is trainable: False")
        print(f"   (Downloading from HuggingFace Hub)")
        
        lora_model = PeftModel.from_pretrained(
            base_model_obj,
            lora_model_id,
            is_trainable=False
        )
        print("✅ LoRA weights loaded successfully")
        print(f"   Model type: {type(lora_model).__name__}")
        
        # Step 5: Test inference
        print("\n[STEP 5/5] Testing inference...")
        test_prompt = "What is 2+2?"
        prompt = f"<|user|>\n{test_prompt}<|end|>\n<|assistant|>\n"
        print(f"   Test prompt: {test_prompt}")
        print(f"   Full prompt: {repr(prompt)}")
        
        print("\n   Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        print(f"   ✅ Input IDs shape: {inputs['input_ids'].shape}")
        print(f"   ✅ Input device: {inputs['input_ids'].device}")
        
        print("\n   Running model.generate()...")
        with torch.no_grad():
            outputs = lora_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=128,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"   ✅ Output shape: {outputs.shape}")
        
        print("\n   Decoding output...")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   ✅ Response length: {len(response)} characters")
        
        print("\n" + "="*80)
        print("🌟 INFERENCE RESULT")
        print("="*80)
        print(f"\nPrompt: {test_prompt}")
        print(f"\nResponse: {response[len(prompt):].strip()}")
        print("\n" + "="*80)
        print("✅ Simple evaluation completed successfully!")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n" + "="*80)
        print("❌ ERROR OCCURRED")
        print("="*80)
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {e}")
        print(f"\nFull traceback:")
        print("\n" + "-"*80)
        traceback.print_exc()
        print("-"*80)
        print("\n" + "="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
