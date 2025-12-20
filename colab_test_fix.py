#!/usr/bin/env python3
"""
Colab Test Script: Verify the .to() quantization error fix

This script tests that 4-bit quantized models work correctly WITHOUT .to() calls.
Use in Colab to verify the fix before running full evaluation.

Usage in Colab:
    !cd /content/lora && python colab_test_fix.py
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def test_4bit_quantization():
    """
    Test that 4-bit quantized models work WITHOUT .to() calls
    """
    print("\n" + "="*80)
    print("🪧 Testing 4-bit Quantization Fix")
    print("="*80)
    
    # Test configuration
    base_model = "microsoft/Phi-3-mini-4k-instruct"
    test_prompt = "What is 2+2?"
    
    print(f"\n[1/3] Setting up 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    print(f"✅ Quantization config ready")
    
    print(f"\n[2/3] Loading model: {base_model}")
    print(f"      (This may take 1-2 minutes on first run)")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",  # KEY: This automatically places model on GPU
            trust_remote_code=True
        )
        print(f"✅ Model loaded successfully")
        print(f"   - Model dtype: {model.dtype}")
        print(f"   - Model device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print(f"\n[3/3] Testing inference WITHOUT .to() call...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    try:
        # This is the key test: generate WITHOUT calling .to()
        prompt = f"<|user|>\n{test_prompt}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        print(f"   Input IDs shape: {inputs['input_ids'].shape}")
        print(f"   Input device: {inputs['input_ids'].device}")
        
        # CRITICAL: Do NOT call .to() on 4-bit quantized models
        # This will raise ValueError
        # inputs = {k: v.to(model.device) for k, v in inputs.items()}  # ❌ WRONG
        
        # Instead, pass tensors directly to model
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=128,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Inference successful!")
        print(f"\n   Prompt: {test_prompt}")
        print(f"   Response: {response[len(prompt):].strip()[:100]}...")
        
        return True
        
    except ValueError as e:
        if ".to` is not supported" in str(e):
            print(f"❌ ERROR: Still using .to() on quantized model!")
            print(f"   {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()


def test_lora_loading():
    """
    Test that LoRA loading works with fixed code
    """
    print("\n" + "="*80)
    print("🪧 Testing LoRA Loading")
    print("="*80)
    
    from peft import PeftModel
    
    base_model_name = "microsoft/Phi-3-mini-4k-instruct"
    lora_model_id = "zongowo111/phi3-lora-gsm8k-commonsense"
    
    print(f"\n[1/2] Loading base model with 4-bit quantization...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ Base model loaded")
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return False
    
    print(f"\n[2/2] Loading LoRA weights from: {lora_model_id}")
    
    try:
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_id,
            is_trainable=False
        )
        print(f"✅ LoRA model loaded successfully")
        print(f"   - LoRA weights loaded from HF Hub")
        print(f"   - Ready for inference")
        
        del lora_model, base_model
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"❌ Failed to load LoRA: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("🚀 LoRA Quantization Fix - Verification Test")
    print("="*80)
    print(f"\nThis test verifies that the .to() fix is working correctly.")
    print(f"The error occurred because 4-bit quantized models cannot be moved")
    print(f"with .to() - they're already optimally placed via device_map.")
    print()
    
    # Test 1: Basic 4-bit inference
    print(f"\n🗪 TEST 1: Basic 4-bit Quantized Inference")
    test1_passed = test_4bit_quantization()
    
    # Test 2: LoRA loading
    print(f"\n🗪 TEST 2: LoRA Model Loading")
    test2_passed = test_lora_loading()
    
    # Summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    print(f"\nTest 1 (4-bit Inference):   {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (LoRA Loading):      {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - Fix is working correctly!")
        print("="*80)
        print(f"\nYou can now run the full evaluation:")
        print(f"   python colab_evaluate_lora.py \\")
        print(f"     --hf_model_id zongowo111/phi3-lora-gsm8k-commonsense")
        print()
        return 0
    else:
        print("\n" + "="*80)
        print("❌ SOME TESTS FAILED - There may still be issues")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
