#!/usr/bin/env python3
"""
Ultra-Simple LoRA Evaluation Script
Minimal dependencies, maximum reliability

Usage:
    python simple_eval.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

def main():
    print("\n" + "="*80)
    print("🚀 Simple LoRA Evaluation")
    print("="*80)
    
    # Configuration
    base_model = "microsoft/Phi-3-mini-4k-instruct"
    lora_model_id = "zongowo111/phi3-lora-gsm8k-commonsense"
    num_samples = 3
    
    try:
        # Step 1: Setup quantization
        print("\n[1/5] Setting up 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("✅ Done")
        
        # Step 2: Load base model
        print("\n[2/5] Loading base model...")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Done")
        
        # Step 3: Load tokenizer
        print("\n[3/5] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Done")
        
        # Step 4: Load LoRA weights
        print("\n[4/5] Loading LoRA weights...")
        lora_model = PeftModel.from_pretrained(
            base_model_obj,
            lora_model_id,
            is_trainable=False
        )
        print("✅ Done")
        
        # Step 5: Test inference
        print("\n[5/5] Testing inference...")
        test_prompt = "What is 2+2?"
        prompt = f"<|user|>\n{test_prompt}<|end|>\n<|assistant|>\n"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = lora_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=128,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✅ Done")
        
        print("\n" + "="*80)
        print("🌟 INFERENCE RESULT")
        print("="*80)
        print(f"\nPrompt: {test_prompt}")
        print(f"\nResponse: {response[len(prompt):].strip()}")
        print("\n" + "="*80)
        print("✅ Simple evaluation completed successfully!")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
