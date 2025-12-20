"""
Model evaluator module for comparing baseline and LoRA-tuned models
Implements inference and performance measurement
Uses half precision (fp16) with automatic memory management
"""

import re
import time
import torch
import gc
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from config import GENERATION_CONFIG, DEVICE, TIMEOUT_SECONDS


def print_device_info():
    """Display current device information (GPU/CPU)"""
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
        print(f"   - WARNING: Using CPU will be very slow for model inference!")
    print()


def extract_answer(response: str) -> str:
    """Extract numerical answer or option letter (A-E) from model response text"""
    if not response or not response.strip():
        return ""
    
    # PRIORITY 1: Look for explicit option format patterns first
    explicit_option_patterns = [
        r"^\s*([A-E])\.\s",
        r"answer[\s:]*([A-E])",
        r"option[\s:]*([A-E])",
        r"correct answer[\s:]*([A-E])",
    ]
    
    for pattern in explicit_option_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    # PRIORITY 2: Extract ALL numbers and take the LAST one
    all_numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', response)
    if all_numbers:
        last_number = all_numbers[-1].replace(',', '')
        try:
            if '.' not in last_number:
                return str(int(last_number))
            else:
                return last_number
        except ValueError:
            return all_numbers[-1]
    
    # PRIORITY 3: Look for fallback option patterns
    fallback_patterns = [
        r"[\b\(]([A-E])[\b\)]",
        r"([A-E])\.",
    ]
    
    for pattern in fallback_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()
    
    # PRIORITY 4: As last resort
    single_letter = re.search(r'[A-E]', response, re.IGNORECASE)
    if single_letter:
        return single_letter.group(0).upper()
    
    return ""


def evaluate_baseline_model(
    model_name: str,
    datasets_dict: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate baseline model (untuned) on all datasets
    """
    print(f"Loading baseline model: {model_name}")
    print(f"Using half precision (fp16) with automatic memory management...")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # 🔧 Use fp16 (half precision) to save 50% memory
    print(f"🔍 Debug: Loading with device_map='auto', torch_dtype=float16...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto",  # Automatic GPU/CPU memory management
        torch_dtype=torch.float16,  # 🔧 CHANGED: Use half precision (fp16) instead of float32
        trust_remote_code=True
    )
    
    print(f"✅ Model dtype after loading: {base_model.dtype}")
    
    tokenizer_base = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_base.pad_token = tokenizer_base.eos_token
    print(f"✅ Baseline model loaded with device_map='auto' (half precision fp16)")
    
    print_device_info()
    
    baseline_results = {}
    
    for dataset_name, dataset_content in datasets_dict.items():
        questions_list = dataset_content["questions_list"]
        ground_truth_answers = dataset_content["ground_truth_answers"]
        
        predictions = []
        correct_count = 0
        start_time = time.time()
        
        print(f"Evaluating baseline on {dataset_name}...")
        
        for idx, question in enumerate(questions_list):
            try:
                prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
                inputs = tokenizer_base(prompt, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(DEVICE)
                attention_mask = inputs["attention_mask"].to(DEVICE)
                
                with torch.no_grad():
                    outputs = base_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=256,  # Use max_new_tokens instead of max_length
                        do_sample=False,
                        pad_token_id=tokenizer_base.eos_token_id
                    )
                
                generated_tokens = outputs[0][len(input_ids[0]):]
                response = tokenizer_base.decode(generated_tokens, skip_special_tokens=True)
                
                extracted_answer = extract_answer(response)
                predictions.append(extracted_answer)
                
                if idx < 3:
                    print(f"\n{'='*80}")
                    print(f"[BASELINE] Debug Sample {idx} - {dataset_name}")
                    print(f"{'='*80}")
                    print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")
                    print(f"\nModel Response: {response[:500]}..." if len(response) > 500 else f"\nModel Response: {response}")
                    print(f"\nExtracted Answer: '{extracted_answer}'")
                    print(f"Ground Truth: '{ground_truth_answers[idx]}'")
                    print(f"Match: {'✅ YES' if extracted_answer == ground_truth_answers[idx] else '❌ NO'}")
                    print(f"{'='*80}\n")
                
                is_correct = (extracted_answer == ground_truth_answers[idx])
                if is_correct:
                    correct_count += 1
                
                del input_ids, attention_mask, outputs
                torch.cuda.empty_cache()
                gc.collect()
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                predictions.append("")
                torch.cuda.empty_cache()
                gc.collect()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        total_count = len(questions_list)
        baseline_accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        baseline_results[dataset_name] = {
            "accuracy": baseline_accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "predictions": predictions,
            "inference_time": elapsed_time
        }
        
        print(f"Baseline {dataset_name} Accuracy: {baseline_accuracy:.2%} ({correct_count}/{total_count})")
    
    print("Cleaning up baseline model from memory...")
    del base_model, tokenizer_base
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleaned.")
    
    return baseline_results


def evaluate_lora_model(
    model_name: str,
    lora_config_dict: Dict[str, Any],
    datasets_dict: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate LoRA-tuned model on all datasets
    """
    print(f"Loading base model for LoRA: {model_name}")
    print(f"Using half precision (fp16) with automatic memory management...")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # 🔧 Use fp16 (half precision) to save 50% memory
    print(f"🔍 Debug: Loading with device_map='auto', torch_dtype=float16...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto",  # Automatic GPU/CPU memory management
        torch_dtype=torch.float16,  # 🔧 CHANGED: Use half precision (fp16) instead of float32
        trust_remote_code=True
    )
    
    print(f"✅ Model dtype after loading: {base_model.dtype}")
    
    tokenizer_base = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_base.pad_token = tokenizer_base.eos_token
    print(f"✅ LoRA base model loaded with device_map='auto' (half precision fp16)")

    print("Preparing model for training...")
    # Note: prepare_model_for_kbit_training is for quantized models
    # For fp16, we skip this step
    print("✅ Model preparation complete.")
    
    print_device_info()
    
    print(f"Applying LoRA configuration...")
    
    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias=lora_config_dict["bias"],
        task_type=lora_config_dict["task_type"]
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    print("\n✅ LoRA model successfully created!\n")
    
    lora_results = {}
    
    for dataset_name, dataset_content in datasets_dict.items():
        questions_list = dataset_content["questions_list"]
        ground_truth_answers = dataset_content["ground_truth_answers"]
        
        predictions = []
        correct_count = 0
        start_time = time.time()
        
        print(f"Evaluating LoRA model on {dataset_name}...")
        
        for idx, question in enumerate(questions_list):
            try:
                prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
                inputs = tokenizer_base(prompt, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(DEVICE)
                attention_mask = inputs["attention_mask"].to(DEVICE)
                
                with torch.no_grad():
                    outputs = lora_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=256,  # Use max_new_tokens instead of max_length
                        do_sample=False,
                        pad_token_id=tokenizer_base.eos_token_id
                    )
                
                generated_tokens = outputs[0][len(input_ids[0]):]
                response = tokenizer_base.decode(generated_tokens, skip_special_tokens=True)
                
                extracted_answer = extract_answer(response)
                predictions.append(extracted_answer)
                
                if idx < 3:
                    print(f"\n{'='*80}")
                    print(f"[LoRA] Debug Sample {idx} - {dataset_name}")
                    print(f"{'='*80}")
                    print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")
                    print(f"\nModel Response: {response[:500]}..." if len(response) > 500 else f"\nModel Response: {response}")
                    print(f"\nExtracted Answer: '{extracted_answer}'")
                    print(f"Ground Truth: '{ground_truth_answers[idx]}'")
                    print(f"Match: {'✅ YES' if extracted_answer == ground_truth_answers[idx] else '❌ NO'}")
                    print(f"{'='*80}\n")
                
                is_correct = (extracted_answer == ground_truth_answers[idx])
                if is_correct:
                    correct_count += 1
                
                del input_ids, attention_mask, outputs
                torch.cuda.empty_cache()
                gc.collect()
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                predictions.append("")
                torch.cuda.empty_cache()
                gc.collect()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        total_count = len(questions_list)
        lora_accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        lora_results[dataset_name] = {
            "accuracy": lora_accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "predictions": predictions,
            "inference_time": elapsed_time
        }
        
        print(f"LoRA {dataset_name} Accuracy: {lora_accuracy:.2%} ({correct_count}/{total_count})\n")
    
    print("Cleaning up LoRA model from memory...")
    del lora_model, base_model, tokenizer_base
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleaned.")

    return lora_results
