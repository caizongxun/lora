"""
Model evaluator module for comparing baseline and LoRA-tuned models
Implements inference and performance measurement
Uses 4-bit quantization for minimal GPU memory usage
Loads trained LoRA weights from HuggingFace Hub or local checkpoint

KEY UPDATE: Now includes evaluate_lora_model_with_checkpoint() function
            which downloads trained LoRA weights from Hugging Face Hub
"""

import re
import time
import torch
import gc
import os
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from config import GENERATION_CONFIG, DEVICE, TIMEOUT_SECONDS, LORA_CHECKPOINT_DIR


def print_device_info():
    """Display current device information (GPU/CPU)"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"\n[INFO] Device Info:")
        print(f"       - Device: GPU (CUDA)")
        print(f"       - GPU Name: {device_name}")
        print(f"       - GPU Count: {device_count}")
        print(f"       - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"       - CUDA Version: {torch.version.cuda}")
    else:
        print(f"\n[INFO] Device Info:")
        print(f"       - Device: CPU")
        print(f"       - WARNING: Using CPU will be very slow for model inference!")
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


def get_quantization_config():
    """Create 4-bit quantization config for aggressive memory reduction"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )


def evaluate_baseline_model(
    model_name: str,
    datasets_dict: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate baseline model (untuned) on all datasets
    Uses 4-bit quantization for minimal GPU memory usage
    """
    print(f"Loading baseline model: {model_name}")
    print(f"Using 4-bit quantization...")

    # Use 4-bit quantization for maximum memory efficiency
    quantization_config = get_quantization_config()
    print(f"[DEBUG] Loading with 4-bit quantization, device_map='cuda:0'...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda:0",
        trust_remote_code=True
    )
    
    print(f"[SUCCESS] Model loaded with 4-bit quantization")
    print(f"[INFO] Model dtype: {base_model.dtype}")
    
    tokenizer_base = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_base.pad_token = tokenizer_base.eos_token
    print(f"[SUCCESS] Baseline model ready for inference")
    
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
                        max_new_tokens=256,
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
                    print(f"Match: {'YES' if extracted_answer == ground_truth_answers[idx] else 'NO'}")
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


def evaluate_lora_model_with_checkpoint(
    model_name: str,
    hf_model_id: str,
    datasets_dict: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Any]]:
    """
    NEW FUNCTION: Evaluate LoRA-tuned model on all datasets
    Downloads trained LoRA weights from Hugging Face Hub
    Uses 4-bit quantization for minimal GPU memory usage
    
    Args:
        model_name: Base model name (e.g., "microsoft/Phi-3-mini-4k-instruct")
        hf_model_id: HuggingFace model ID with trained LoRA (e.g., "zongowo111/phi3-lora-gsm8k-commonsense")
        datasets_dict: Dictionary containing all datasets for evaluation
    """
    print(f"Loading base model for LoRA: {model_name}")
    print(f"Using 4-bit quantization...")

    # Use 4-bit quantization for maximum memory efficiency
    quantization_config = get_quantization_config()
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda:0",
        trust_remote_code=True
    )
    
    print(f"[SUCCESS] Model loaded with 4-bit quantization")
    print(f"[INFO] Model dtype: {base_model.dtype}")
    
    tokenizer_base = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_base.pad_token = tokenizer_base.eos_token
    print(f"[SUCCESS] LoRA base model ready")

    print("\nPreparing model for LoRA...")
    base_model = prepare_model_for_kbit_training(base_model)
    print("[SUCCESS] Model preparation complete.")
    
    print_device_info()
    
    # CRITICAL: Load trained LoRA weights from HuggingFace Hub
    print(f"\n[INFO] Downloading trained LoRA weights from Hugging Face Hub...")
    print(f"       Repository: {hf_model_id}")
    print(f"       (This may take a few seconds on first run)")
    
    try:
        # Load the trained LoRA weights from HF Hub
        lora_model = PeftModel.from_pretrained(
            base_model,
            hf_model_id,
            is_trainable=False  # Set to False for inference only
        )
        print(f"\n[SUCCESS] Successfully downloaded and loaded LoRA weights from HF Hub!")
        print(f"[SUCCESS] Using TRAINED LoRA adapters (not random initialization)")
        print(f"[SUCCESS] Model is now ready for evaluation")
        
    except Exception as e:
        print(f"\n[ERROR] Could not download LoRA weights from HF Hub: {e}")
        print(f"\n[INFO] Please ensure:")
        print(f"       1. Internet connection is working")
        print(f"       2. Repository exists: https://huggingface.co/{hf_model_id}")
        print(f"       3. Repository is public or you have access token")
        raise
    
    print("\n[SUCCESS] LoRA model successfully loaded!\n")
    
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
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer_base.eos_token_id
                    )
                
                generated_tokens = outputs[0][len(input_ids[0]):]
                response = tokenizer_base.decode(generated_tokens, skip_special_tokens=True)
                
                extracted_answer = extract_answer(response)
                predictions.append(extracted_answer)
                
                if idx < 3:
                    print(f"\n{'='*80}")
                    print(f"[LORA] Debug Sample {idx} - {dataset_name}")
                    print(f"{'='*80}")
                    print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")
                    print(f"\nModel Response: {response[:500]}..." if len(response) > 500 else f"\nModel Response: {response}")
                    print(f"\nExtracted Answer: '{extracted_answer}'")
                    print(f"Ground Truth: '{ground_truth_answers[idx]}'")
                    print(f"Match: {'YES' if extracted_answer == ground_truth_answers[idx] else 'NO'}")
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


def evaluate_lora_model(
    model_name: str,
    lora_config_dict: Dict[str, Any],
    datasets_dict: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate LoRA-tuned model on all datasets
    Uses 4-bit quantization for minimal GPU memory usage
    LOADS TRAINED LoRA WEIGHTS FROM LOCAL CHECKPOINT
    
    DEPRECATED: Use evaluate_lora_model_with_checkpoint() instead
    This function is kept for backward compatibility only
    """
    print(f"Loading base model for LoRA: {model_name}")
    print(f"Using 4-bit quantization...")

    # Use 4-bit quantization for maximum memory efficiency
    quantization_config = get_quantization_config()
    print(f"[DEBUG] Loading with 4-bit quantization, device_map='cuda:0'...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda:0",
        trust_remote_code=True
    )
    
    print(f"[SUCCESS] Model loaded with 4-bit quantization")
    print(f"[INFO] Model dtype: {base_model.dtype}")
    
    tokenizer_base = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_base.pad_token = tokenizer_base.eos_token
    print(f"[SUCCESS] LoRA base model ready for configuration")

    print("Preparing model for LoRA training...")
    base_model = prepare_model_for_kbit_training(base_model)
    print("[SUCCESS] Model preparation complete.")
    
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
    
    # CRITICAL: Load trained LoRA weights from local checkpoint
    print(f"\n[INFO] Attempting to load trained LoRA weights...")
    if os.path.exists(LORA_CHECKPOINT_DIR):
        try:
            # Load the trained LoRA weights from checkpoint
            lora_model = PeftModel.from_pretrained(
                base_model,
                LORA_CHECKPOINT_DIR,
                is_trainable=False  # Set to False for inference only
            )
            print(f"[SUCCESS] Loaded trained LoRA weights from: {LORA_CHECKPOINT_DIR}")
            print(f"[SUCCESS] This model now uses TRAINED LoRA adapters, NOT random initialization")
        except Exception as e:
            print(f"[WARNING] Could not load trained weights: {e}")
            print(f"[WARNING] Using randomly initialized LoRA (model won't show improvement)")
    else:
        print(f"[WARNING] LoRA checkpoint not found at: {LORA_CHECKPOINT_DIR}")
        print(f"[WARNING] Using randomly initialized LoRA (model won't show improvement)")
        print(f"[WARNING] To use trained weights, ensure checkpoint exists at:")
        print(f"           {os.path.abspath(LORA_CHECKPOINT_DIR)}")
    
    print("\n[SUCCESS] LoRA model successfully created!\n")
    
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
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer_base.eos_token_id
                    )
                
                generated_tokens = outputs[0][len(input_ids[0]):]
                response = tokenizer_base.decode(generated_tokens, skip_special_tokens=True)
                
                extracted_answer = extract_answer(response)
                predictions.append(extracted_answer)
                
                if idx < 3:
                    print(f"\n{'='*80}")
                    print(f"[LORA] Debug Sample {idx} - {dataset_name}")
                    print(f"{'='*80}")
                    print(f"Question: {question[:200]}..." if len(question) > 200 else f"Question: {question}")
                    print(f"\nModel Response: {response[:500]}..." if len(response) > 500 else f"\nModel Response: {response}")
                    print(f"\nExtracted Answer: '{extracted_answer}'")
                    print(f"Ground Truth: '{ground_truth_answers[idx]}'")
                    print(f"Match: {'YES' if extracted_answer == ground_truth_answers[idx] else 'NO'}")
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
