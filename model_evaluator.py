"""
Model evaluator module for comparing baseline and LoRA-tuned models
Implements inference and performance measurement
Supports 8-bit quantization for reduced GPU memory usage
"""

import re
import time
import torch
import gc
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import get_peft_model, LoraConfig
from config import GENERATION_CONFIG, DEVICE, TIMEOUT_SECONDS

# Configuration for 8-bit quantization
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_use_double_quant=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)


class StopOnToken(StoppingCriteria):
    """
    Stop generation when specific tokens are encountered
    """
    def __init__(self, stop_ids, tokenizer):
        self.stop_ids = stop_ids
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0][-1] in self.stop_ids:
            return True
        return False


def print_device_info():
    """
    Display current device information (GPU/CPU)
    """
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
    """
    Extract numerical answer or option letter (A-E) from model response text
    
    Strategy:
    1. For numeric answers: Find ALL numbers in text, return the LAST one
       (since final answer is always at the end of reasoning)
    2. For options (A-E): Use priority rules to find the most explicit answer
    
    Args:
        response (str): Model generated text response
        
    Returns:
        answer (str): Extracted numeric answer string or option letter
    """
    if not response or not response.strip():
        return ""
    
    # PRIORITY 1: Look for explicit option format patterns first
    # These are high-confidence indicators for multiple choice
    explicit_option_patterns = [
        r"^\s*([A-E])\.\s",  # Starts with "A. ", "B. ", etc
        r"answer[\s:]*([A-E])",  # "answer: A" or "answer A"
        r"option[\s:]*([A-E])",  # "option: A" or "option A"
        r"correct answer[\s:]*([A-E])",  # "correct answer: A"
    ]
    
    for pattern in explicit_option_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    # PRIORITY 2: Extract ALL numbers and take the LAST one
    # This handles numeric questions where final answer is at the end
    all_numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', response)
    if all_numbers:
        # Remove commas and decimal points to get clean numbers
        last_number = all_numbers[-1].replace(',', '')
        # Convert to int if it's a whole number, otherwise keep as is
        try:
            if '.' not in last_number:
                return str(int(last_number))
            else:
                return last_number
        except ValueError:
            return all_numbers[-1]
    
    # PRIORITY 3: Look for fallback option patterns
    fallback_patterns = [
        r"[\b\(]([A-E])[\b\)]",  # "(A)" or "(A) text"
        r"([A-E])\.",  # "A." but not at start
    ]
    
    for pattern in fallback_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()
    
    # PRIORITY 4: As last resort, just look for any single letter A-E
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
    
    Args:
        model_name (str): Model identifier from HuggingFace (e.g., "microsoft/phi-2")
        datasets_dict (dict): Dictionary containing datasets
            Structure: {"dataset_name": {"questions_list": [...], "ground_truth_answers": [...]}}
            
    Returns:
        baseline_results (dict): Evaluation results for baseline model
            Structure: {
                "dataset_name": {
                    "accuracy": float (0-1),
                    "correct_count": int,
                    "total_count": int,
                    "predictions": list[str],
                    "inference_time": float (seconds)
                }
            }
    """
    print(f"Loading baseline model: {model_name}")
    print(f"Using 8-bit quantization to save GPU memory...")

    # [BREAKPOINT_3] - Model Loading with 8-bit Quantization
    # Description: Load pretrained model with 8-bit quantization for memory efficiency
    # Load config first with explicit trust_remote_code=True
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="cuda",
        trust_remote_code=True  # Required for Phi-3 custom code
    )
    tokenizer_base = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True  # Required for Phi-3 custom code
    )
    tokenizer_base.pad_token = tokenizer_base.eos_token
    print(f"✅ Baseline model loaded on device: {base_model.device} (8-bit quantized)")
    
    # Print device information
    print_device_info()
    
    baseline_results = {}  # Container for all results (dict[str, dict])
    
    for dataset_name, dataset_content in datasets_dict.items():
        questions_list = dataset_content["questions_list"]  # Question text list (list[str])
        ground_truth_answers = dataset_content["ground_truth_answers"]  # Correct answer list (list[str])
        
        predictions = []  # Model predictions list (list[str])
        correct_count = 0  # Number of correct predictions (int)
        start_time = time.time()  # Inference start timestamp (float)
        
        print(f"Evaluating baseline on {dataset_name}...")
        
        for idx, question in enumerate(questions_list):
            try:
                # [BREAKPOINT_1] - Model Inference
                # Description: Execute model inference with optimized prompt
                # OPTIMIZED: Use Phi-3 standard chat format to prevent question repetition
                prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
                inputs = tokenizer_base(prompt, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(DEVICE)
                attention_mask = inputs["attention_mask"].to(DEVICE)
                
                with torch.no_grad():
                    outputs = base_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=512,              # Balanced: enough for complete answer + reasoning
                        do_sample=False,                 # Greedy decoding for deterministic output
                        pad_token_id=tokenizer_base.eos_token_id
                    )
                # Only decode the NEW tokens to avoid including the prompt in the response
                # This makes extraction much easier
                generated_tokens = outputs[0][len(input_ids[0]):]
                response = tokenizer_base.decode(generated_tokens, skip_special_tokens=True)
                
                # [BREAKPOINT_2] - Answer Extraction and Comparison
                # Description: Extract final numeric answer using regex and compare with ground truth
                extracted_answer = extract_answer(response)  # Returns string
                predictions.append(extracted_answer)
                
                # DEBUG: Print first 3 samples to inspect model output
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
                
                # Compare extracted answer with ground truth
                is_correct = (extracted_answer == ground_truth_answers[idx])
                if is_correct:
                    correct_count += 1
                
                # MEMORY OPTIMIZATION: Clean up intermediate tensors
                del input_ids, attention_mask, outputs
                torch.cuda.empty_cache()
                gc.collect()
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                predictions.append("")
                # Clear cache on error too
                torch.cuda.empty_cache()
                gc.collect()
        
        end_time = time.time()  # Inference end timestamp (float)
        elapsed_time = end_time - start_time  # Total inference time in seconds (float)
        
        # Calculate accuracy
        total_count = len(questions_list)  # Total number of samples (int)
        baseline_accuracy = correct_count / total_count if total_count > 0 else 0.0  # Accuracy ratio (float, 0-1)
        
        baseline_results[dataset_name] = {
            "accuracy": baseline_accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "predictions": predictions,
            "inference_time": elapsed_time
        }
        
        print(f"Baseline {dataset_name} Accuracy: {baseline_accuracy:.2%} ({correct_count}/{total_count})")
    
    # CRITICAL: Free memory before returning
    print("Cleaning up baseline model from memory...")
    del base_model
    del tokenizer_base
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
    
    Args:
        model_name (str): Base model identifier from HuggingFace
        lora_config_dict (dict): LoRA configuration parameters
            Keys: r, lora_alpha, target_modules, lora_dropout, bias, task_type
        datasets_dict (dict): Dictionary containing datasets
            
    Returns:
        lora_results (dict): Evaluation results for LoRA model
            Structure: Same as baseline_results
    """
    print(f"Loading base model for LoRA: {model_name}")
    print(f"Using 8-bit quantization to save GPU memory...")
    
    # [BREAKPOINT_3] - Model Loading with 8-bit Quantization
    # Description: Load pretrained model with 8-bit quantization for memory efficiency
    # Load config first with explicit trust_remote_code=True
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="cuda",
        trust_remote_code=True  # Required for Phi-3 custom code
    )
    tokenizer_base = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True  # Required for Phi-3 custom code
    )
    tokenizer_base.pad_token = tokenizer_base.eos_token
    print(f"✅ LoRA base model loaded on device: {base_model.device} (8-bit quantized)")
    
    # Print device information
    print_device_info()
    
    print(f"Applying LoRA configuration...")
    
    # [BREAKPOINT_4] - LoRA Configuration Application
    # Description: Create LoRA configuration and wrap model with PEFT for parameter-efficient tuning
    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias=lora_config_dict["bias"],
        task_type=lora_config_dict["task_type"]
    )
    
    lora_model = get_peft_model(base_model, lora_config)  # LoRA-adapted model instance (PeftModel)
    lora_model.print_trainable_parameters()
    
    lora_results = {}  # Container for LoRA results (dict[str, dict])
    
    for dataset_name, dataset_content in datasets_dict.items():
        questions_list = dataset_content["questions_list"]  # Question text list (list[str])
        ground_truth_answers = dataset_content["ground_truth_answers"]  # Correct answer list (list[str])
        
        predictions = []  # Model predictions list (list[str])
        correct_count = 0  # Number of correct predictions (int)
        start_time = time.time()  # Inference start timestamp (float)
        
        print(f"Evaluating LoRA model on {dataset_name}...")
        
        for idx, question in enumerate(questions_list):
            try:
                # [BREAKPOINT_5] - LoRA Model Inference
                # Description: Execute inference with LoRA-tuned model
                # OPTIMIZED: Use Phi-3 standard chat format to prevent question repetition
                prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
                inputs = tokenizer_base(prompt, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(DEVICE)
                attention_mask = inputs["attention_mask"].to(DEVICE)
                
                with torch.no_grad():
                    outputs = lora_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=512,              # Balanced: enough for complete answer + reasoning
                        do_sample=False,                 # Greedy decoding for deterministic output
                        pad_token_id=tokenizer_base.eos_token_id
                    )
                # Only decode the NEW tokens to avoid including the prompt in the response
                generated_tokens = outputs[0][len(input_ids[0]):]
                response = tokenizer_base.decode(generated_tokens, skip_special_tokens=True)
                
                # [BREAKPOINT_6] - Answer Extraction and Accuracy Calculation
                # Description: Extract final numeric answer and calculate accuracy
                extracted_answer = extract_answer(response)  # Returns string
                predictions.append(extracted_answer)
                
                # DEBUG: Print first 3 samples to inspect model output
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
                
                # Compare extracted answer with ground truth
                is_correct = (extracted_answer == ground_truth_answers[idx])
                if is_correct:
                    correct_count += 1
                
                # MEMORY OPTIMIZATION: Clean up intermediate tensors
                del input_ids, attention_mask, outputs
                torch.cuda.empty_cache()
                gc.collect()
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                predictions.append("")
                # Clear cache on error too
                torch.cuda.empty_cache()
                gc.collect()
        
        end_time = time.time()  # Inference end timestamp (float)
        elapsed_time = end_time - start_time  # Total inference time in seconds (float)
        
        # Calculate accuracy
        total_count = len(questions_list)  # Total number of samples (int)
        lora_accuracy = correct_count / total_count if total_count > 0 else 0.0  # Accuracy ratio (float, 0-1)
        
        lora_results[dataset_name] = {
            "accuracy": lora_accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "predictions": predictions,
            "inference_time": elapsed_time
        }
        
        print(f"LoRA {dataset_name} Accuracy: {lora_accuracy:.2%} ({correct_count}/{total_count})\n")
    
    # CRITICAL: Free memory after LoRA evaluation
    print("Cleaning up LoRA model from memory...")
    del lora_model
    del base_model
    del tokenizer_base
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleaned.")

    return lora_results
