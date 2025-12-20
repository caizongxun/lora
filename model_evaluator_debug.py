"""
Debug Model Evaluator - Shows all generated content
顯示所有生成的內容
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
from typing import Dict, List
import warnings
import gc

warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

def print_device_info():
    print("\n[INFO] Device Info:")
    if torch.cuda.is_available():
        print(f"       - Device: GPU (CUDA)")
        print(f"       - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"       - GPU Count: {torch.cuda.device_count()}")
        print(f"       - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"       - CUDA Version: {torch.version.cuda}")
    else:
        print(f"       - Device: CPU")
    print()

def load_base_model(model_name: str):
    print(f"[INFO] Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    print("✓ Model loaded successfully")
    return model, tokenizer

def format_phi3_prompt(prompt: str) -> str:
    """Format prompt with Phi-3 chat template"""
    return f"""<|user|>
{prompt}
<|assistant|>
"""

def generate_response_debug(model, tokenizer, prompt: str, max_tokens: int = 128) -> str:
    """
    Generate response with detailed debug output
    """
    print(f"\n    [DEBUG] Raw prompt: {prompt[:100]}...")
    
    # Format prompt
    formatted_prompt = format_phi3_prompt(prompt)
    print(f"    [DEBUG] Formatted prompt length: {len(formatted_prompt)} chars")
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    print(f"    [DEBUG] Input tokens shape: {inputs['input_ids'].shape}")
    print(f"    [DEBUG] Input tokens (first 20): {inputs['input_ids'][0][:20].tolist()}")
    
    try:
        with torch.inference_mode():
            # Generate
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=min(max_tokens, 128),
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            print(f"    [DEBUG] Output shape: {output.shape}")
            print(f"    [DEBUG] Output tokens (all): {output[0].tolist()}")
        
        # Decode
        response_full = tokenizer.decode(output[0], skip_special_tokens=False)
        print(f"    [DEBUG] Decoded response length: {len(response_full)}")
        print(f"    [DEBUG] Full decoded response:")
        print(f"    >>>START>>>")
        print(f"    {response_full}")
        print(f"    >>>END>>>")
        
        # Extract assistant response
        if "<|assistant|>" in response_full:
            response = response_full.split("<|assistant|>")[-1]
        else:
            response = response_full
        
        response = response.strip()
        
        # Remove end markers
        if "<|end|>" in response:
            response = response.split("<|end|>")[0]
        if "<|user|>" in response:
            response = response.split("<|user|>")[0]
        
        response = response.strip()
        
        print(f"    [DEBUG] Cleaned response length: {len(response)}")
        print(f"    [DEBUG] Cleaned response: {response[:200]}")
        
        return response if response else ""
        
    except Exception as e:
        print(f"    [ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return ""

def evaluate_on_dataset_debug(model, tokenizer, questions: List[str], ground_truths: List[str]) -> Dict:
    """
    Evaluate with debug output
    """
    correct = 0
    total = len(questions)
    start_time = time.time()
    
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        print(f"\n  [Sample {i+1}/{total}]")
        print(f"    Question: {q}")
        print(f"    Ground Truth: {gt}")
        
        try:
            response = generate_response_debug(model, tokenizer, q, max_tokens=128)
            
            if not response:
                print(f"    [Result] ✗ No response generated")
                continue
            
            # Simple check
            if gt.lower() in response.lower() or response.lower()[:20].find(gt.lower()[:5]) != -1:
                correct += 1
                print(f"    [Result] ✓ Correct")
            else:
                print(f"    [Result] ✗ Incorrect")
            
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    inference_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    
    return {
        "correct_count": correct,
        "total_count": total,
        "accuracy": accuracy,
        "inference_time": inference_time
    }

def evaluate_baseline_model_debug(model_name: str, datasets_dict: Dict) -> Dict:
    print("\n" + "="*80)
    print("[BASELINE EVALUATION - DEBUG MODE]")
    print("="*80)
    
    model, tokenizer = load_base_model(model_name)
    
    results = {}
    
    for dataset_name, data in datasets_dict.items():
        print(f"\n[Dataset: {dataset_name}]")
        
        result = evaluate_on_dataset_debug(
            model, tokenizer,
            data["questions_list"],
            data["ground_truth_answers"]
        )
        
        results[dataset_name] = result
        print(f"\n  Accuracy: {result['accuracy']*100:.1f}% ({result['correct_count']}/{result['total_count']})")
        print(f"  Time: {result['inference_time']:.2f}s")
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results
