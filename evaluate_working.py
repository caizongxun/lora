"""
Standalone LoRA Evaluation Script
Automatically downloads trained LoRA weights from HuggingFace Hub
and compares Base vs LoRA model performance.

Key fix: Do NOT use device_map, do NOT call .to() or .cuda()
Let BitsAndBytes handle device placement automatically

Usage:
    python evaluate_working.py --hf_model_id zongowo111/phi3-lora-gsm8k-commonsense
"""

import os
import sys
import argparse
import json
import torch
from datetime import datetime
from typing import Dict, List, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm


class LoraEvaluator:
    """
    Evaluates and compares Base Model vs LoRA-tuned Model
    """
    
    def __init__(
        self,
        base_model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        hf_model_id: str = None,
        use_4bit: bool = True,
        max_samples: int = 100
    ):
        """
        Initialize evaluator
        
        Args:
            base_model_name: HF Hub ID of base model
            hf_model_id: HF Hub ID of trained LoRA repository
            use_4bit: Whether to use 4-bit quantization
            max_samples: Max samples to evaluate per dataset
        """
        self.base_model_name = base_model_name
        self.hf_model_id = hf_model_id
        self.use_4bit = use_4bit
        self.max_samples = max_samples
        
        self.base_model = None
        self.lora_model = None
        self.tokenizer = None
        
        print("\n" + "="*80)
        print("LoRA Evaluation: Base vs Fine-tuned Model")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"   - Base Model: {base_model_name}")
        print(f"   - LoRA Model ID: {hf_model_id}")
        print(f"   - 4-bit Quantization: {use_4bit}")
        print(f"   - Max Samples per Dataset: {max_samples}")
        print()
    
    def setup_models(self):
        """Load base model and LoRA weights"""
        print("\n" + "="*80)
        print("STEP 1: LOADING MODELS")
        print("="*80)
        
        quantization_config = None
        if self.use_4bit:
            print("\nUsing 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        print(f"\nLoading base model: {self.base_model_name}...")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("Base model loaded")
        
        print(f"\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer loaded")
        
        if self.hf_model_id:
            print(f"\nLoading LoRA weights from: {self.hf_model_id}...")
            try:
                self.lora_model = PeftModel.from_pretrained(
                    self.base_model,
                    self.hf_model_id,
                    is_trainable=False
                )
                print("LoRA weights loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load LoRA weights: {e}")
                print("Will only evaluate base model")
                self.lora_model = None
        else:
            print("No LoRA model ID provided, will only evaluate base model")
            self.lora_model = None
    
    def load_datasets(self):
        """Load evaluation datasets"""
        print("\n" + "="*80)
        print("STEP 2: LOADING DATASETS")
        print("="*80)
        
        datasets_dict = {}
        
        print("\nLoading GSM8K...")
        try:
            gsm8k = load_dataset("gsm8k", "main", split=f"test[:{self.max_samples}]")
            datasets_dict["gsm8k"] = gsm8k
            print(f"GSM8K: {len(gsm8k)} samples")
        except Exception as e:
            print(f"Failed to load GSM8K: {e}")
        
        print("\nLoading CommonsenseQA...")
        try:
            cqa = load_dataset("commonsense_qa", split=f"validation[:{self.max_samples}]")
            datasets_dict["commonsenseqa"] = cqa
            print(f"CommonsenseQA: {len(cqa)} samples")
        except Exception as e:
            print(f"Failed to load CommonsenseQA: {e}")
        
        print("\nLoading SVAMP...")
        try:
            svamp = load_dataset("svamp", split=f"test[:{self.max_samples}]")
            datasets_dict["svamp"] = svamp
            print(f"SVAMP: {len(svamp)} samples")
        except Exception as e:
            print(f"Failed to load SVAMP: {e}")
        
        return datasets_dict
    
    def extract_answer_gsm8k(self, text: str) -> str:
        """Extract final answer from GSM8K format (#### answer)"""
        if "####" in text:
            return text.split("####")[-1].strip()
        return text.strip()
    
    def generate_response(self, prompt: str, model, max_length: int = 256) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                num_beams=1,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response
    
    def evaluate_gsm8k(self, dataset, model) -> Tuple[int, int, float]:
        """Evaluate on GSM8K dataset"""
        correct = 0
        total = len(dataset)
        
        print(f"\nEvaluating on GSM8K ({total} samples)...")
        
        for i, sample in enumerate(tqdm(dataset, desc="GSM8K")):
            try:
                question = sample["question"]
                ground_truth = sample["answer"]
                
                true_answer = self.extract_answer_gsm8k(ground_truth)
                
                prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
                response = self.generate_response(prompt, model, max_length=256)
                
                pred_answer = self.extract_answer_gsm8k(response)
                
                if true_answer in pred_answer or pred_answer == true_answer:
                    correct += 1
            except Exception as e:
                pass
        
        accuracy = correct / total if total > 0 else 0
        return correct, total, accuracy
    
    def evaluate_commonsenseqa(self, dataset, model) -> Tuple[int, int, float]:
        """Evaluate on CommonsenseQA dataset"""
        correct = 0
        total = len(dataset)
        
        print(f"\nEvaluating on CommonsenseQA ({total} samples)...")
        
        for i, sample in enumerate(tqdm(dataset, desc="CommonsenseQA")):
            try:
                question = sample["question"]
                choices = sample["choices"]["text"]
                ground_truth = choices[int(sample["answerKey"][0], 36) - 10]
                
                prompt = f"<|user|>\n{question}\nChoices: {', '.join(choices)}<|end|>\n<|assistant|>\n"
                response = self.generate_response(prompt, model, max_length=128)
                
                for choice in choices:
                    if choice.lower() in response.lower():
                        if choice == ground_truth:
                            correct += 1
                        break
            except Exception as e:
                pass
        
        accuracy = correct / total if total > 0 else 0
        return correct, total, accuracy
    
    def evaluate_svamp(self, dataset, model) -> Tuple[int, int, float]:
        """Evaluate on SVAMP dataset"""
        correct = 0
        total = len(dataset)
        
        print(f"\nEvaluating on SVAMP ({total} samples)...")
        
        for i, sample in enumerate(tqdm(dataset, desc="SVAMP")):
            try:
                question = sample["Body"] + " " + sample["Question"]
                ground_truth = str(sample["Answer"])
                
                prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
                response = self.generate_response(prompt, model, max_length=128)
                
                if ground_truth in response:
                    correct += 1
            except Exception as e:
                pass
        
        accuracy = correct / total if total > 0 else 0
        return correct, total, accuracy
    
    def evaluate(self, datasets_dict: Dict):
        """Run full evaluation"""
        print("\n" + "="*80)
        print("STEP 3: EVALUATION")
        print("="*80)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "base_model": self.base_model_name,
            "lora_model": self.hf_model_id,
            "datasets": {}
        }
        
        for dataset_name, dataset in datasets_dict.items():
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name.upper()}")
            print(f"{'='*80}")
            
            print(f"\n[1/2] Evaluating Base Model...")
            if dataset_name == "gsm8k":
                base_correct, base_total, base_acc = self.evaluate_gsm8k(dataset, self.base_model)
            elif dataset_name == "commonsenseqa":
                base_correct, base_total, base_acc = self.evaluate_commonsenseqa(dataset, self.base_model)
            else:
                base_correct, base_total, base_acc = self.evaluate_svamp(dataset, self.base_model)
            
            lora_acc = None
            if self.lora_model:
                print(f"\n[2/2] Evaluating LoRA Model...")
                if dataset_name == "gsm8k":
                    lora_correct, lora_total, lora_acc = self.evaluate_gsm8k(dataset, self.lora_model)
                elif dataset_name == "commonsenseqa":
                    lora_correct, lora_total, lora_acc = self.evaluate_commonsenseqa(dataset, self.lora_model)
                else:
                    lora_correct, lora_total, lora_acc = self.evaluate_svamp(dataset, self.lora_model)
            
            results["datasets"][dataset_name] = {
                "base_accuracy": base_acc,
                "lora_accuracy": lora_acc,
                "improvement": (lora_acc - base_acc) if lora_acc else None,
                "samples": base_total
            }
            
            print(f"\n{dataset_name.upper()} Results:")
            print(f"   Base Model Accuracy: {base_acc:.2%}")
            if lora_acc:
                print(f"   LoRA Model Accuracy: {lora_acc:.2%}")
                improvement = lora_acc - base_acc
                print(f"   Improvement: {improvement:+.2%}")
        
        return results
    
    def save_results(self, results: Dict):
        """Save evaluation results to JSON"""
        output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model vs Base model")
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default=os.getenv("HF_MODEL_ID"),
        help="HuggingFace Hub ID of trained LoRA model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="HuggingFace Hub ID of base model"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum samples to evaluate per dataset"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Don't use 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    if not args.hf_model_id:
        print("\nError: HF_MODEL_ID not provided!")
        print("\nUsage:")
        print("   python evaluate_working.py --hf_model_id username/phi3-lora-xxx")
        print("\nOr set environment variable:")
        print("   export HF_MODEL_ID=username/phi3-lora-xxx")
        print("   python evaluate_working.py")
        sys.exit(1)
    
    evaluator = LoraEvaluator(
        base_model_name=args.base_model,
        hf_model_id=args.hf_model_id,
        use_4bit=not args.no_4bit,
        max_samples=args.max_samples
    )
    
    evaluator.setup_models()
    datasets = evaluator.load_datasets()
    results = evaluator.evaluate(datasets)
    evaluator.save_results(results)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
