"""
Colab-Optimized LoRA vs Base Model Evaluation Script
Automatically downloads LoRA weights from HuggingFace Hub
and generates comprehensive comparison reports.

Designed for Google Colab - just run and it handles everything!

Usage in Colab:
    !python colab_evaluate_lora.py --hf_model_id zongowo111/phi3-lora-gsm8k-commonsense
"""

import os
import sys
import argparse
import json
import torch
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm


class ColabLoRAEvaluator:
    """
    Colab-optimized evaluator for comparing Base Model vs LoRA-tuned Model
    """
    
    def __init__(
        self,
        base_model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        hf_model_id: str = None,
        use_4bit: bool = True,
        max_samples: int = 50,  # Smaller for faster Colab execution
        save_to_drive: bool = True
    ):
        self.base_model_name = base_model_name
        self.hf_model_id = hf_model_id
        self.use_4bit = use_4bit
        self.max_samples = max_samples
        self.save_to_drive = save_to_drive
        
        self.base_model = None
        self.lora_model = None
        self.tokenizer = None
        self.results_dir = None
        
        # Setup Google Drive
        self._setup_drive()
        
        print("\n" + "="*80)
        print("🚀 LoRA vs Base Model Evaluation (Colab Optimized)")
        print("="*80)
        print(f"\n📊 Configuration:")
        print(f"   - Base Model: {base_model_name}")
        print(f"   - LoRA Model: {hf_model_id}")
        print(f"   - 4-bit Quantization: {use_4bit}")
        print(f"   - Max Samples: {max_samples}")
        if self.results_dir:
            print(f"   - Results Save Path: {self.results_dir}")
        print()
    
    def _setup_drive(self):
        """Setup Google Drive for saving results"""
        if self.save_to_drive:
            try:
                from google.colab import drive
                if not os.path.exists('/content/drive'):
                    print("📁 Mounting Google Drive...")
                    drive.mount('/content/drive', force_remount=True)
                
                self.results_dir = '/content/drive/MyDrive/lora_evaluation_results'
                os.makedirs(self.results_dir, exist_ok=True)
                print(f"✅ Google Drive mounted. Results will be saved to:\n   {self.results_dir}\n")
            except Exception as e:
                print(f"⚠️  Warning: Could not mount Google Drive: {e}")
                print("   Results will be saved locally.\n")
                self.results_dir = '/content/lora_evaluation_results'
                os.makedirs(self.results_dir, exist_ok=True)
        else:
            self.results_dir = '/content/lora_evaluation_results'
            os.makedirs(self.results_dir, exist_ok=True)
    
    def setup_models(self):
        """Load base model and LoRA weights"""
        print("\n" + "="*80)
        print("STEP 1: LOADING MODELS")
        print("="*80)
        
        # Setup quantization
        quantization_config = None
        if self.use_4bit:
            print("\n📦 Using 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load base model
        print(f"\n⬇️ Loading base model: {self.base_model_name}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Base model loaded")
        
        # Load tokenizer
        print(f"\n⬇️ Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ Tokenizer loaded")
        
        # Load LoRA weights
        if self.hf_model_id:
            print(f"\n⬇️ Loading LoRA weights from HF Hub: {self.hf_model_id}...")
            try:
                self.lora_model = PeftModel.from_pretrained(
                    self.base_model,
                    self.hf_model_id,
                    is_trainable=False
                )
                print("✅ LoRA weights loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load LoRA weights: {e}")
                print("   Will only evaluate base model")
                self.lora_model = None
        else:
            print("⚠️  No LoRA model ID provided")
            self.lora_model = None
    
    def load_datasets(self):
        """Load evaluation datasets"""
        print("\n" + "="*80)
        print("STEP 2: LOADING DATASETS")
        print("="*80)
        
        datasets_dict = {}
        
        print("\n📥 Loading GSM8K (math reasoning)...")
        try:
            gsm8k = load_dataset("gsm8k", "main", split=f"test[:{self.max_samples}]")
            datasets_dict["gsm8k"] = gsm8k
            print(f"✅ GSM8K: {len(gsm8k)} samples")
        except Exception as e:
            print(f"❌ Failed to load GSM8K: {e}")
        
        print("\n📥 Loading CommonsenseQA (common sense)...")
        try:
            cqa = load_dataset("commonsense_qa", split=f"validation[:{self.max_samples}]")
            datasets_dict["commonsenseqa"] = cqa
            print(f"✅ CommonsenseQA: {len(cqa)} samples")
        except Exception as e:
            print(f"❌ Failed to load CommonsenseQA: {e}")
        
        print("\n📥 Loading SVAMP (math word problems)...")
        try:
            svamp = load_dataset("svamp", split=f"test[:{self.max_samples}]")
            datasets_dict["svamp"] = svamp
            print(f"✅ SVAMP: {len(svamp)} samples")
        except Exception as e:
            print(f"❌ Failed to load SVAMP: {e}")
        
        return datasets_dict
    
    def evaluate_gsm8k(self, dataset, model) -> float:
        """Evaluate on GSM8K dataset"""
        correct = 0
        total = min(len(dataset), self.max_samples)
        
        for i, sample in enumerate(tqdm(dataset[:total], desc="GSM8K", leave=False)):
            try:
                question = sample["question"]
                ground_truth = sample["answer"].split("####")[-1].strip()
                
                prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=1,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                
                if ground_truth in response or ground_truth == response:
                    correct += 1
            except:
                pass
        
        return correct / total if total > 0 else 0
    
    def evaluate_commonsenseqa(self, dataset, model) -> float:
        """Evaluate on CommonsenseQA dataset"""
        correct = 0
        total = min(len(dataset), self.max_samples)
        
        for i, sample in enumerate(tqdm(dataset[:total], desc="CommonsenseQA", leave=False)):
            try:
                question = sample["question"]
                choices = sample["choices"]["text"]
                
                # Get ground truth answer
                try:
                    ground_truth_idx = int(sample["answerKey"][0], 36) - 10
                    if 0 <= ground_truth_idx < len(choices):
                        ground_truth = choices[ground_truth_idx]
                    else:
                        continue
                except:
                    continue
                
                prompt = f"<|user|>\n{question}\nChoices: {', '.join(choices)}<|end|>\n<|assistant|>\n"
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=1,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip().lower()
                
                if ground_truth.lower() in response:
                    correct += 1
            except:
                pass
        
        return correct / total if total > 0 else 0
    
    def evaluate_svamp(self, dataset, model) -> float:
        """Evaluate on SVAMP dataset"""
        correct = 0
        total = min(len(dataset), self.max_samples)
        
        for i, sample in enumerate(tqdm(dataset[:total], desc="SVAMP", leave=False)):
            try:
                question = sample["Body"] + " " + sample["Question"]
                ground_truth = str(sample["Answer"])
                
                prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=1,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                
                if ground_truth in response:
                    correct += 1
            except:
                pass
        
        return correct / total if total > 0 else 0
    
    def evaluate(self, datasets_dict: Dict) -> Dict:
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
            print(f"Evaluating: {dataset_name.upper()}")
            print(f"{'='*80}")
            
            # Evaluate base model
            print(f"\n[1/2] Base Model...")
            if dataset_name == "gsm8k":
                base_acc = self.evaluate_gsm8k(dataset, self.base_model)
            elif dataset_name == "commonsenseqa":
                base_acc = self.evaluate_commonsenseqa(dataset, self.base_model)
            else:
                base_acc = self.evaluate_svamp(dataset, self.base_model)
            
            # Evaluate LoRA model
            lora_acc = None
            if self.lora_model:
                print(f"\n[2/2] LoRA Model...")
                if dataset_name == "gsm8k":
                    lora_acc = self.evaluate_gsm8k(dataset, self.lora_model)
                elif dataset_name == "commonsenseqa":
                    lora_acc = self.evaluate_commonsenseqa(dataset, self.lora_model)
                else:
                    lora_acc = self.evaluate_svamp(dataset, self.lora_model)
            
            # Store results
            result_entry = {
                "base_accuracy": round(base_acc * 100, 2),
                "lora_accuracy": round(lora_acc * 100, 2) if lora_acc else None,
                "improvement_pct": round((lora_acc - base_acc) * 100, 2) if lora_acc else None,
                "samples": len(dataset)
            }
            results["datasets"][dataset_name] = result_entry
            
            # Print summary
            print(f"\n✅ {dataset_name.upper()} Results:")
            print(f"   Base Model Accuracy: {base_acc:.1%}")
            if lora_acc:
                print(f"   LoRA Model Accuracy: {lora_acc:.1%}")
                improvement = lora_acc - base_acc
                arrow = "📈" if improvement > 0 else "📉"
                print(f"   {arrow} Improvement: {improvement:+.1%}")
        
        return results
    
    def save_results(self, results: Dict) -> Dict:
        """Save evaluation results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_file = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.json")
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save CSV for easy viewing
        csv_file = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "Base Accuracy (%)", "LoRA Accuracy (%)", "Improvement (%)", "Samples"])
            for dataset_name, data in results["datasets"].items():
                writer.writerow([
                    dataset_name,
                    data["base_accuracy"],
                    data["lora_accuracy"] if data["lora_accuracy"] else "N/A",
                    data["improvement_pct"] if data["improvement_pct"] else "N/A",
                    data["samples"]
                ])
        
        print(f"\n✅ Results saved:")
        print(f"   JSON: {json_file}")
        print(f"   CSV:  {csv_file}")
        
        return {"json": json_file, "csv": csv_file}
    
    def print_summary(self, results: Dict):
        """Print summary report"""
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nBase Model: {self.base_model_name}")
        print(f"LoRA Model: {self.hf_model_id}")
        
        print(f"\n{'Dataset':<20} {'Base':<12} {'LoRA':<12} {'Improvement':<12}")
        print("-" * 60)
        
        total_improvement = 0
        dataset_count = 0
        
        for dataset_name, data in results["datasets"].items():
            base = data["base_accuracy"]
            lora = data["lora_accuracy"]
            improvement = data["improvement_pct"]
            
            lora_str = f"{lora}%" if lora else "N/A"
            imp_str = f"{improvement:+.2f}%" if improvement else "N/A"
            
            print(f"{dataset_name:<20} {base:>10.2f}% {lora_str:>10} {imp_str:>10}")
            
            if improvement:
                total_improvement += improvement
                dataset_count += 1
        
        if dataset_count > 0:
            avg_improvement = total_improvement / dataset_count
            print("-" * 60)
            print(f"{'Average':<20} {'':<12} {'':<12} {avg_improvement:>+10.2f}%")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model vs Base model in Colab")
    parser.add_argument(
        "--hf_model_id",
        type=str,
        required=True,
        help="HuggingFace Hub ID of trained LoRA model (e.g., zongowo111/phi3-lora-gsm8k-commonsense)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum samples to evaluate per dataset (default: 50)"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Don't use 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ColabLoRAEvaluator(
        hf_model_id=args.hf_model_id,
        use_4bit=not args.no_4bit,
        max_samples=args.max_samples,
        save_to_drive=True
    )
    
    # Setup models
    evaluator.setup_models()
    
    # Load datasets
    datasets = evaluator.load_datasets()
    
    if not datasets:
        print("❌ No datasets loaded. Exiting.")
        return
    
    # Run evaluation
    results = evaluator.evaluate(datasets)
    
    # Save results
    evaluator.save_results(results)
    
    # Print summary
    evaluator.print_summary(results)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
