"""
Standalone LoRA Testing Script
Evaluates only the LoRA-tuned model WITHOUT running baseline evaluation.
Perfect for quick iteration and debugging during development.
"""

import os
import sys
from pathlib import Path
import torch
import gc
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_datasets
from model_evaluator import evaluate_lora_model, print_device_info
from config import LORA_CONFIG, DATASET_CONFIG, OUTPUT_DIR, DEVICE
from utils import print_step_header, print_variable_info, save_results, calculate_comparison_metrics


def main():
    """
    Main function to test LoRA model evaluation only
    """
    print(f"\n{'='*80}")
    print(f"LoRA Model Testing (BASELINE SKIPPED)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ================================================================================
    # STEP 1: Load Datasets
    # ================================================================================
    print_step_header(1, "Loading datasets")
    print("-"*80)
    
    datasets_dict = load_all_datasets()
    print(f"\n✅ Loaded {len(datasets_dict)} datasets")
    for dataset_name in datasets_dict.keys():
        num_samples = len(datasets_dict[dataset_name]["questions_list"])
        print(f"   - {dataset_name}: {num_samples} samples")
    print()
    
    print_device_info()
    
    # ================================================================================
    # STEP 2: Evaluate LoRA Model (NO BASELINE)
    # ================================================================================
    print_step_header(2, "Evaluating LoRA fine-tuned model")
    print("-"*80)
    
    try:
        lora_results = evaluate_lora_model(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            lora_config_dict=LORA_CONFIG,
            datasets_dict=datasets_dict
        )
        print("\n✅ LoRA evaluation completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Error during LoRA evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ================================================================================
    # STEP 3: Display LoRA Results
    # ================================================================================
    print_step_header(3, "LoRA Evaluation Results Summary")
    print("-"*80)
    print("\nLoRA Model Performance by Dataset:\n")
    
    for dataset_name, metrics in lora_results.items():
        accuracy = metrics["accuracy"]
        correct_count = metrics["correct_count"]
        total_count = metrics["total_count"]
        inference_time = metrics["inference_time"]
        
        print(f"  {dataset_name.upper()}:")
        print(f"    - Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        print(f"    - Inference Time: {inference_time:.2f}s")
        print()
    
    # ================================================================================
    # STEP 4: Save LoRA Results
    # ================================================================================
    print_step_header(4, "Saving LoRA evaluation results")
    print("-"*80)
    
    # For LoRA-only testing, we create dummy baseline results
    baseline_results = {
        dataset_name: {
            "accuracy": 0.0,
            "correct_count": 0,
            "total_count": 0,
            "inference_time": 0.0,
            "predictions": []
        }
        for dataset_name in datasets_dict.keys()
    }
    
    save_results(
        baseline_results=baseline_results,
        lora_results=lora_results,
        datasets_dict=datasets_dict
    )
    
    print(f"✅ Results saved to: {OUTPUT_DIR}")
    print()
    
    # ================================================================================
    # COMPLETION
    # ================================================================================
    print("="*80)
    print("LoRA Testing Completed!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    # Memory cleanup
    del datasets_dict, lora_results
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
