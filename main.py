"""
Main entry point for LoRA Fine-tuning Evaluation Pipeline
Orchestrates data loading, model evaluation, and result generation
With detailed variable inspection at each step
"""

import sys
import os
import json

# Configure HuggingFace Hub environment BEFORE importing transformers
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# Patch signal module for Windows compatibility (SIGALRM doesn't exist on Windows)
import signal
if not hasattr(signal, 'SIGALRM'):
    # Create a dummy SIGALRM for Windows
    signal.SIGALRM = None

from data_loader import load_all_datasets
from model_evaluator import evaluate_baseline_model, evaluate_lora_model
from utils import (
    calculate_comparison_metrics,
    generate_performance_comparison,
    generate_improvement_percentage,
    generate_radar_chart,
    save_metrics_json,
    save_predictions_json,
    generate_evaluation_log
)
from config import BASE_MODEL_NAME, LORA_CONFIG, DATASET_CONFIG, OUTPUT_DIR


def print_step_header(step_num, step_name):
    """Print formatted step header."""
    print("\n" + "="*80)
    print(f"STEP {step_num}: {step_name}")
    print("="*80)


def print_variable_info(var_name, var_value, var_type=None):
    """Print variable information in formatted way."""
    if var_type is None:
        var_type = type(var_value).__name__
    
    print(f"\nVariable: {var_name}")
    print(f"Type: {var_type}")
    
    if isinstance(var_value, dict):
        print(f"Keys: {list(var_value.keys())}")
        print(f"Size: {len(var_value)} items")
        print("\nContent Preview:")
        for key, val in list(var_value.items())[:3]:  # Show first 3 items
            if isinstance(val, dict):
                accuracy_val = val.get('accuracy', 'N/A')
                print(f"  {key}: {{accuracy: {accuracy_val}, ...}}")
            else:
                print(f"  {key}: {val}")
        if len(var_value) > 3:
            print(f"  ... and {len(var_value) - 3} more items")
    
    elif isinstance(var_value, list):
        print(f"Length: {len(var_value)}")
        print(f"First 3 items: {var_value[:3]}")
        if len(var_value) > 3:
            print(f"... and {len(var_value) - 3} more items")
    
    elif isinstance(var_value, (int, float)):
        print(f"Value: {var_value}")
    
    else:
        print(f"Value: {var_value}")


def main():
    """
    Main function orchestrating the evaluation pipeline
    Each step includes detailed variable inspection
    """
    
    print("="*80)
    print(" "*20 + "LoRA Fine-Tuning Evaluation Pipeline")
    print(" "*15 + "With Step-by-Step Variable Inspection")
    print("="*80)
    print("")
    
    try:
        # STEP 1: Load Datasets
        print_step_header(1, "Loading datasets")
        print("-"*80)
        num_samples = 100  # Number of samples per dataset (int)
        datasets_dict = load_all_datasets(num_samples=num_samples)  # Container for all datasets (dict[str, dict])
        print("")
        
        # [BREAKPOINT_1_VARS] - Dataset loading variables
        print_variable_info("num_samples", num_samples, "int")
        print_variable_info("datasets_dict", datasets_dict, "dict")
        
        # Additional statistics
        print("\n" + "-"*80)
        print("Dataset Statistics:")
        print("-"*80)
        for dataset_name, dataset_content in datasets_dict.items():
            questions_count = len(dataset_content["questions_list"])  # Number of questions (int)
            answers_count = len(dataset_content["ground_truth_answers"])  # Number of answers (int)
            print(f"  {dataset_name.upper()}:")
            print(f"    - Questions: {questions_count}")
            print(f"    - Answers: {answers_count}")
            if questions_count > 0:
                sample_q = dataset_content['questions_list'][0][:80]
                sample_a = dataset_content['ground_truth_answers'][0]
                print(f"    - Sample question: {sample_q}...")
                print(f"    - Sample answer: {sample_a}")
        print("")
        
        # STEP 2: Evaluate Baseline Model
        print_step_header(2, "Evaluating baseline model")
        print("-"*80)
        baseline_results = evaluate_baseline_model(
            model_name=BASE_MODEL_NAME,
            datasets_dict=datasets_dict
        )  # Results for baseline model (dict[str, dict])
        print("")
        
        # [BREAKPOINT_2_VARS] - Baseline results
        print_variable_info("baseline_results", baseline_results, "dict")
        
        # Additional statistics
        print("\n" + "-"*80)
        print("Baseline Model Statistics:")
        print("-"*80)
        for dataset_name, results in baseline_results.items():
            baseline_accuracy = results["accuracy"]  # Accuracy ratio (float, 0-1)
            correct_count = results["correct_count"]  # Number of correct predictions (int)
            total_count = results["total_count"]  # Total number of samples (int)
            inf_time = results['inference_time']
            print(f"  {dataset_name.upper()}:")
            print(f"    - Accuracy: {baseline_accuracy:.2%}")
            print(f"    - Correct: {correct_count}/{total_count}")
            print(f"    - Inference Time: {inf_time:.2f}s")
        print("")
        
        # STEP 3: Evaluate LoRA Model
        print_step_header(3, "Evaluating LoRA fine-tuned model")
        print("-"*80)
        lora_results = evaluate_lora_model(
            model_name=BASE_MODEL_NAME,
            lora_config_dict=LORA_CONFIG,
            datasets_dict=datasets_dict
        )  # Results for LoRA model (dict[str, dict])
        print("")
        
        # [BREAKPOINT_3_VARS] - LoRA results
        print_variable_info("lora_results", lora_results, "dict")
        
        # Additional statistics
        print("\n" + "-"*80)
        print("LoRA Model Statistics:")
        print("-"*80)
        for dataset_name, results in lora_results.items():
            lora_accuracy = results["accuracy"]  # Accuracy ratio (float, 0-1)
            correct_count = results["correct_count"]  # Number of correct predictions (int)
            total_count = results["total_count"]  # Total number of samples (int)
            inf_time = results['inference_time']
            print(f"  {dataset_name.upper()}:")
            print(f"    - Accuracy: {lora_accuracy:.2%}")
            print(f"    - Correct: {correct_count}/{total_count}")
            print(f"    - Inference Time: {inf_time:.2f}s")
        print("")
        
        # STEP 4: Calculate Comparison Metrics
        print_step_header(4, "Calculating comparison metrics")
        print("-"*80)
        comparison_metrics = calculate_comparison_metrics(
            baseline_results=baseline_results,
            lora_results=lora_results
        )  # Computed comparison metrics (dict[str, dict])
        print("")
        
        # [BREAKPOINT_4_VARS] - Comparison metrics
        print_variable_info("comparison_metrics", comparison_metrics, "dict")
        
        # Additional statistics
        print("\n" + "-"*80)
        print("Performance Improvement Summary:")
        print("-"*80)
        for dataset_name, metrics in comparison_metrics.items():
            baseline_acc = metrics["baseline_acc"]  # Baseline accuracy (float, 0-1)
            lora_acc = metrics["lora_acc"]  # LoRA accuracy (float, 0-1)
            improvement_abs = metrics["improvement_abs"]  # Absolute improvement (float, 0-1)
            improvement_pct = metrics["improvement_pct"]  # Relative improvement (float, %)
            
            print(f"  {dataset_name.upper()}:")
            print(f"    - Baseline Accuracy: {baseline_acc:.2%}")
            print(f"    - LoRA Accuracy: {lora_acc:.2%}")
            print(f"    - Absolute Improvement: {improvement_abs:.2%}")
            print(f"    - Relative Improvement: +{improvement_pct:.2f}%")
        print("")
        
        # STEP 5: Generate Visualizations
        print_step_header(5, "Generating visualizations")
        print("-"*80)
        generate_performance_comparison(baseline_results, lora_results)
        generate_improvement_percentage(comparison_metrics)
        generate_radar_chart(baseline_results, lora_results)
        print("")
        
        # STEP 6: Save Results
        print_step_header(6, "Saving evaluation results")
        print("-"*80)
        save_metrics_json(baseline_results, lora_results, comparison_metrics)
        save_predictions_json(baseline_results, lora_results, datasets_dict)
        print("")
        
        # STEP 7: Generate Evaluation Log
        print_step_header(7, "Generating evaluation log")
        print("-"*80)
        generate_evaluation_log(baseline_results, lora_results, comparison_metrics)
        print("")
        
        print("="*80)
        print("Evaluation pipeline completed successfully!")
        print(f"All results saved to: {OUTPUT_DIR}/")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"")
        print(f"Error during evaluation: {str(e)}")
        print(f"")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
