"""
Main entry point for LoRA Fine-tuning Evaluation Pipeline
Orchestrates data loading, model evaluation, and result generation
"""

import sys
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


def main():
    """
    Main function orchestrating the evaluation pipeline
    
    Execution flow:
    1. Load all three datasets (GSM8K, CommonsenseQA, SVAMP)
    2. Evaluate baseline model on all datasets
    3. Evaluate LoRA-tuned model on all datasets
    4. Calculate performance comparison metrics
    5. Generate visualization charts
    6. Save evaluation results and predictions
    7. Generate comprehensive evaluation log
    """
    
    print("="*80)
    print(" "*20 + "LoRA Fine-Tuning Evaluation Pipeline")
    print("="*80)
    print("")
    
    try:
        # Step 1: Load Datasets
        print("Step 1: Loading datasets...")
        print("-"*80)
        num_samples = 100  # Number of samples per dataset
        datasets_dict = load_all_datasets(num_samples=num_samples)  # Container for all datasets
        print("")
        
        # Verify dataset loading
        for dataset_name, dataset_content in datasets_dict.items():
            questions_count = len(dataset_content["questions_list"])  # Number of questions (int)
            answers_count = len(dataset_content["ground_truth_answers"])  # Number of answers (int)
            print(f"  {dataset_name}: {questions_count} questions, {answers_count} answers")
        print("")
        
        # Step 2: Evaluate Baseline Model
        print("Step 2: Evaluating baseline model...")
        print("-"*80)
        baseline_results = evaluate_baseline_model(
            model_name=BASE_MODEL_NAME,
            datasets_dict=datasets_dict
        )
        print("")
        
        # Step 3: Evaluate LoRA Model
        print("Step 3: Evaluating LoRA fine-tuned model...")
        print("-"*80)
        lora_results = evaluate_lora_model(
            model_name=BASE_MODEL_NAME,
            lora_config_dict=LORA_CONFIG,
            datasets_dict=datasets_dict
        )
        print("")
        
        # Step 4: Calculate Comparison Metrics
        print("Step 4: Calculating comparison metrics...")
        print("-"*80)
        comparison_metrics = calculate_comparison_metrics(
            baseline_results=baseline_results,
            lora_results=lora_results
        )
        print("")
        
        # Display summary
        for dataset_name, metrics in comparison_metrics.items():
            print(f"  {dataset_name.upper()}:")
            print(f"    Baseline Accuracy: {metrics['baseline_acc']:.2%}")
            print(f"    LoRA Accuracy: {metrics['lora_acc']:.2%}")
            print(f"    Improvement: +{metrics['improvement_pct']:.2f}%")
        print("")
        
        # Step 5: Generate Visualizations
        print("Step 5: Generating visualizations...")
        print("-"*80)
        generate_performance_comparison(baseline_results, lora_results)
        generate_improvement_percentage(comparison_metrics)
        generate_radar_chart(baseline_results, lora_results)
        print("")
        
        # Step 6: Save Results
        print("Step 6: Saving evaluation results...")
        print("-"*80)
        save_metrics_json(baseline_results, lora_results, comparison_metrics)
        save_predictions_json(baseline_results, lora_results, datasets_dict)
        print("")
        
        # Step 7: Generate Evaluation Log
        print("Step 7: Generating evaluation log...")
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
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
