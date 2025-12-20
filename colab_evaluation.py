"""
Colab-Optimized LoRA Evaluation Script
Full evaluation with 100 samples per dataset and enhanced visualizations
Designed for Google Colab with T4/P100 GPU
"""

import os
import sys
import signal
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ============================================================================
# Signal patching for Windows compatibility (safe to run on any OS)
# ============================================================================
if not hasattr(signal, 'SIGALRM'):
    signal.SIGALRM = 14
if not hasattr(signal, 'SIGCHLD'):
    signal.SIGCHLD = 17
if not hasattr(signal, 'SIGUSR1'):
    signal.SIGUSR1 = 10
if not hasattr(signal, 'SIGUSR2'):
    signal.SIGUSR2 = 12

if not hasattr(signal, 'alarm'):
    def _dummy_alarm(seconds):
        return 0
    signal.alarm = _dummy_alarm

_original_signal_signal = signal.signal
def _safe_signal_handler(sig, handler):
    try:
        return _original_signal_signal(sig, handler)
    except (ValueError, OSError, AttributeError, TypeError):
        return None

signal.signal = _safe_signal_handler
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Now import our modules
from data_loader import load_all_datasets
from model_evaluator import evaluate_baseline_model, evaluate_lora_model
from config import BASE_MODEL_NAME, LORA_CONFIG, OUTPUT_DIR


def generate_comparison_charts(baseline_results, lora_results, output_dir):
    """
    Generate 3 separate comparison bar charts (one per dataset)
    Each chart shows Baseline vs LoRA accuracy (scale 0-100)
    """
    print("\n" + "="*80)
    print("GENERATING COMPARISON CHARTS")
    print("="*80)
    
    datasets = list(baseline_results.keys())
    colors = {'baseline': '#FF6B6B', 'lora': '#4ECDC4'}
    
    for dataset_name in datasets:
        baseline_acc = baseline_results[dataset_name]["accuracy"] * 100
        lora_acc = lora_results[dataset_name]["accuracy"] * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        # Plot bars
        x_pos = [0, 1]
        accuracies = [baseline_acc, lora_acc]
        labels = ['Baseline (Phi-3)', 'LoRA Fine-tuned']
        bars = ax.bar(x_pos, accuracies, width=0.5, color=[colors['baseline'], colors['lora']], 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Customize chart
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name.upper()} - Baseline vs LoRA', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add improvement annotation
        improvement = lora_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
        improvement_text = f'Improvement: {improvement:+.1f}% ({improvement_pct:+.1f}% relative)'
        ax.text(0.5, 95, improvement_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(output_dir, f'{dataset_name}_comparison.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {chart_path}")
    
    print("")


def generate_improvement_csv(baseline_results, lora_results, datasets_dict, output_dir):
    """
    Generate CSV containing questions where:
    - Baseline model got WRONG
    - LoRA model got RIGHT
    """
    print("="*80)
    print("GENERATING IMPROVEMENT CSV")
    print("="*80)
    
    improvement_records = []
    
    for dataset_name in baseline_results.keys():
        baseline_preds = baseline_results[dataset_name]["predictions"]
        lora_preds = lora_results[dataset_name]["predictions"]
        ground_truth = datasets_dict[dataset_name]["ground_truth_answers"]
        questions = datasets_dict[dataset_name]["questions_list"]
        
        for idx in range(len(questions)):
            baseline_correct = baseline_preds[idx] == ground_truth[idx]
            lora_correct = lora_preds[idx] == ground_truth[idx]
            
            # Record if Baseline FAILED but LoRA SUCCEEDED
            if not baseline_correct and lora_correct:
                improvement_records.append({
                    'Dataset': dataset_name.upper(),
                    'Question_Index': idx,
                    'Question': questions[idx][:200],  # First 200 chars
                    'Ground_Truth': ground_truth[idx],
                    'Baseline_Prediction': baseline_preds[idx],
                    'LoRA_Prediction': lora_preds[idx],
                    'Baseline_Correct': baseline_correct,
                    'LoRA_Correct': lora_correct
                })
    
    # Create DataFrame and save to CSV
    if improvement_records:
        df = pd.DataFrame(improvement_records)
        csv_path = os.path.join(output_dir, 'baseline_wrong_lora_correct.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"✅ Found {len(improvement_records)} questions where Baseline failed but LoRA succeeded")
        print(f"✅ Saved: {csv_path}")
        print(f"\n📊 Breakdown by dataset:")
        for dataset in df['Dataset'].unique():
            count = len(df[df['Dataset'] == dataset])
            print(f"   {dataset}: {count} questions")
    else:
        print("⚠️  No questions found where Baseline failed but LoRA succeeded")
    
    print("")


def main():
    """
    Main Colab evaluation pipeline
    """
    print("\n" + "="*80)
    print(" "*15 + "🚀 LoRA Evaluation - Colab Optimized Edition")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # ========================================================================
        # STEP 1: Load Datasets (100 samples each)
        # ========================================================================
        print("="*80)
        print("STEP 1: LOADING DATASETS (100 samples per dataset)")
        print("="*80)
        print("")
        
        datasets_dict = load_all_datasets(num_samples=100)
        
        for dataset_name in datasets_dict.keys():
            num_questions = len(datasets_dict[dataset_name]["questions_list"])
            print(f"✅ {dataset_name.upper()}: {num_questions} samples loaded")
        print("")
        
        # ========================================================================
        # STEP 2: Evaluate Baseline Model
        # ========================================================================
        print("="*80)
        print("STEP 2: EVALUATING BASELINE MODEL")
        print("="*80)
        print("💡 Using 4-bit quantization (int4) for memory efficiency")
        print("")
        
        baseline_results = evaluate_baseline_model(
            model_name=BASE_MODEL_NAME,
            datasets_dict=datasets_dict
        )
        
        print("\n" + "-"*80)
        print("BASELINE RESULTS SUMMARY:")
        print("-"*80)
        for dataset_name, results in baseline_results.items():
            acc = results["accuracy"] * 100
            correct = results["correct_count"]
            total = results["total_count"]
            print(f"  {dataset_name.upper()}: {acc:.1f}% ({correct}/{total})")
        print("")
        
        # ========================================================================
        # STEP 3: Evaluate LoRA Model
        # ========================================================================
        print("="*80)
        print("STEP 3: EVALUATING LORA FINE-TUNED MODEL")
        print("="*80)
        print("💡 Using same 4-bit quantization for fair comparison")
        print("")
        
        lora_results = evaluate_lora_model(
            model_name=BASE_MODEL_NAME,
            lora_config_dict=LORA_CONFIG,
            datasets_dict=datasets_dict
        )
        
        print("\n" + "-"*80)
        print("LORA RESULTS SUMMARY:")
        print("-"*80)
        for dataset_name, results in lora_results.items():
            acc = results["accuracy"] * 100
            correct = results["correct_count"]
            total = results["total_count"]
            print(f"  {dataset_name.upper()}: {acc:.1f}% ({correct}/{total})")
        print("")
        
        # ========================================================================
        # STEP 4: Generate Comparison Charts
        # ========================================================================
        generate_comparison_charts(baseline_results, lora_results, OUTPUT_DIR)
        
        # ========================================================================
        # STEP 5: Calculate and Display Improvements
        # ========================================================================
        print("="*80)
        print("PERFORMANCE IMPROVEMENTS")
        print("="*80)
        print("")
        
        for dataset_name in baseline_results.keys():
            baseline_acc = baseline_results[dataset_name]["accuracy"] * 100
            lora_acc = lora_results[dataset_name]["accuracy"] * 100
            improvement_abs = lora_acc - baseline_acc
            improvement_rel = (improvement_abs / baseline_acc * 100) if baseline_acc > 0 else 0
            
            print(f"📊 {dataset_name.upper()}:")
            print(f"   Baseline: {baseline_acc:.1f}%")
            print(f"   LoRA:     {lora_acc:.1f}%")
            print(f"   Improvement: {improvement_abs:+.1f}% ({improvement_rel:+.1f}% relative)")
            print("")
        
        # ========================================================================
        # STEP 6: Generate CSV of improvements
        # ========================================================================
        generate_improvement_csv(baseline_results, lora_results, datasets_dict, OUTPUT_DIR)
        
        # ========================================================================
        # STEP 7: Save detailed results as JSON
        # ========================================================================
        print("="*80)
        print("SAVING DETAILED RESULTS")
        print("="*80)
        
        results_summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "baseline_results": baseline_results,
            "lora_results": lora_results
        }
        
        json_path = os.path.join(OUTPUT_DIR, 'detailed_results.json')
        with open(json_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json.dump(
                {
                    "evaluation_timestamp": results_summary["evaluation_timestamp"],
                    "baseline_results": {
                        k: {
                            "accuracy": float(v["accuracy"]),
                            "correct_count": int(v["correct_count"]),
                            "total_count": int(v["total_count"]),
                            "inference_time": float(v["inference_time"]),
                            "predictions": v["predictions"]
                        } for k, v in baseline_results.items()
                    },
                    "lora_results": {
                        k: {
                            "accuracy": float(v["accuracy"]),
                            "correct_count": int(v["correct_count"]),
                            "total_count": int(v["total_count"]),
                            "inference_time": float(v["inference_time"]),
                            "predictions": v["predictions"]
                        } for k, v in lora_results.items()
                    }
                },
                f,
                indent=2
            )
        
        print(f"✅ Saved: {json_path}")
        print("")
        
        # ========================================================================
        # Final Summary
        # ========================================================================
        print("="*80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 All results saved to: {OUTPUT_DIR}/")
        print("")
        print("📂 Generated files:")
        print("   ✅ gsm8k_comparison.png - GSM8K comparison chart")
        print("   ✅ commonsenseqa_comparison.png - CommonsenseQA comparison chart")
        print("   ✅ svamp_comparison.png - SVAMP comparison chart")
        print("   ✅ baseline_wrong_lora_correct.csv - Improvement analysis")
        print("   ✅ detailed_results.json - Complete results data")
        print("="*80)
        print("")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
