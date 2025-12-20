"""
Base Model vs LoRA Model Comparison Visualization
Generates comprehensive comparison charts and statistics
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, Any, List
import json
from datetime import datetime

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_accuracy_comparison_chart(baseline_results: Dict, lora_results: Dict) -> str:
    """
    Create accuracy comparison bar chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Baseline vs LoRA Model - Accuracy Comparison', fontsize=16, fontweight='bold')
    
    datasets = list(baseline_results.keys())
    baseline_accs = [baseline_results[d]['accuracy'] * 100 for d in datasets]
    lora_accs = [lora_results[d]['accuracy'] * 100 for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Chart 1: Side-by-side comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, baseline_accs, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, lora_accs, width, label='LoRA', color='#4ECDC4', alpha=0.8)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy by Dataset', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=15, ha='right')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Improvement percentage
    ax2 = axes[1]
    improvements = [(lora_accs[i] - baseline_accs[i]) for i in range(len(datasets))]
    colors = ['#2ECC71' if imp >= 0 else '#E74C3C' for imp in improvements]
    bars3 = ax2.bar(datasets, improvements, color=colors, alpha=0.8)
    
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('LoRA Improvement over Baseline', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xticklabels(datasets, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%',
                ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    output_path = '/content/lora/comparison_accuracy.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] Saved: {output_path}")
    plt.close()
    return output_path


def create_inference_time_comparison(baseline_results: Dict, lora_results: Dict) -> str:
    """
    Create inference time comparison chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Baseline vs LoRA Model - Inference Time Comparison', fontsize=16, fontweight='bold')
    
    datasets = list(baseline_results.keys())
    baseline_times = [baseline_results[d]['inference_time'] for d in datasets]
    lora_times = [lora_results[d]['inference_time'] for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Chart 1: Side-by-side comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, baseline_times, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, lora_times, width, label='LoRA', color='#4ECDC4', alpha=0.8)
    
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Inference Time by Dataset', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=15, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Time difference
    ax2 = axes[1]
    time_diff = [(baseline_times[i] - lora_times[i]) for i in range(len(datasets))]
    colors = ['#2ECC71' if diff >= 0 else '#E74C3C' for diff in time_diff]
    bars3 = ax2.bar(datasets, time_diff, color=colors, alpha=0.8)
    
    ax2.set_ylabel('Time Difference (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Baseline Time - LoRA Time', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xticklabels(datasets, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, diff in zip(bars3, time_diff):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{diff:+.2f}s',
                ha='center', va='bottom' if diff >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    output_path = '/content/lora/comparison_inference_time.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] Saved: {output_path}")
    plt.close()
    return output_path


def create_detailed_stats_chart(baseline_results: Dict, lora_results: Dict) -> str:
    """
    Create detailed statistics comparison table
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    datasets = list(baseline_results.keys())
    
    # Prepare table data
    table_data = []
    table_data.append(['Dataset', 'Model', 'Accuracy', 'Correct/Total', 'Inference Time (s)'])
    
    for dataset in datasets:
        baseline = baseline_results[dataset]
        lora = lora_results[dataset]
        
        # Baseline row
        table_data.append([
            dataset if dataset == list(datasets)[0] else '',
            'Baseline',
            f"{baseline['accuracy']*100:.2f}%",
            f"{baseline['correct_count']}/{baseline['total_count']}",
            f"{baseline['inference_time']:.2f}"
        ])
        
        # LoRA row
        improvement = (lora['accuracy'] - baseline['accuracy']) * 100
        table_data.append([
            '',
            'LoRA',
            f"{lora['accuracy']*100:.2f}% ({improvement:+.2f}%)",
            f"{lora['correct_count']}/{lora['total_count']}",
            f"{lora['inference_time']:.2f}"
        ])
        
        # Separator row
        table_data.append(['', '', '', '', ''])
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                    colWidths=[0.2, 0.15, 0.25, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#34495E')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 3 == 0:  # Separator rows
                cell.set_facecolor('#ECF0F1')
            elif table_data[i][1] == 'Baseline':
                cell.set_facecolor('#FFE5E5')
            elif table_data[i][1] == 'LoRA':
                cell.set_facecolor('#E5F5F3')
            cell.set_text_props(size=10)
    
    plt.title('Detailed Comparison Statistics', fontsize=16, fontweight='bold', pad=20)
    
    output_path = '/content/lora/comparison_statistics.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SUCCESS] Saved: {output_path}")
    plt.close()
    return output_path


def create_summary_report(baseline_results: Dict, lora_results: Dict) -> str:
    """
    Create a text summary report
    """
    report = []
    report.append("=" * 80)
    report.append("BASE MODEL vs LoRA MODEL - COMPREHENSIVE COMPARISON REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    all_baseline_acc = [baseline_results[d]['accuracy'] for d in baseline_results]
    all_lora_acc = [lora_results[d]['accuracy'] for d in lora_results]
    
    avg_baseline = np.mean(all_baseline_acc) * 100
    avg_lora = np.mean(all_lora_acc) * 100
    avg_improvement = avg_lora - avg_baseline
    
    report.append("OVERALL SUMMARY")
    report.append("-" * 80)
    report.append(f"Average Baseline Accuracy:  {avg_baseline:.2f}%")
    report.append(f"Average LoRA Accuracy:      {avg_lora:.2f}%")
    report.append(f"Average Improvement:        {avg_improvement:+.2f}%")
    report.append("")
    
    # Per-dataset statistics
    report.append("PER-DATASET BREAKDOWN")
    report.append("-" * 80)
    
    for dataset in baseline_results.keys():
        baseline = baseline_results[dataset]
        lora = lora_results[dataset]
        
        acc_improvement = (lora['accuracy'] - baseline['accuracy']) * 100
        time_diff = baseline['inference_time'] - lora['inference_time']
        
        report.append(f"\n[{dataset.upper()}]")
        report.append(f"  Baseline:")
        report.append(f"    - Accuracy:      {baseline['accuracy']*100:.2f}% ({baseline['correct_count']}/{baseline['total_count']})")
        report.append(f"    - Inference Time: {baseline['inference_time']:.2f}s")
        report.append(f"  ")
        report.append(f"  LoRA:")
        report.append(f"    - Accuracy:      {lora['accuracy']*100:.2f}% ({lora['correct_count']}/{lora['total_count']})")
        report.append(f"    - Inference Time: {lora['inference_time']:.2f}s")
        report.append(f"  ")
        report.append(f"  Improvement:")
        report.append(f"    - Accuracy:      {acc_improvement:+.2f}%")
        report.append(f"    - Speed:         {time_diff:+.2f}s (Baseline-LoRA)")
    
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    report_path = '/content/lora/comparison_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n[SUCCESS] Report saved: {report_path}")
    return report_path


def generate_all_comparisons(baseline_results: Dict, lora_results: Dict) -> Dict[str, str]:
    """
    Generate all comparison visualizations
    """
    print("\n" + "="*80)
    print("[START] Generating Comparison Visualizations")
    print("="*80)
    
    output_files = {}
    
    print("\n[STEP 1] Creating accuracy comparison chart...")
    output_files['accuracy'] = create_accuracy_comparison_chart(baseline_results, lora_results)
    
    print("\n[STEP 2] Creating inference time comparison chart...")
    output_files['inference_time'] = create_inference_time_comparison(baseline_results, lora_results)
    
    print("\n[STEP 3] Creating detailed statistics chart...")
    output_files['statistics'] = create_detailed_stats_chart(baseline_results, lora_results)
    
    print("\n[STEP 4] Creating summary report...")
    output_files['report'] = create_summary_report(baseline_results, lora_results)
    
    print("\n" + "="*80)
    print("[SUCCESS] All visualizations generated successfully!")
    print("="*80)
    print("\nGenerated files:")
    for chart_type, path in output_files.items():
        print(f"  - {chart_type}: {path}")
    
    return output_files


if __name__ == "__main__":
    print("[INFO] This module is for comparison visualization")
    print("[INFO] Import and call generate_all_comparisons() with your results")
