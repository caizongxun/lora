"""
Example: Using indexed question selection for comparison analysis
Demonstrates how to select specific questions where Base fails but LoRA succeeds
"""

from data_loader_indexed import load_all_datasets_indexed

def example_usage():
    """
    Three ways to use the indexed data loader
    """
    
    # Method 1: Load specific indices for each dataset
    print("Method 1: Load specific question indices")
    print("="*80)
    
    question_indices = {
        "gsm8k": [0, 5, 15],           # Load questions at indices 0, 5, 15
        "commonsenseqa": [2, 8, 20],   # Load questions at indices 2, 8, 20
        "svamp": [1, 3, 7]             # Load questions at indices 1, 3, 7
    }
    
    datasets_indexed = load_all_datasets_indexed(question_indices=question_indices)
    
    for dataset_name, data in datasets_indexed.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Loaded indices: {data['indices']}")
        print(f"  Number of questions: {len(data['questions_list'])}")
        if data['questions_list']:
            print(f"  First question: {data['questions_list'][0][:100]}...")
    
    # Method 2: Load first N samples from each dataset
    print("\n" + "="*80)
    print("Method 2: Load first N samples (default behavior)")
    print("="*80)
    
    datasets_first_n = load_all_datasets_indexed(num_samples=3)
    
    for dataset_name, data in datasets_first_n.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Loaded indices: {data['indices']}")
        print(f"  Number of questions: {len(data['questions_list'])}")
    
    # Method 3: For comparison - pick questions where Base fails
    print("\n" + "="*80)
    print("Method 3: Pick indices where Base model failed (for after evaluation)")
    print("="*80)
    
    # After running evaluation, you can identify failed questions:
    # base_results["gsm8k"]["predictions"] != base_results["gsm8k"]["ground_truth_answers"]
    # Then select those indices:
    
    failed_indices_example = {
        "gsm8k": [2, 7, 12],              # Indices where Base failed
        "commonsenseqa": [5, 10],
        "svamp": [3, 8]
    }
    
    print(f"\nFailed question indices (example):")
    print(f"  {failed_indices_example}")
    
    datasets_failed = load_all_datasets_indexed(question_indices=failed_indices_example)
    
    print(f"\nLoaded failed questions for re-evaluation:")
    for dataset_name, data in datasets_failed.items():
        print(f"  {dataset_name}: {len(data['questions_list'])} questions")


if __name__ == "__main__":
    example_usage()
