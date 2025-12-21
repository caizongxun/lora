"""
Data loader module with support for indexed question selection
Allows specifying exact question indices for comparison analysis
Handles GSM8K, CommonsenseQA, and SVAMP datasets
"""

import signal

if not hasattr(signal, 'SIGALRM'):
    signal.SIGALRM = 14
if not hasattr(signal, 'SIGCHLD'):
    signal.SIGCHLD = 17
if not hasattr(signal, 'SIGUSR1'):
    signal.SIGUSR1 = 10
if not hasattr(signal, 'SIGUSR2'):
    signal.SIGUSR2 = 12

import re
from datasets import load_dataset
from typing import Tuple, List, Any, Dict, Optional
from config import DATASET_CONFIG


def load_gsm8k_samples(
    num_samples: int = None,
    question_indices: List[int] = None
) -> Tuple[List[str], List[str], List[int]]:
    """
    Load GSM8K dataset samples
    
    Args:
        num_samples: Number of samples to load (overrides config if provided)
        question_indices: Specific indices to select (e.g., [0, 5, 10])
                         If provided, num_samples is ignored
    
    Returns:
        Tuple of (questions_list, ground_truth_answers, used_indices)
    """
    try:
        if num_samples is None:
            num_samples = DATASET_CONFIG["gsm8k"]["num_samples"]
            
        dataset = load_dataset("gsm8k", "main", split="test")
        
        if question_indices is not None:
            print(f"[INFO] GSM8K: Using specified indices: {question_indices}")
            selected_indices = question_indices
            if max(selected_indices) >= len(dataset):
                print(f"[WARNING] Index {max(selected_indices)} exceeds dataset size {len(dataset)}")
                selected_indices = [i for i in selected_indices if i < len(dataset)]
            dataset = dataset.select(selected_indices)
        else:
            selected_indices = list(range(min(num_samples, len(dataset))))
            if num_samples and len(dataset) > num_samples:
                dataset = dataset.select(selected_indices)
        
        questions_list = []
        ground_truth_answers = []
        
        for sample in dataset:
            questions_list.append(sample["question"])
            answer_text = sample["answer"]
            match = re.search(r"####\s*(-?\d+)", answer_text)
            if match:
                ground_truth_answers.append(match.group(1))
            else:
                ground_truth_answers.append("0")
        
        print(f"[SUCCESS] GSM8K: Loaded {len(questions_list)} samples (indices: {selected_indices})")
        return questions_list, ground_truth_answers, selected_indices
        
    except Exception as e:
        print(f"[ERROR] Failed to load GSM8K: {e}")
        return [], [], []


def load_commonsenseqa_samples(
    num_samples: int = None,
    question_indices: List[int] = None
) -> Tuple[List[str], List[str], List[int]]:
    """
    Load CommonsenseQA dataset samples
    
    Args:
        num_samples: Number of samples to load (overrides config if provided)
        question_indices: Specific indices to select (e.g., [0, 5, 10])
    
    Returns:
        Tuple of (questions_list, ground_truth_answers, used_indices)
    """
    try:
        if num_samples is None:
            num_samples = DATASET_CONFIG["commonsenseqa"]["num_samples"]
            
        dataset = None
        for dataset_name in ["commonsense_qa", "tau/commonsense_qa"]:
            try:
                dataset = load_dataset(dataset_name, split="validation")
                break
            except:
                continue
        
        if dataset is None:
            print(f"[WARNING] CommonsenseQA dataset not found")
            return [], [], []
        
        if question_indices is not None:
            print(f"[INFO] CommonsenseQA: Using specified indices: {question_indices}")
            selected_indices = question_indices
            if max(selected_indices) >= len(dataset):
                print(f"[WARNING] Index {max(selected_indices)} exceeds dataset size {len(dataset)}")
                selected_indices = [i for i in selected_indices if i < len(dataset)]
            dataset = dataset.select(selected_indices)
        else:
            selected_indices = list(range(min(num_samples, len(dataset))))
            if num_samples and len(dataset) > num_samples:
                dataset = dataset.select(selected_indices)
        
        questions_list = []
        ground_truth_answers = []
        
        for sample in dataset:
            question_text = sample["question"]
            choices = sample["choices"]["text"]
            choices_labels = sample["choices"]["label"]
            
            formatted_question = f"{question_text}\nOptions:\n"
            for label, choice in zip(choices_labels, choices):
                formatted_question += f"{label}. {choice}\n"
            
            questions_list.append(formatted_question)
            ground_truth_answers.append(sample["answerKey"])
        
        print(f"[SUCCESS] CommonsenseQA: Loaded {len(questions_list)} samples (indices: {selected_indices})")
        return questions_list, ground_truth_answers, selected_indices
        
    except Exception as e:
        print(f"[ERROR] Failed to load CommonsenseQA: {e}")
        return [], [], []


def load_svamp_samples(
    num_samples: int = None,
    question_indices: List[int] = None
) -> Tuple[List[str], List[str], List[int]]:
    """
    Load SVAMP dataset samples
    
    Args:
        num_samples: Number of samples to load (overrides config if provided)
        question_indices: Specific indices to select (e.g., [0, 5, 10])
    
    Returns:
        Tuple of (questions_list, ground_truth_answers, used_indices)
    """
    try:
        if num_samples is None:
            num_samples = DATASET_CONFIG["svamp"]["num_samples"]
        
        dataset = None
        dataset_names = [
            ("ChilleD/SVAMP", "test"),
            ("rkcosner/SVAMP", "test"),
            ("svamp", "test"),
        ]
        
        for dataset_name, split in dataset_names:
            try:
                print(f"[INFO] Attempting to load SVAMP from: {dataset_name}")
                dataset = load_dataset(dataset_name, split=split)
                print(f"[SUCCESS] Successfully loaded SVAMP from: {dataset_name}")
                break
            except Exception as e:
                print(f"[DEBUG] Failed to load from {dataset_name}: {type(e).__name__}")
                continue
        
        if dataset is None:
            print(f"[WARNING] SVAMP dataset not found")
            return [], [], []
        
        if question_indices is not None:
            print(f"[INFO] SVAMP: Using specified indices: {question_indices}")
            selected_indices = question_indices
            if max(selected_indices) >= len(dataset):
                print(f"[WARNING] Index {max(selected_indices)} exceeds dataset size {len(dataset)}")
                selected_indices = [i for i in selected_indices if i < len(dataset)]
            dataset = dataset.select(selected_indices)
        else:
            selected_indices = list(range(min(num_samples, len(dataset))))
            if num_samples and len(dataset) > num_samples:
                dataset = dataset.select(selected_indices)
        
        questions_list = []
        ground_truth_answers = []
        
        for sample in dataset:
            try:
                if "Body" in sample and "Question" in sample:
                    questions_list.append(sample["Body"] + " " + sample["Question"])
                    ground_truth_answers.append(str(sample["Answer"]))
                elif "question" in sample:
                    questions_list.append(sample["question"])
                    ground_truth_answers.append(str(sample.get("answer", "0")))
            except:
                continue
        
        if questions_list:
            print(f"[SUCCESS] SVAMP: Loaded {len(questions_list)} samples (indices: {selected_indices})")
        else:
            print(f"[WARNING] SVAMP: No valid samples loaded")
            
        return questions_list, ground_truth_answers, selected_indices
        
    except Exception as e:
        print(f"[WARNING] Failed to load SVAMP: {e}")
        return [], [], []


def load_all_datasets_indexed(
    num_samples: int = None,
    question_indices: Optional[Dict[str, List[int]]] = None
) -> dict:
    """
    Load all datasets with support for indexed selection
    
    Args:
        num_samples: Number of samples per dataset (overrides config)
        question_indices: Dictionary mapping dataset names to index lists
                         Example: {
                             "gsm8k": [0, 2, 5],
                             "commonsenseqa": [1, 3],
                             "svamp": [0, 1]
                         }
    
    Returns:
        dict: Dictionary containing loaded datasets with indices
              Format: {
                  "dataset_name": {
                      "questions_list": [...],
                      "ground_truth_answers": [...],
                      "indices": [...]  # NEW: actual indices used
                  }
              }
    """
    datasets_dict = {}
    question_indices = question_indices or {}
    
    print("[INFO] Loading all datasets...\n")
    
    # Load GSM8K
    questions, answers, indices = load_gsm8k_samples(
        num_samples,
        question_indices.get("gsm8k")
    )
    if questions:
        datasets_dict["gsm8k"] = {
            "questions_list": questions,
            "ground_truth_answers": answers,
            "indices": indices
        }
    
    # Load CommonsenseQA
    questions, answers, indices = load_commonsenseqa_samples(
        num_samples,
        question_indices.get("commonsenseqa")
    )
    if questions:
        datasets_dict["commonsenseqa"] = {
            "questions_list": questions,
            "ground_truth_answers": answers,
            "indices": indices
        }
    
    # Load SVAMP
    questions, answers, indices = load_svamp_samples(
        num_samples,
        question_indices.get("svamp")
    )
    if questions:
        datasets_dict["svamp"] = {
            "questions_list": questions,
            "ground_truth_answers": answers,
            "indices": indices
        }
    else:
        print("[INFO] SVAMP will be skipped\n")
    
    print(f"\n[INFO] Total datasets loaded: {len(datasets_dict)}")
    print(f"[INFO] Available datasets: {list(datasets_dict.keys())}\n")
    
    return datasets_dict
