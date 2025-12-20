"""
Data loader module for loading evaluation datasets
Handles GSM8K, CommonsenseQA, and SVAMP datasets
With improved error handling for datasets that may not exist on Hub
"""

# FIX: Patch signal module BEFORE importing datasets to avoid multiprocess Windows bug
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
from typing import Tuple, List, Any
from config import DATASET_CONFIG


def load_gsm8k_samples(num_samples: int = None) -> Tuple[List[str], List[str]]:
    """Load GSM8K dataset samples"""
    try:
        # Use config num_samples if not explicitly provided
        if num_samples is None:
            num_samples = DATASET_CONFIG["gsm8k"]["num_samples"]
            
        dataset = load_dataset("gsm8k", "main", split="test")
        
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
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
        
        print(f"[SUCCESS] GSM8K: Loaded {len(questions_list)} samples")
        return questions_list, ground_truth_answers
        
    except Exception as e:
        print(f"[ERROR] Failed to load GSM8K: {e}")
        return [], []


def load_commonsenseqa_samples(num_samples: int = None) -> Tuple[List[str], List[str]]:
    """Load CommonsenseQA dataset samples"""
    try:
        # Use config num_samples if not explicitly provided
        if num_samples is None:
            num_samples = DATASET_CONFIG["commonsenseqa"]["num_samples"]
            
        # Try multiple dataset sources
        dataset = None
        for dataset_name in ["commonsense_qa", "tau/commonsense_qa"]:
            try:
                dataset = load_dataset(dataset_name, split="validation")
                break
            except:
                continue
        
        if dataset is None:
            print(f"[WARNING] CommonsenseQA dataset not found on any source")
            return [], []
        
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
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
        
        print(f"[SUCCESS] CommonsenseQA: Loaded {len(questions_list)} samples")
        return questions_list, ground_truth_answers
        
    except Exception as e:
        print(f"[ERROR] Failed to load CommonsenseQA: {e}")
        return [], []


def load_svamp_samples(num_samples: int = None) -> Tuple[List[str], List[str]]:
    """Load SVAMP dataset samples with graceful failure"""
    try:
        # Use config num_samples if not explicitly provided
        if num_samples is None:
            num_samples = DATASET_CONFIG["svamp"]["num_samples"]
        
        # Try multiple SVAMP dataset sources and names
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
            print(f"[WARNING] SVAMP dataset not found on any source")
            print(f"[WARNING] Skipping SVAMP evaluation")
            return [], []
        
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        questions_list = []
        ground_truth_answers = []
        
        for sample in dataset:
            # Handle different SVAMP formats
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
            print(f"[SUCCESS] SVAMP: Loaded {len(questions_list)} samples")
        else:
            print(f"[WARNING] SVAMP: No valid samples loaded")
            
        return questions_list, ground_truth_answers
        
    except Exception as e:
        print(f"[WARNING] Failed to load SVAMP: {e}")
        print(f"[WARNING] Skipping SVAMP evaluation")
        return [], []


def load_all_datasets(num_samples: int = None) -> dict:
    """
    Load all datasets using num_samples from DATASET_CONFIG
    Gracefully skips datasets that fail to load
    
    Args:
        num_samples (int, optional): Override config num_samples. 
                                    If None, use values from DATASET_CONFIG.
    
    Returns:
        dict: Dictionary containing successfully loaded datasets
              Only includes datasets that loaded at least some samples
    """
    datasets_dict = {}
    
    print("[INFO] Loading all datasets...\n")
    
    # Load GSM8K with its configured num_samples
    questions, answers = load_gsm8k_samples(num_samples)
    if questions:
        datasets_dict["gsm8k"] = {
            "questions_list": questions,
            "ground_truth_answers": answers
        }
    
    # Load CommonsenseQA with its configured num_samples
    questions, answers = load_commonsenseqa_samples(num_samples)
    if questions:
        datasets_dict["commonsenseqa"] = {
            "questions_list": questions,
            "ground_truth_answers": answers
        }
    
    # Load SVAMP with its configured num_samples (gracefully skip if fails)
    questions, answers = load_svamp_samples(num_samples)
    if questions:
        datasets_dict["svamp"] = {
            "questions_list": questions,
            "ground_truth_answers": answers
        }
    else:
        print("[INFO] SVAMP will be skipped in evaluation\n")
    
    print(f"\n[INFO] Total datasets loaded: {len(datasets_dict)}")
    print(f"[INFO] Available datasets: {list(datasets_dict.keys())}\n")
    
    return datasets_dict
