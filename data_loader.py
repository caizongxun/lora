"""
Data loader module for loading evaluation datasets
Handles GSM8K, CommonsenseQA, and SVAMP datasets
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
        
        print(f"[GSM8K] Loaded {len(questions_list)} samples")
        return questions_list, ground_truth_answers
        
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return [], []


def load_commonsenseqa_samples(num_samples: int = None) -> Tuple[List[str], List[str]]:
    """Load CommonsenseQA dataset samples"""
    try:
        # Use config num_samples if not explicitly provided
        if num_samples is None:
            num_samples = DATASET_CONFIG["commonsenseqa"]["num_samples"]
            
        # Fixed: Removed trust_remote_code=True as it's deprecated for standard datasets
        try:
            dataset = load_dataset("commonsense_qa", split="validation")
        except:
            dataset = load_dataset("tau/commonsense_qa", split="validation")
        
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
        
        print(f"[CommonsenseQA] Loaded {len(questions_list)} samples")
        return questions_list, ground_truth_answers
        
    except Exception as e:
        print(f"Error loading CommonsenseQA: {e}")
        return [], []


def load_svamp_samples(num_samples: int = None) -> Tuple[List[str], List[str]]:
    """Load SVAMP dataset samples"""
    try:
        # Use config num_samples if not explicitly provided
        if num_samples is None:
            num_samples = DATASET_CONFIG["svamp"]["num_samples"]
            
        # Fixed: Removed trust_remote_code=True as it's deprecated for standard datasets
        try:
            dataset = load_dataset("ChilleD/SVAMP", split="test")
        except:
            dataset = load_dataset("rkcosner/SVAMP", split="test")
        
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        questions_list = []
        ground_truth_answers = []
        
        for sample in dataset:
            questions_list.append(sample["Body"] + " " + sample["Question"])
            ground_truth_answers.append(str(sample["Answer"]))
        
        print(f"[SVAMP] Loaded {len(questions_list)} samples")
        return questions_list, ground_truth_answers
        
    except Exception as e:
        print(f"Error loading SVAMP: {e}")
        return [], []


def load_all_datasets(num_samples: int = None) -> dict:
    """
    Load all datasets using num_samples from DATASET_CONFIG
    
    Args:
        num_samples (int, optional): Override config num_samples. 
                                    If None, use values from DATASET_CONFIG.
    
    Returns:
        dict: Dictionary containing datasets with questions and answers
    """
    datasets_dict = {}
    
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
    
    # Load SVAMP with its configured num_samples
    questions, answers = load_svamp_samples(num_samples)
    if questions:
        datasets_dict["svamp"] = {
            "questions_list": questions,
            "ground_truth_answers": answers
        }
    
    return datasets_dict
