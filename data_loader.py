"""
Data loader module for loading evaluation datasets
Handles GSM8K, CommonsenseQA, and SVAMP datasets
"""

import re
from datasets import load_dataset
from typing import Tuple, List, Any


def load_gsm8k_samples(num_samples: int = 100) -> Tuple[List[str], List[str]]:
    """
    Load GSM8K dataset samples for mathematical reasoning evaluation
    
    Args:
        num_samples (int): Number of test samples to load (default: 100)
        
    Returns:
        questions_list (list[str]): List of mathematical question texts
        ground_truth_answers (list[str]): List of correct answer strings
    """
    try:
        # Load GSM8K dataset from HuggingFace
        dataset = load_dataset("gsm8k", "main", split="test")
        
        # Limit to specified number of samples
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        questions_list = []  # Problem text list (list[str])
        ground_truth_answers = []  # Correct answer list (list[str])
        
        for sample in dataset:
            questions_list.append(sample["question"])
            # Extract final numeric answer from answer field
            answer_text = sample["answer"]  # Format: "... answer\n#### <number>"
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


def load_commonsenseqa_samples(num_samples: int = 100) -> Tuple[List[str], List[str]]:
    """
    Load CommonsenseQA dataset samples for commonsense reasoning evaluation
    
    Args:
        num_samples (int): Number of test samples to load (default: 100)
        
    Returns:
        questions_list (list[str]): List of multiple-choice question texts
        ground_truth_answers (list[str]): List of correct answer strings
    """
    try:
        # Load CommonsenseQA dataset from HuggingFace
        dataset = load_dataset("commonsenseqa", split="validation")
        
        # Limit to specified number of samples
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        questions_list = []  # Question text list (list[str])
        ground_truth_answers = []  # Correct answer list (list[str])
        
        for sample in dataset:
            question_text = sample["question"]  # Main question
            choices = sample["choices"]["text"]  # List of choice texts
            choices_labels = sample["choices"]["label"]  # Labels (A, B, C, D, E)
            
            # Format question with options
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


def load_svamp_samples(num_samples: int = 100) -> Tuple[List[str], List[str]]:
    """
    Load SVAMP dataset samples for symbolic reasoning evaluation
    
    Args:
        num_samples (int): Number of test samples to load (default: 100)
        
    Returns:
        questions_list (list[str]): List of arithmetic word problem texts
        ground_truth_answers (list[str]): List of correct numerical answers
    """
    try:
        # Load SVAMP dataset from HuggingFace
        dataset = load_dataset("svamp", split="test")
        
        # Limit to specified number of samples
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        questions_list = []  # Problem text list (list[str])
        ground_truth_answers = []  # Correct answer list (list[str])
        
        for sample in dataset:
            questions_list.append(sample["Body"] + " " + sample["Question"])
            ground_truth_answers.append(str(sample["Answer"]))
        
        print(f"[SVAMP] Loaded {len(questions_list)} samples")
        return questions_list, ground_truth_answers
        
    except Exception as e:
        print(f"Error loading SVAMP: {e}")
        return [], []


def load_all_datasets(num_samples: int = 100) -> dict:
    """
    Load all three evaluation datasets
    
    Args:
        num_samples (int): Number of samples per dataset (default: 100)
        
    Returns:
        datasets_dict (dict): Dictionary containing all loaded datasets
            Structure:
            {
                "gsm8k": {"questions": [...], "answers": [...]},
                "commonsenseqa": {"questions": [...], "answers": [...]},
                "svamp": {"questions": [...], "answers": [...]}
            }
    """
    datasets_dict = {}  # Container for all datasets (dict[str, dict])
    
    # Load GSM8K dataset
    gsm8k_questions, gsm8k_answers = load_gsm8k_samples(num_samples)
    datasets_dict["gsm8k"] = {
        "questions_list": gsm8k_questions,
        "ground_truth_answers": gsm8k_answers
    }
    
    # Load CommonsenseQA dataset
    cqa_questions, cqa_answers = load_commonsenseqa_samples(num_samples)
    datasets_dict["commonsenseqa"] = {
        "questions_list": cqa_questions,
        "ground_truth_answers": cqa_answers
    }
    
    # Load SVAMP dataset
    svamp_questions, svamp_answers = load_svamp_samples(num_samples)
    datasets_dict["svamp"] = {
        "questions_list": svamp_questions,
        "ground_truth_answers": svamp_answers
    }
    
    return datasets_dict
