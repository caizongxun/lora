"""
Data loader module for loading evaluation datasets
Handles GSM8K, CommonsenseQA, and SVAMP datasets
"""

import re
from datasets import load_dataset
from typing import Tuple, List, Any


def load_gsm8k_samples(num_samples: int = 100) -> Tuple[List[str], List[str]]:
    """Load GSM8K dataset samples"""
    try:
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


def load_commonsenseqa_samples(num_samples: int = 100) -> Tuple[List[str], List[str]]:
    """Load CommonsenseQA dataset samples"""
    try:
        # Fixed: correct dataset name is 'tau/commonsense_qa' or just 'commonsense_qa'
        # Trying both common identifiers
        try:
            dataset = load_dataset("commonsense_qa", split="validation", trust_remote_code=True)
        except:
            dataset = load_dataset("tau/commonsense_qa", split="validation", trust_remote_code=True)
        
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


def load_svamp_samples(num_samples: int = 100) -> Tuple[List[str], List[str]]:
    """Load SVAMP dataset samples"""
    try:
        # Fixed: SVAMP is often hosted under specific organizations or names
        # Trying ChilleD/SVAMP as it is a reliable mirror
        try:
            dataset = load_dataset("ChilleD/SVAMP", split="test", trust_remote_code=True)
        except:
            dataset = load_dataset("rkcosner/SVAMP", split="test", trust_remote_code=True)
        
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


def load_all_datasets(num_samples: int = 100) -> dict:
    """Load all datasets"""
    datasets_dict = {}
    
    questions, answers = load_gsm8k_samples(num_samples)
    if questions:
        datasets_dict["gsm8k"] = {
            "questions_list": questions,
            "ground_truth_answers": answers
        }
    
    questions, answers = load_commonsenseqa_samples(num_samples)
    if questions:
        datasets_dict["commonsenseqa"] = {
            "questions_list": questions,
            "ground_truth_answers": answers
        }
    
    questions, answers = load_svamp_samples(num_samples)
    if questions:
        datasets_dict["svamp"] = {
            "questions_list": questions,
            "ground_truth_answers": answers
        }
    
    return datasets_dict
