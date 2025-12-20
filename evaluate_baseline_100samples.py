import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import gc

def evaluate_baseline_model(model_name, datasets):
    """
    评估 Base 模型在多个数据集上的表现
    
    Args:
        model_name: 模型名称
        datasets: 数据字典 {dataset_name: {questions_list, ground_truth_answers}}
    
    Returns:
        results: 评估结果字典
    """
    
    print(f"[INFO] 加载模型: {model_name}\n")
    
    # 加载模型和分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ 模型加载成功\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None
    
    # 初始化结果
    results = {
        "model": f"Baseline ({model_name})",
        "timestamp": datetime.now().isoformat(),
        "total_samples": 0,
        "total_correct": 0,
        "overall_accuracy": 0.0,
        "datasets": []
    }
    
    # 评估每个数据集
    for dataset_name, dataset_info in datasets.items():
        print(f"[EVALUATION] {dataset_name.upper()}")
        print("-" * 80)
        
        questions = dataset_info["questions_list"]
        answers = dataset_info["ground_truth_answers"]
        
        correct_count = 0
        total_count = len(questions)
        
        # 评估每个问题
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            try:
                # 准备输入
                inputs = tokenizer(question, return_tensors="pt").to(model.device)
                
                # 生成答案
                with torch.no_grad():
                    output = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask'),
                        max_new_tokens=128,
                        num_beams=1,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                # 解码答案
                generated_text = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                
                # 检查答案是否正确
                is_correct = answer.lower().strip() in generated_text.lower().strip()
                
                if is_correct:
                    correct_count += 1
                
                # 显示进度
                if (idx + 1) % 10 == 0:
                    print(f"  Progress: {idx + 1}/{total_count}")
                    
            except Exception as e:
                print(f"  ✗ Error on question {idx + 1}: {e}")
                continue
        
        # 计算数据集准确率
        dataset_accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        dataset_result = {
            "dataset": dataset_name,
            "correct_count": correct_count,
            "total_count": total_count,
            "accuracy": dataset_accuracy
        }
        
        results["datasets"].append(dataset_result)
        results["total_samples"] += total_count
        results["total_correct"] += correct_count
        
        print(f"  ✓ {dataset_name}: {dataset_accuracy*100:.2f}% ({correct_count}/{total_count})")
        print()
    
    # 计算总体准确率
    results["overall_accuracy"] = results["total_correct"] / results["total_samples"] if results["total_samples"] > 0 else 0.0
    
    # 保存结果
    output_file = '/content/lora/baseline_results_100samples.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ 结果已保存到 {output_file}")
    
    # 清理内存
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results
