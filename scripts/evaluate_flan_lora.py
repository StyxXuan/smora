#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

# 定义设备
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_for_task(base_model_path, lora_path, dtype=torch.bfloat16):
    """加载基础模型和任务特定的LoRA"""
    print(f"加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
    
    print(f"加载LoRA适配器: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    return model

def load_tokenizer(model_path):
    """加载分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

def evaluate_tasks(
    base_model_path,
    lora_models_dir,
    data_path,
    output_dir,
    batch_size=8,
    max_length=1024,
    max_new_tokens=100,
    temperature=0.1,
    tasks=None
):
    """对每个任务使用对应的LoRA模型进行评估"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 结果文件路径
    results_file = os.path.join(output_dir, "evaluation_results.json")
    
    # 如果结果文件已存在，则加载现有结果
    all_results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            print(f"加载了现有结果文件，包含 {len(all_results)} 个评估结果")
        except Exception as e:
            print(f"尝试加载现有结果文件时出错: {e}")
    
    # 加载数据集
    print(f"加载数据集: {data_path}")
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    else:
        raise ValueError("目前只支持JSON格式的数据集")
    
    # 如果数据集是列表形式，按task分组
    if isinstance(dataset, list):
        grouped_data = {}
        for item in dataset:
            task = item.get("task", "default_task")
            if task not in grouped_data:
                grouped_data[task] = []
            grouped_data[task].append(item)
        dataset = grouped_data
    
    # 确定要评估的任务
    all_tasks = list(dataset.keys())
    if tasks:
        tasks_to_eval = [task for task in tasks if task in all_tasks]
        if not tasks_to_eval:
            raise ValueError(f"指定的任务 {tasks} 在数据集中未找到")
    else:
        tasks_to_eval = all_tasks
    
    print(f"将评估以下任务: {tasks_to_eval}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./LLaMA-Factory/results/exp_res_flan/llama2-7b-srmole-64-8-epoch3-v2")
    
    # 跟踪已评估的任务，避免重复评估
    evaluated_tasks = set([result.get("task") for result in all_results if "task" in result])
    print(f"已评估过的任务: {evaluated_tasks}")
    
    # 逐个任务评估
    for task_name in tasks_to_eval:
        # 如果任务已评估过且有结果，则跳过
        if task_name in evaluated_tasks:
            print(f"任务 {task_name} 已经评估过，跳过")
            continue
            
        task_data = dataset[task_name]
        print(f"评估任务 {task_name}，样本数量: {len(task_data)}")
        
        # 查找对应的任务特定LoRA
        task_lora_path = os.path.join(lora_models_dir, f"standard_lora_{task_name}")
        if not os.path.exists(task_lora_path):
            print(f"警告: 找不到任务 {task_name} 的LoRA模型，跳过该任务")
            continue
        
        # 加载模型
        model = load_model_for_task(base_model_path, task_lora_path)
        
        # 创建任务输出目录
        task_output_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # 该任务的评估结果
        task_results = []
        
        # 批量处理样本
        for i in range(0, len(task_data)):
            sample = task_data[i]
            input_text = sample["inputs"]
            target = sample["targets"]
            metric = sample["metric"]
            domain = sample["domain"]
            task = sample["task"]
            # 准备输入
            # prompt = f"[INST]{input_text}[/INST]"
            conversation = [
                {
                    "role": "user",
                    "content": input_text
                }
            ]
            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,  # 这里先不直接 tokenize
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(device)

            # 生成回复
            with torch.no_grad():
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,
                    temperature=0.01,
                )

            output = output.cpu()
            generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)

            # 从生成的文本中提取助手的回复
            try:
                # 尝试提取Assistant部分的回复
                assistant_response = generated_answer.split("[/INST]")[-1].strip()
            except:
                # 如果提取失败，使用完整的生成文本
                assistant_response = generated_answer.strip()


            print(f"generated_answer: {assistant_response} \n")
            print(f"expected_answer: {target} \n")
            print('-' * 80)
            # 存储结果
            result = {
                "input": input_text,
                "target": target,
                "prediction": assistant_response,
                "metric": metric,
                "domain": domain,
                "task": task
            }                
            task_results.append(result)
                        
        # 将该任务的结果添加到总结果中
        all_results.extend(task_results)
        
        # 在每个任务完成后保存一次结果
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"任务 {task_name} 评估完成，结果已更新到 {results_file}")
        
        # 释放内存
        del model
        torch.cuda.empty_cache()
    
    print(f"所有任务评估完成，共 {len(all_results)} 个结果保存到 {results_file}")

def main():
    parser = argparse.ArgumentParser(description="使用任务特定LoRA评估不同任务")
    parser.add_argument("--base_model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_models_dir", type=str, required=True, help="包含任务特定LoRA模型的目录")
    parser.add_argument("--json_path", type=str, required=True, help="数据集JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="保存结果的目录")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="要评估的特定任务")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--max_length", type=int, default=1024, help="最大输入长度")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="最大生成标记数")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度")
    
    args = parser.parse_args()
    
    evaluate_tasks(
        args.base_model_path,
        args.lora_models_dir,
        args.json_path,
        args.output_dir,
        args.batch_size,
        args.max_length,
        args.max_new_tokens,
        args.temperature,
        args.tasks
    )

if __name__ == "__main__":
    main() 