import json
import os
import argparse
import logging
from typing import List, Dict, Any
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 打印一下看看路径对不对
print(f"[Debug] Adding root path to sys.path: {current_dir}") 
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# 导入核心图定义
from core.graph import HPOAgentGraph
# 导入数据加载器 (支持 .tsv 和 .json)
from utils.dataloader import parse_tsv_data
from utils.export import (
    export_phenopacket, 
    export_to_owl, 
    trim_modifiers_for_retry, 
    drop_all_modifiers_for_retry
)
# 设置日志格式
log_save_path = "data/outputs/run.log"
os.makedirs(os.path.dirname(log_save_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_save_path, mode='w', encoding='utf-8'), # 写入文件
        logging.StreamHandler() # 同时打印到 Slurm 输出文件，方便查看进度
    ]
)

def save_results(results: List[Dict], output_path: str):
    """保存最终提取结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"Results saved to {output_path}")

def save_trace(trace_logs: List[Dict], trace_path: str):
    """保存 LLM 详细交互日志 (追加模式)"""
    if not trace_logs:
        return
        
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    with open(trace_path, 'a', encoding='utf-8') as f:
        for trace in trace_logs:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    logging.info(f"Appended {len(trace_logs)} trace records to {trace_path}")

def process_file(input_file: str, output_file: str, trace_file: str):
    # 1. 实例化 Agent
    # 这会触发单例加载 (OBO, BERT, LLM Client)，可能需要几秒钟到几分钟
    logging.info("Initializing HPO Agent Graph...")
    agent = HPOAgentGraph()
    
    # 2. 读取数据
    logging.info(f"Reading input file: {input_file}")
    data = []
    
    try:
        if input_file.endswith(".tsv") or input_file.endswith(".txt"):
            # 使用我们写的专用 TSV 解析器
            data = parse_tsv_data(input_file)
        elif input_file.endswith(".json") or input_file.endswith(".jsonl"):
            # 标准 JSON 加载
            with open(input_file, 'r', encoding='utf-8') as f:
                if input_file.endswith(".jsonl"):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use .json, .jsonl, or .tsv/.txt")
    except Exception as e:
        logging.error(f"Failed to load input file: {e}")
        return

    if not data:
        logging.warning("Input file is empty!")
        return

    logging.info(f"Loaded {len(data)} records. Starting processing...")

    final_results = []
    
    # 3. 循环处理每条记录
    for i, item in enumerate(data):
        note_id = item.get("note_id", f"unknown_{i}")
        text = item.get("text", "")
        
        if not text:
            logging.warning(f"Skipping record {note_id}: Empty text.")
            continue
            
        logging.info(f"Processing [{i+1}/{len(data)}] Note ID: {note_id}")
        
        # === 核心调用：运行 Agent ===
        try:
            final_state = agent.run(text, note_id)
            
            # [新增] --- 开始导出逻辑 (带局部容错循环) ---
            hpo_data = final_state.get("phenotypes", [])
            
            # 只有提取到了数据才导出
            if hpo_data:
                # 获取输出目录 (例如 data/outputs/)
                export_dir = os.path.dirname(output_file) 
                
                # Loop 1: Phenopacket 导出 (失败 -> 修剪 -> 重试)
                try:
                    export_phenopacket(hpo_data, note_id, export_dir)
                except Exception as e:
                    logging.warning(f"[{note_id}] Phenopacket export failed: {e}. Retrying with trimming...")
                    try:
                        trimmed_data = trim_modifiers_for_retry(hpo_data)
                        export_phenopacket(trimmed_data, note_id, export_dir)
                        logging.info(f"[{note_id}] Phenopacket retry successful.")
                    except Exception as e2:
                        logging.error(f"[{note_id}] Phenopacket retry failed: {e2}")

                # Loop 2: OWL 导出 (失败 -> 全丢弃 -> 重试)
                try:
                    export_to_owl(hpo_data, note_id, export_dir)
                except Exception as e:
                    logging.warning(f"[{note_id}] OWL export failed: {e}. Retrying with strict cleaning...")
                    try:
                        clean_data = drop_all_modifiers_for_retry(hpo_data)
                        export_to_owl(clean_data, note_id, export_dir)
                        logging.info(f"[{note_id}] OWL retry successful.")
                    except Exception as e2:
                        logging.error(f"[{note_id}] OWL retry failed: {e2}")
            # ---------------------------------------------

            # 4. 提取并保存结果 (这部分保持原样)
            output_record = {
                "note_id": final_state["note_id"],
                "raw_text": final_state["raw_text"],
                "gold_phenotypes": item.get("gold_phenotypes", []),
                "extracted_phenotypes": final_state["phenotypes"]
            }
            final_results.append(output_record)
            
            # 5. 实时保存 LLM Trace (这部分保持原样)
            if final_state.get("llm_trace"):
                save_trace(final_state["llm_trace"], trace_file)
                
        except Exception as e:
            logging.error(f"Error processing note {note_id}: {e}")

    # 6. 保存最终结构化结果
    save_results(final_results, output_file)
    logging.info("All processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HPO Extraction Agent")
    
    # 定义命令行参数
    parser.add_argument("--input", type=str, default="data/inputs/GSCplus_test_gold.tsv", 
                        help="Path to input file (.json, .jsonl, .tsv)")
    parser.add_argument("--output", type=str, default="data/outputs1/final_results.json", 
                        help="Path to save final structured output")
    parser.add_argument("--trace", type=str, default="data/outputs1/llm_trace.jsonl", 
                        help="Path to save LLM debug logs")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        exit(1)
        
    process_file(args.input, args.output, args.trace)