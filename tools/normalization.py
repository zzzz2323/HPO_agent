import re
import yaml
from typing import List, Dict, Any
from core.state import AgentState
from core.llm_interface import UnifiedLLMClient
from utils.prompt_manager import PromptManager

# 加载配置
def load_config():
    # 假设配置文件路径正确
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)["normalization"]

CONFIG = load_config()

class Normalizer:
    _instance = None

    def __init__(self):
        self.pm = PromptManager.get()
        self.llm_client = UnifiedLLMClient.get()
        
        # 决策逻辑：只有当配置文件允许 LLM 且系统模式不是 "no_llm" 时，才真正启用 LLM
        # 如果系统模式是 "no_llm"，无论 config 怎么写，都强制为 False
        self.enable_llm_normalize = CONFIG["enable_llm"] and self.llm_client.is_enabled()
        self.llm_threshold = CONFIG["llm_threshold"]

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = Normalizer()
        return cls._instance

    def _should_trigger_llm(self, ranked_candidates: List[Dict]) -> bool:
        if len(ranked_candidates) < 2:
            return False
        
        top1 = ranked_candidates[0]
        top2 = ranked_candidates[1]
        
        # 完美匹配不触发 (分数为 1.001 的那种)
        if top1["score"] > 1.0:
            return False

        # 分数差距小于阈值，说明 BGE 犹豫不决，触发 LLM
        return abs(top1["score"] - top2["score"]) < self.llm_threshold

    def _llm_choose(self, mention: str, context: str, candidates: List[Dict]) -> Dict:
        # 准备候选列表字符串
        # 截取前 10 个候选
        top_cands = candidates[:10]
        cand_list_str = "\n".join(
            f"{i+1}. {c['id']} {c['label']}"
            for i, c in enumerate(top_cands)
        )
        cand_ids = {c["id"] for c in top_cands}

        prompt = self.pm.render(
            stage='normalization',
            template_name='disambiguation',
            mention=mention,
            context=context,
            candidate_list=cand_list_str
        )

        # 默认回退结果
        fallback_result = {
            "id": candidates[0]["id"],
            "method": "llm_fallback",
            "confidence": candidates[0]["score"],
            "reason": "llm_failed"
        }

        # 调用统一 LLM 接口
        # 注意：max_new_tokens 对于归一化这种短输出任务不需要太大
        response_text = self.llm_client.chat(prompt, max_new_tokens=64)
        
        if not response_text:
            return fallback_result

        try:
            text = response_text.strip()

            # 解析逻辑：按行拆分，找 HP ID
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            hp_ordered = []
            seen = set()
            
            for line in lines:
                # 提取 HP:xxxxxxx
                for m in re.findall(r"HP:\d{7}", line):
                    if m in cand_ids and m not in seen:
                        seen.add(m)
                        hp_ordered.append(m)
            
            if hp_ordered:
                chosen_id = hp_ordered[-1]
                return {
                    "id": chosen_id,
                    "method": "llm",
                    "confidence": 0.95, # LLM 选中的给高置信度
                    "reason": f"llm_selected: {text[:50]}..."
                }
            
            # 全文扫描兜底
            hp_id_full = re.search(r"HP:\d{7}", text)
            if hp_id_full:
                found_id = hp_id_full.group(0)
                if found_id in cand_ids:
                    return {
                        "id": found_id,
                        "method": "llm_fulltext_scan",
                        "confidence": 0.9,
                        "reason": "llm_scan"
                    }

            return fallback_result

        except Exception as e:
            print(f"[Norm Tool Error] {e}")
            return fallback_result

    def process_phenotype(self, phenotype: Dict, processed_text: str) -> Dict:
        """
        处理单个表型实体
        """
        candidates = phenotype.get("candidates", [])
        if not candidates:
            return {"final_id": None, "method": "no_candidate", "confidence": 0.0, "reason": "no candidates"}

        # 确保候选已排序
        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        phenotype["candidates"] = ranked # 更新排序后的列表

        # 1. 判断是否使用 LLM
        use_llm = False
        if self.enable_llm_normalize and self._should_trigger_llm(ranked):
            use_llm = True

        result = {}
        if use_llm:
            # 查找 Span 所在的句子上下文
            # 这里简化为直接传入 processed_text (或者是调用方传进来的 context_sent)
            context = processed_text 
            
            llm_res = self._llm_choose(phenotype["span_text"], context, ranked)
            result = {
                "final_id": llm_res["id"],
                "method": llm_res["method"],
                "confidence": llm_res["confidence"],
                "reason": llm_res["reason"]
            }
        else:
            # Baseline Confident
            top1 = ranked[0]
            result = {
                "final_id": top1["id"],
                "method": "baseline_confident",
                "confidence": top1["score"],
                "reason": "score_gap_clear"
            }
        
        return result

# === Tool Function ===

def tool_normalize(state: AgentState):
    """
    Agent 节点函数：归一化
    """
    normalizer = Normalizer.get()
    
    # 遍历处理所有提取出的表型
    updated_phenotypes = []
    
    # 为了优化上下文查找，我们可以先构建 sent_id -> sentence text 的映射
    # 从 metadata 取
    sent_map = {s["sent_id"]: s["expanded"] for s in state.get("metadata", {}).get("sentence_map", [])}
    
    for p in state['phenotypes']:
        # 获取当前实体的上下文句子
        sent_id = p.get("sent_id", 0)
        # 如果找不到句子，回退到全文
        context_sent = sent_map.get(sent_id, state["processed_text"])
        
        # 执行核心逻辑
        norm_result = normalizer.process_phenotype(p, context_sent)
        
        # 更新字段
        p["final_id"] = norm_result["final_id"]
        p["norm_method"] = norm_result["method"] # 记录方法
        p["confidence"] = norm_result["confidence"]
        
        updated_phenotypes.append(p)
        
        # 日志 (仅记录 LLM 触发的情况)
        if norm_result["method"].startswith("llm"):
             state['logs'].append(f"Norm: LLM used for '{p['span_text']}' -> {p['final_id']}")

    state['phenotypes'] = updated_phenotypes
    return state