import re
import json
import yaml
from typing import List, Dict, Optional, Literal, Any
from core.state import AgentState
from core.llm_interface import UnifiedLLMClient
from utils.prompt_manager import PromptManager

# === 常量定义 ===
ASSERTION_LABELS = ["present", "absent", "uncertain", "family", "historical"]
AssertionLabel = Literal["present", "absent", "uncertain", "family", "historical"]

# === 1. Rule Engine (逻辑保持不变) ===
class RuleEngine:
    def __init__(self, neg_window: int = 8):
        self.neg_window = neg_window
        self.family_cues = [
            "family history", "fhx", "mother has", "father has",
            "sister has", "brother has", "parents have"
        ]
        self.hist_cues = [
            "history of", "hx of", "previous", "prior",
            "remote", "in childhood", "as a child"
        ]
        self.uncertain_cues = [
            "possible", "possibly", "suspected", "suspicion of",
            "likely", "cannot exclude", "rule out", "r/o", "might be"
        ]
        self.neg_tokens = ["no", "denies", "without", "never", "none"]
        self.neg_phrases = ["no evidence of", "negative for", "free of"]

    def apply(self, mention: str, context: str) -> Optional[AssertionLabel]:
        text = context.lower()
        m = mention.lower()

        tokens = text.split()
        m_idx = None

        m_first = m.split()[0]
        for i, tok in enumerate(tokens):
            if m_first in tok:
                m_idx = i
                break

        # Negation
        for phrase in self.neg_phrases:
            if phrase in text and m in text:
                if text.index(phrase) < text.index(m):
                    return "absent"

        if m_idx is not None:
            for i, tok in enumerate(tokens):
                if tok in self.neg_tokens and 0 <= (m_idx - i) <= self.neg_window:
                    return "absent"

        # Family
        if m_idx is not None:
            for cue in self.family_cues:
                cue_tokens = cue.split()
                clen = len(cue_tokens)
                for i in range(max(0, m_idx - self.neg_window), m_idx + 1):
                    if i + clen <= len(tokens) and tokens[i:i+clen] == cue_tokens:
                        return "family"

        # Historical
        if m_idx is not None:
            for cue in self.hist_cues:
                cue_tokens = cue.split()
                clen = len(cue_tokens)
                for i in range(max(0, m_idx - self.neg_window), m_idx + 1):
                    if i + clen <= len(tokens) and tokens[i:i+clen] == cue_tokens:
                        return "historical"

        # Uncertainty
        for cue in self.uncertain_cues:
            if cue in text:
                return "uncertain"

        return None


# === 2. Assertion LLM Wrapper ===
class AssertionLLM:
    _instance = None
    
    def __init__(self):
        self.pm = PromptManager.get()
        # 替换 GlobalLLM 为统一接口
        self.llm_client = UnifiedLLMClient.get()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = AssertionLLM()
        return cls._instance

    def _extract_json(self, text: str) -> Dict:
        """
        保留您鲁棒的 JSON 提取逻辑
        """
        # 1. 移除 DeepSeek 的 <think> 标签
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # 2. 移除 Markdown
        text = re.sub(r'```[a-zA-Z]*', '', text).replace('```', '')
        
        candidates = []
        stack = 0
        start_idx = -1
        
        # 3. 堆栈遍历
        for i, char in enumerate(text):
            if char == '{':
                if stack == 0: start_idx = i
                stack += 1
            elif char == '}':
                if stack > 0:
                    stack -= 1
                    if stack == 0:
                        try:
                            obj = json.loads(text[start_idx : i+1], strict=False)
                            candidates.append(obj)
                        except: pass
        
        if candidates: return candidates[-1] # 返回最后一个合法 JSON
        
        # 4. 暴力兜底
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end+1], strict=False)
        except: pass
        
        return {} # 返回空字典表示失败

    def infer(self, mention: str, context: str) -> Dict:
        # === 核心分支逻辑 ===
        # 如果 LLM 不可用（模式为 no_llm），直接返回默认值
        if not self.llm_client.is_enabled():
            return {
                "label": "present",  # 默认假设
                "confidence": 0.5,
                "reason": "no_llm_fallback",
                "raw_output": "LLM disabled"
            }

        prompt = self.pm.render(
            stage='assertion', 
            template_name='classification',
            mention=mention,
            context=context
        )

        try:
            # 使用统一接口调用
            response_text = self.llm_client.chat(prompt, max_new_tokens=300)
            
            # 解析
            parsed = self._extract_json(response_text)
            
            # 结果校验
            label = parsed.get("label", "present")
            if label not in ASSERTION_LABELS: label = "present"
            
            return {
                "label": label,
                "confidence": float(parsed.get("confidence", 0.55)),
                "reason": parsed.get("reason", "llm_choice"),
                "raw_output": response_text # 保留原始输出用于调试
            }
        except Exception as e:
            return {
                "label": "present", 
                "confidence": 0.5, 
                "reason": f"error: {str(e)}", 
                "raw_output": ""
            }

# === 3. Tool Function ===

def tool_assertion(state: AgentState):
    """
    Agent 节点：断言分类
    逻辑：Rule -> if None -> LLM (if enabled) -> else Default
    """
    rule_engine = RuleEngine() # 规则引擎轻量，可直接实例化
    llm_engine = AssertionLLM.get()
    
    # 辅助：构建上下文窗口 (前一句 + 当前句 + 后一句)
    sent_map = state.get("metadata", {}).get("sentence_map", [])
    # 转为 dict 方便查找: sent_id -> text
    text_map = {s["sent_id"]: s["expanded"] for s in sent_map}
    
    updated_phenotypes = []
    
    for p in state['phenotypes']:
        # 如果 C 阶段没找到 ID，通常这里可以直接跳过或默认为 present
        
        mention = p["span_text"]
        sid = p.get("sent_id", 0)
        
        # 构建宽上下文
        prev_s = text_map.get(sid - 1, "")
        curr_s = text_map.get(sid, "")
        next_s = text_map.get(sid + 1, "")
        context_window = f"{prev_s} {curr_s} {next_s}".strip()
        
        # 1. Rule Engine (始终运行，不受 mode 影响)
        rule_res = rule_engine.apply(mention, context_window)
        
        if rule_res:
            p["assertion"] = rule_res
            p["assertion_method"] = "rule"
            p["assertion_conf"] = 0.9
        else:
            # 2. LLM (内部会检查 mode)
            llm_res = llm_engine.infer(mention, context_window)
            
            p["assertion"] = llm_res["label"]
            p["assertion_method"] = "llm" if llm_res["reason"] != "no_llm_fallback" else "default"
            p["assertion_conf"] = llm_res["confidence"]
            
            # 日志记录 (State Log)
            if p["assertion_method"] == "llm":
                state['logs'].append(f"Assertion(LLM): {mention} -> {llm_res['label']}")
                
                # [可选] 记录到 llm_trace
                if 'llm_trace' not in state: state['llm_trace'] = []
                state['llm_trace'].append({
                    "stage": "D",
                    "note_id": state["note_id"],
                    "mention": mention,
                    "context": context_window,
                    "output": llm_res
                })
            
        updated_phenotypes.append(p)
        
    state['phenotypes'] = updated_phenotypes
    return state