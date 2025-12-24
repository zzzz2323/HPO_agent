import re
import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from collections import defaultdict

from core.state import AgentState
from core.llm_interface import UnifiedLLMClient
from utils.prompt_manager import PromptManager

# === 加载配置 ===
def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)["postcoordination"]

CONFIG = load_config()

# === 常量定义 (保持不变) ===
VALUE_DOMAINS = {
    "severity": ["mild", "moderate", "severe"],
    "laterality": ["left", "right", "bilateral", "unilateral"],
    "onset": ["infancy", "childhood", "adult", "neonatal", "congenital"],
    "frequency": ["recurrent", "occasional", "frequent", "intermittent"],
    "progression": ["progressive", "worsening", "improving", "stable", "deteriorating"],
}

VALUE_SYNONYMS = {
    "severity": {
        "mildly": "mild", "slight": "mild", "moderately": "moderate", 
        "marked": "severe", "severely": "severe",
    },
    "laterality": {
        "left-sided": "left", "right-sided": "right", "both sides": "bilateral",
    },
    "onset": {
        "adult-onset": "adult", "since childhood": "childhood", "since infancy": "infancy",
    },
    "frequency": {
        "often": "frequent", "frequently": "frequent", "sporadic": "occasional",
    },
    "progression": {
        "getting worse": "worsening", "getting better": "improving", "unchanged": "stable",
    },
}

@dataclass
class ModifierSpan:
    slot: str
    value: str
    raw_text: str
    char_start: int
    char_end: int

# === 1. Extractor (保持不变) ===
class ModifierExtractor:
    def __init__(self):
        self.patterns = {
            "severity": re.compile(r"\b(mild|mildly|slight|moderate|moderately|severe|severely|marked)\b", re.I),
            "laterality": re.compile(r"\b(left|right|bilateral|unilateral|left-sided|right-sided|both sides)\b", re.I),
            "onset": re.compile(r"\b(infancy|childhood|adult-onset|adult|neonatal|congenital|since childhood|since infancy)\b", re.I),
            "frequency": re.compile(r"\b(recurrent|occasional|frequent|frequently|often|intermittent|sporadic)\b", re.I),
            "progression": re.compile(r"\b(progressive|worsening|getting worse|improving|getting better|stable|unchanged|deteriorating)\b", re.I),
        }

    def _canonicalize(self, slot: str, text: str) -> Optional[str]:
        t = text.lower()
        if t in VALUE_DOMAINS[slot]: return t
        if t in VALUE_SYNONYMS.get(slot, {}): return VALUE_SYNONYMS[slot][t]
        t_norm = t.replace("-", " ")
        for v in VALUE_DOMAINS[slot]:
            if v in t_norm: return v
        return None

    def extract(self, sentence: str) -> List[ModifierSpan]:
        spans = []
        for slot, pat in self.patterns.items():
            for m in pat.finditer(sentence):
                raw = m.group(0)
                canonical = self._canonicalize(slot, raw)
                if canonical:
                    spans.append(ModifierSpan(slot, canonical, raw, m.start(), m.end()))
        return spans

# === 2. Coordinator Class ===
class PostCoordinator:
    _instance = None

    def __init__(self):
        self.extractor = ModifierExtractor()
        self.pm = PromptManager.get()
        self.llm_client = UnifiedLLMClient.get()
        
        # 决策逻辑：只有当配置文件允许 LLM 且系统模式不是 "no_llm" 时，才真正启用 LLM
        # 如果系统模式是 "no_llm"，无论 config 怎么写，都强制为 False
        self.enable_llm_post_coordination = CONFIG.get("enable_llm", True) and self.llm_client.is_enabled()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = PostCoordinator()
        return cls._instance

    def _extract_json(self, text: str) -> Dict:
        """ 复用 D 阶段那个强大的提取器逻辑，这里简化写 """
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip().replace("```json", "").replace("```", "")
        
        # 尝试简单解析
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end+1])
        except: pass
        return {}

    @staticmethod
    def _ph_mid(p):
        if "char_start" in p and "char_end" in p:
            return int((p["char_start"] + p["char_end"]) / 2)
        return 0

    @staticmethod
    def _mod_mid(m):
        return int((m.char_start + m.char_end) / 2)

    def _rule_bind(self, phenotypes: List[Dict], modifiers: List[ModifierSpan]) -> Dict[int, Dict]:
        bindings = defaultdict(dict)
        mods_by_slot = defaultdict(list)
        for m in modifiers:
            mods_by_slot[m.slot].append(m)

        for i, ph in enumerate(phenotypes):
            # 获取实体在句子中的相对位置
            # 注意：这里假设 B 阶段存的是全局位置。
            # 为了简化，我们假设 phenotypes 已经被转换为了当前句子内的相对坐标
            # 或者我们简单地认为没有坐标信息时无法使用规则
            ph_mid = self._ph_mid(ph)

            for slot, mods in mods_by_slot.items():
                dist_list = []
                for m in mods:
                    d = abs(self._mod_mid(m) - ph_mid)
                    dist_list.append((d, m))
                dist_list.sort(key=lambda x: x[0])

                if not dist_list:
                    continue

                best = dist_list[0][1]
                bindings[i][slot] = best.value

        return bindings

    def annotate_sentence_batch(self, sentence: str, phenotypes: List[Dict], note_id: str) -> List[Dict]:
        """
        处理单句内的所有表型绑定
        """
        modifiers = self.extractor.extract(sentence)
        if not modifiers:
            return phenotypes, None # 无修饰词，直接返回

        # 1. 规则绑定 (Rule Binding)
        # 这里的规则绑定是必需的基线
        rule_bindings = self._rule_bind(phenotypes, modifiers)
        
        # 2. LLM 绑定 (LLM Override)
        # 只有当：有修饰词 且 (有多个表型 或 需要消歧) 时触发
        
        should_trigger_llm = self.enable_llm_post_coordination and len(phenotypes) >= 1 and len(modifiers) >= 1
        
        # 准备 LLM 输出记录（如果触发）
        llm_record = None
        
        if should_trigger_llm:
            # 准备 Prompt 数据
            ph_list = [{"index": i, "text": p["span_text"]} for i, p in enumerate(phenotypes)]
            
            mods_by_slot = defaultdict(list)
            for m in modifiers:
                if m.value not in mods_by_slot[m.slot]:
                    mods_by_slot[m.slot].append(m.value)
            
            # 生成 Template
            template_str = "{\n" + ",\n".join([f'  "{i}": {{}}' for i in range(len(ph_list))]) + "\n}"
            
            prompt = self.pm.render(
                stage='postcoordination',
                template_name='binding',
                sentence=sentence,
                phenotypes_json=json.dumps(ph_list, ensure_ascii=False),
                modifiers_json=json.dumps(mods_by_slot, ensure_ascii=False),
                template_json=template_str
            )

            try:
                # 调用统一接口
                response_text = self.llm_client.chat(prompt, max_new_tokens=CONFIG.get("max_new_tokens", 1024))
                parsed = self._extract_json(response_text)
                
                # 记录原始数据
                modifiers_serializable = [asdict(m) for m in modifiers]
                llm_record = {
                    "stage": "E",
                    "note_id": note_id,
                    "sentence": sentence,
                    "phenotypes": phenotypes, 
                    "modifiers": modifiers_serializable,
                    "raw_output": response_text,
                    "parsed_output": parsed
                }

                # 解析并覆盖规则
                for key, slotmap in parsed.items():
                    try:
                        idx = int(key)
                        if idx < 0 or idx >= len(phenotypes): continue
                        if not isinstance(slotmap, dict): continue
                        
                        for slot, val in slotmap.items():
                            if not isinstance(val, str): continue
                            v = str(val).lower()
                            if slot in VALUE_DOMAINS and v in VALUE_DOMAINS[slot]:
                                rule_bindings[idx][slot] = v
                    except: continue
                    
            except Exception as e:
                print(f"[Coord Error] {e}")

        # 3. 将结果写回 Phenotypes
        for i, ph in enumerate(phenotypes):
            if "modifiers" not in ph:
                ph["modifiers"] = {}
            # 合并绑定结果
            if i in rule_bindings:
                ph["modifiers"].update(rule_bindings[i])
            
        return phenotypes, llm_record

# === Tool Function ===

def tool_coordination(state: AgentState):
    coordinator = PostCoordinator.get()
    
    # 1. 准备句子映射 (sent_id -> text & offset)
    sentence_map = state.get("metadata", {}).get("sentence_map", [])
    sent_lookup = {s["sent_id"]: s for s in sentence_map}
    
    # 2. 按句子分组 Phenotypes
    phens_by_sent = defaultdict(list)
    for p in state['phenotypes']:
        # 仅处理 Present 的表型？
        # 如果需要，可以在这里加 if p["assertion"] == "present":
        sid = p.get("sent_id", 0)
        phens_by_sent[sid].append(p)
        
    updated_phenotypes = []
    
    # 3. 逐句处理
    for sid, p_list in phens_by_sent.items():
        sent_info = sent_lookup.get(sid)
        if not sent_info:
            # 找不到句子信息，可能只有单句
            sentence_text = state["processed_text"]
        else:
            sentence_text = sent_info["expanded"]
            
        # 调用核心逻辑
        annotated_group, llm_record = coordinator.annotate_sentence_batch(sentence_text, p_list, state["note_id"])
        updated_phenotypes.extend(annotated_group)
        
        # 记录 LLM Trace
        if llm_record:
            if 'llm_trace' not in state: state['llm_trace'] = []
            state['llm_trace'].append(llm_record)
        
    # 4. 更新状态
    # 注意：这里的 updated_phenotypes 顺序可能乱了（按句子ID排序了），如果需要保持原序，可以用 map 还原
    # 这里直接覆盖，只要后续处理不依赖顺序即可
    state['phenotypes'] = updated_phenotypes
    
    # 简单日志，不暴露过多细节
    llm_status = "LLM" if coordinator.enable_llm_post_coordination else "Rule-Only"
    state['logs'].append(f"Coordination: Modifiers bound ({llm_status}).")
    
    return state