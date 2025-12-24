import re
import yaml
import spacy
from typing import List, Dict, Any, Set
from core.state import AgentState
from core.llm_interface import UnifiedLLMClient
from utils.prompt_manager import PromptManager

# === 加载配置 ===
def load_config():
    # 假设配置文件路径在 config/settings.yaml
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)["preprocess"]

CONFIG = load_config()

# === 常量定义 (保持不变) ===
COMMON_WORDS = {
    "A", "I", "AM", "PM", "NO", "YES", "OF", "IN", "ON", "AT", "TO", "FOR", "BY", "AND", "OR",
    "THE", "IS", "WAS", "ARE", "WERE", "BE", "HAS", "HAVE", "HAD", "DO", "DID",
    "HE", "SHE", "IT", "HIS", "HER", "MY", "WE", "YOU",
    "PATIENT", "HISTORY", "PRESENT", "ILLNESS", "PHYSICAL", "EXAMINATION",
    "NOTE", "DATE", "TIME", "SIGNED", "DR", "MD", "RN", "LEFT", "RIGHT",
    "MALE", "FEMALE", "AGE", "DOB", "SEX", "SERVICE", "ADMISSION", "DISCHARGE",
    "PLAN", "ASSESSMENT", "DIAGNOSIS", "ALLERGIES", "MEDICATIONS",
    "HOSPITAL", "CLINIC", "DEPARTMENT", "STATUS", "REVIEW", "SYSTEMS",
    "MILD", "MODERATE", "SEVERE", "ACUTE", "CHRONIC", "NORMAL", "STABLE",
    "MG", "KG", "ML", "L", "CM", "MM"
}

# 全局加载 Spacy，避免每次调用都加载
try:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
except Exception:
    print("[Warning] Spacy load failed in text_processing. using minimal fallback.")
    nlp = None

class TextProcessor:
    _instance = None

    def __init__(self, abbrev_path: str = "data/abbrev.tsv"):
        self.abbr_dict = self._load_abbrev(abbrev_path)
        self.pm = PromptManager.get()
        self.llm_client = UnifiedLLMClient.get()
        
        # 决策逻辑：只有当配置文件允许 且 客户端不是 no_llm 模式时才启用
        self.use_llm = CONFIG.get("use_llm_expansion", True) and self.llm_client.is_enabled()
        
        # 缓存可以跨请求保留，提高效率
        self.cache: Dict[str, str] = {}

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = TextProcessor()
        return cls._instance

    def _load_abbrev(self, path: str) -> Dict[str, str]:
        # ... (保持原逻辑) ...
        abbr = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        abbr[parts[0].lower()] = parts[1]
        except Exception as e:
            print(f"Warning: Abbreviation file error: {e}")
        return abbr

    def _clean(self, text: str) -> str:
        text = text.replace("\u200b", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _is_abbrev_candidate(self, token: str) -> bool:
        # ... (保持原逻辑) ...
        clean_token = re.sub(r"[^\w]", "", token)
        if not clean_token or clean_token.isdigit(): return False
        if clean_token.upper() in COMMON_WORDS: return False
        if clean_token.lower() in self.abbr_dict: return False
        if len(clean_token) > 6: return False
        if clean_token.islower(): return False
        if len(clean_token) < 2: return False
        return True

    def _expand_with_llm(self, word: str, sent_text: str) -> str:
        if not self.use_llm:
            return word

        token_clean = re.sub(r"[^\w]", "", word)
        
        # 1. 查缓存
        if token_clean in self.cache:
            return self.cache[token_clean]
        
        # 2. 规则过滤
        if not self._is_abbrev_candidate(token_clean):
            return word

        # 3. LLM 调用
        # 使用 PromptManager 渲染
        prompt = self.pm.render(
            stage='preprocess',
            template_name='abbreviation_expansion',
            sentence=sent_text,
            word=word
        )

        try:
            # 调用统一接口
            response_text = self.llm_client.chat(prompt, max_new_tokens=20)
            
            # 清理结果
            text = response_text.strip().strip('"').strip("'")
            
            # 简单的幻觉校验
            if len(text.split()) > 8 or not text:
                result = word
            else:
                result = text
            
            self.cache[token_clean] = result
            return result
            
        except Exception:
            return word

    def process_note(self, note_text: str, note_id: str) -> dict:
        """
        处理单篇笔记，返回结构化结果
        """
        text = self._clean(note_text)
        
        # 容错处理 Spacy
        if nlp:
            doc = nlp(text)
            sentences = doc.sents
        else:
            # 如果 spacy 挂了，简单的正则分句回退
            sentences = [type('obj', (object,), {'text': s, 'start_char': 0, 'end_char': len(s)}) 
                         for s in re.split(r'(?<=[.!?])\s+', text)]

        llm_trigger_count = 0
        sentence_map = []
        expanded_full_text_parts = []
        
        for i, sent in enumerate(sentences):
            raw = sent.text
            
            # 句子级别的展开逻辑
            # 注意：原代码依赖 spacy tokenization，这里我们简单复刻
            # 为了不破坏原有逻辑，还是假设 spacy 正常工作
            if nlp:
                out_tokens = []
                for token in sent:
                    raw_token = token.text
                    core = re.sub(r"[^\w]", "", raw_token).lower()
                    
                    if core in self.abbr_dict:
                        out_tokens.append(self.abbr_dict[core])
                    elif re.sub(r"[^\w]", "", raw_token) in self.cache:
                        out_tokens.append(self.cache[re.sub(r"[^\w]", "", raw_token)])
                    else:
                        # 记录 LLM 调用前的状态，用于统计
                        before_cache_size = len(self.cache)
                        expanded = self._expand_with_llm(raw_token, raw)
                        if len(self.cache) > before_cache_size:
                            llm_trigger_count += 1
                        out_tokens.append(expanded)
                expanded_sent = " ".join(out_tokens)
            else:
                # Fallback: 不做展开
                expanded_sent = raw

            expanded_full_text_parts.append(expanded_sent)

            # 获取字符位置
            start_char = getattr(sent, 'start_char', 0)
            end_char = getattr(sent, 'end_char', len(raw))

            sentence_map.append({
                "sent_id": i,
                "raw": raw,
                "expanded": expanded_sent,
                "char_start": start_char,
                "char_end": end_char,
            })
            
        return {
            "processed_text": " ".join(expanded_full_text_parts),
            "sentence_map": sentence_map,
            "llm_calls": llm_trigger_count
        }

# === 暴露给 Graph 的 Tool 函数 ===

def tool_preprocess(state: AgentState):
    """
    Agent 节点函数：预处理
    """
    processor = TextProcessor.get() # 获取单例
    
    # 执行核心逻辑
    result = processor.process_note(state['raw_text'], state['note_id'])
    
    # 更新状态
    state['processed_text'] = result['processed_text']
    
    # 我们可以把 sentence_map 存入 metadata 或者单独的字段，方便后续步骤查阅
    # 这里假设 State 中有一个 metadata 字典
    if 'metadata' not in state:
        state['metadata'] = {}
    state['metadata']['sentence_map'] = result['sentence_map']
    
    # 记录日志
    mode_str = "LLM" if processor.use_llm else "Rule-Only"
    state['logs'].append(f"Preprocess: Expanded text ({mode_str}). LLM triggered {result['llm_calls']} times.")
    
    return state