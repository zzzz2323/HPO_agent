import os
import re
import spacy
import numpy as np
import yaml
from typing import List, Dict, Any, Set
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from core.state import AgentState

# === 1. 加载配置 ===
def load_config():
    # 假设 config 路径正确
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)["recall"]

CONFIG = load_config()

# === 2. 辅助常数 (完全保持不变) ===
STOP_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "and", 
    "or", "but", "no", "not", "is", "are", "was", "were", "be", "has", "have", 
    "had", "can", "will", "may", "should", "would", "either", "who", "which",
    "patients", "patient", "present", "presents", "additional", "type", "series",
    "identified", "cases", "available", "suggest", "analysis", "strategy", "fulfil",
    "criter", "instance", "instances",
    "history", "report", "review", "findings", "physical", "examination", 
    "notes", "record", "documented", "child", "infant", "adult", "age", "year", 
    "old", "male", "female", "mother", "father", "family", "member", "onset", 
    "time", "mode", "inheritance", "syndrome", "disease", "disorder", "condition", 
    "sign", "symptom", "feature", "abnormality", "finding", "process", 
    "measurement", "clinical", "tumours", "tumour", "material"
}
ILLEGAL_POS_START = {"VERB", "AUX", "ADP", "PRON", "CCONJ", "SCONJ", "DET", "ADV"}
INVALID_CHARS = set("()[]{}<>,:;?./\\|!@#$%^&*")

# 加载 Spacy (带容错)
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
except:
    print("[Warning] Spacy load failed in recall. using minimal fallback.")
    nlp = spacy.blank("en")

# === 3. 辅助函数 (从 B_candidates.py 移植) ===
def filter_nested_spans(spans: List[str]) -> List[str]:
    """
    过滤掉长度小于3个词的、且被其他更长 Span 包含的冗余子 Span。
    """
    # 按照长度从长到短排序
    spans.sort(key=len, reverse=True)
    filtered_spans = []
    
    for current_span in spans:
        current_span_words = len(current_span.split())
        is_nested_and_short = False
        
        # 仅检查长度小于3个词的 Span
        if current_span_words < 3:
            for kept_span in filtered_spans:
                # 检查是否是已保留 Span 的子集
                if current_span in kept_span:
                    is_nested_and_short = True
                    break
        
        if not is_nested_and_short:
            filtered_spans.append(current_span)
            
    return filtered_spans

class CandidateExtractor:
    @staticmethod
    def get_spans(text: str) -> List[str]:
        spans = []
        doc = nlp(text)
        tokens = doc
        
        # --- A. 优先级最高：Spacy Noun Chunks (句法结构) ---
        if doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                
                # 过滤掉包含非法标点或纯数字/纯停用词的 Chunk
                if any(char in INVALID_CHARS for char in chunk_text):
                    continue
                
                # 确保 Span 不以强停用词开头
                if chunk[0].lower_ in STOP_WORDS: 
                    continue

                spans.append(chunk_text)

        # --- B. 优先级次之：N-gram 滑动窗口（补漏）---
        max_len = 6
        for i in range(len(tokens)):
            start_token = tokens[i]
            
            # 1. 严格过滤 Span 起点
            if start_token.pos_ in ILLEGAL_POS_START:
                continue
            if start_token.lower_ in STOP_WORDS:
                continue
            if any(char in INVALID_CHARS for char in start_token.text):
                continue

            # 循环遍历 Span 的结束位置 j
            for j in range(i, min(i + max_len, len(tokens))):
                end_token = tokens[j]
                
                # 遇到标点符号直接打断整个内循环，不再延伸
                if any(char in INVALID_CHARS for char in end_token.text):
                    break
                
                current_span_tokens = tokens[i:j+1]
                span_text = current_span_tokens.text.strip()
                
                # --- 过滤检查 ---
                
                # 2. 规则：Span 必须是名词或形容词
                if len(current_span_tokens) == 1 and current_span_tokens[0].pos_ not in ['NOUN', 'ADJ']:
                    continue

                # 3. 规则：Span 不能以强停用词或非法词性结尾
                if end_token.lower_ in STOP_WORDS or end_token.pos_ in ILLEGAL_POS_START:
                    continue 

                # 4. 规则：最终 Span 长度检查
                if len(span_text) < 2:
                    continue
                
                if span_text:
                    spans.append(span_text)

        # --- C. 去重与清洗 ---
        unique_spans = []
        seen = set()
        for s in spans:
            clean = s.strip()
            key = clean.lower()
            
            # 最终过滤
            if len(key) < 3: continue
            if key in STOP_WORDS: continue
            if any(char in INVALID_CHARS for char in key): continue

            if key not in seen:
                seen.add(key)
                unique_spans.append(clean)
        
        return unique_spans

class HPORetriever:
    _instance = None

    def __init__(self):
        self.surface_forms = []
        self.hp_ids = []
        self.hp_labels = []
        
        # 黑名单与忽略ID
        self.blacklist = {
            "all", "other", "none", "finding", "abnormality", "disease", 
            "disorder", "syndrome", "phenotype", "mode of inheritance", 
            "clinical course", "past medical history"
        }
        self.ignore_ids = {"HP:0000001", "HP:0000118", "HP:0000005"}

        # === 手动加载 OBO ===
        print(f"[Recall] Loading HPO Ontology (Manual Parse): {CONFIG['obo_path']} ...")
        self._load_obo_manual(CONFIG["obo_path"])
        print(f"HPO Lexicon ready: {len(self.surface_forms)} terms loaded.")

        # BM25
        print("Building BM25 index...")
        self.tokenized_corpus = [sf.split() for sf in self.surface_forms]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Embedding
        print(f"Loading BGE embedder from {CONFIG['model_path']}...")
        self.embedder = SentenceTransformer(CONFIG["model_path"])
        self.vecs = self.embedder.encode(
            self.surface_forms,
            batch_size=512,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = HPORetriever()
        return cls._instance

    def _load_obo_manual(self, path):
        """
        手动解析 OBO 文件，跳过 pronto 库的 strict 检查
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        current_id = None
        current_name = None
        current_synonyms = []
        is_obsolete = False
        
        # 正则匹配 synonym 行
        syn_pattern = re.compile(r'synonym: "(.*?)"')

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line == '[Term]':
                    self._add_entry(current_id, current_name, current_synonyms, is_obsolete)
                    current_id = None
                    current_name = None
                    current_synonyms = []
                    is_obsolete = False
                    continue
                
                if line.startswith('id: HP:'):
                    current_id = line.split('id: ')[1].split()[0]
                elif line.startswith('name:'):
                    current_name = line.split('name: ')[1]
                elif line.startswith('synonym:'):
                    m = syn_pattern.search(line)
                    if m:
                        current_synonyms.append(m.group(1))
                elif line.startswith('is_obsolete: true'):
                    is_obsolete = True

            # 循环结束后保存最后一个
            self._add_entry(current_id, current_name, current_synonyms, is_obsolete)

    def _add_entry(self, hp_id, name, synonyms, is_obsolete):
        if not hp_id or not name: return
        if is_obsolete: return
        if hp_id in self.ignore_ids: return

        forms = set()
        forms.add(name)
        forms.update(synonyms)

        for text in forms:
            text_lower = text.lower().strip()
            
            # 过滤逻辑
            if len(text_lower) < 3: continue
            if text_lower in self.blacklist: continue
            
            self.surface_forms.append(text_lower)
            self.hp_ids.append(hp_id)
            self.hp_labels.append(name) # 统一存标准名

    def hybrid_recall(self, mention: str):
        # 使用配置中的 topk
        bm25_topk = CONFIG.get("bm25_topk", 100)
        final_topk = CONFIG.get("final_topk", 20)
        
        mention_lower = mention.lower().strip()

        # 1. BM25 初始召回
        scores = self.bm25.get_scores(mention_lower.split())
        bm25_idx = np.argsort(scores)[::-1][:bm25_topk]

        if len(bm25_idx) == 0:
            return [], []

        # 2. Embedding 相似度计算
        query_vec = self.embedder.encode(mention_lower, normalize_embeddings=True, convert_to_numpy=True)
        cand_vecs = self.vecs[bm25_idx]
        embed_scores = (cand_vecs @ query_vec.T).flatten()

        # 3. 排序
        ranked_local_idx = np.argsort(embed_scores)[::-1]
        ranked_global_idx = bm25_idx[ranked_local_idx]
        ranked_scores = embed_scores[ranked_local_idx].copy()

        # ====== 4. 关键修复：词汇完美匹配优先级覆盖 ======
        perfect_match_found = False
        
        for i, idx in enumerate(ranked_global_idx):
            if self.surface_forms[idx] == mention_lower:
                # 强制将完美匹配的项分数提升到 1.001
                ranked_scores[i] = 1.001 
                perfect_match_found = True
        
        if perfect_match_found:
            new_order = np.argsort(ranked_scores)[::-1]
            ranked_global_idx = ranked_global_idx[new_order]
            ranked_scores = ranked_scores[new_order]

        # 5. 动态截断策略
        valid_global_idx = []
        valid_scores = []
        
        # 从配置读取阈值，如果配置没给就用默认值保持逻辑一致
        min_score = CONFIG.get("min_candidate_score", 0.80)
        
        for idx, score in zip(ranked_global_idx, ranked_scores):
            # 截断条件：如果分数低于阈值 且 不是完美匹配 (1.001)，则停止
            if score < min_score and score < 1.001: 
                break 
            
            valid_global_idx.append(int(idx))
            valid_scores.append(float(score))
            
            if len(valid_global_idx) >= final_topk:
                break

        return valid_global_idx, valid_scores

    def get_candidate_details(self, idxs, scores):
        cands = []
        for i, idx in enumerate(idxs):
            cands.append({
                "id": self.hp_ids[idx],
                "label": self.hp_labels[idx],
                "surface": self.surface_forms[idx],
                "score": scores[i],
                "match_type": "hybrid"
            })
        return cands

# === Tool Function ===

def tool_recall(state: AgentState):
    """
    Agent 节点函数：实体召回
    """
    retriever = HPORetriever.get()
    
    # 假设 A 阶段产生的分句信息存在 state['metadata']['sentence_map']
    sentence_map = state.get('metadata', {}).get('sentence_map', [])
    if not sentence_map:
        sentence_map = [{"expanded": state['processed_text'], "sent_id": 0, "note_id": state['note_id']}]

    extracted_phenotypes = []
    
    # 调试统计
    stats_span_kept = 0
    stats_cands_total = 0
    
    # 从配置读取 Span 级阈值
    min_span_score = CONFIG.get("min_span_score", 0.90)
    
    for s in sentence_map:
        text = s["expanded"]
        spans = CandidateExtractor.get_spans(text)
        
        # 如果需要启用 filter_nested_spans，可以在这里解开注释
        # spans = filter_nested_spans(spans)
        
        for sp in spans:
            idxs, scores = retriever.hybrid_recall(sp)
            
            if not idxs: continue
            
            # ====== 门槛 1: Span 准入过滤 ======
            # 只有最高分非常高，才认为这个 Span 是有效的表型
            if scores[0] < min_span_score:
                continue
                
            stats_span_kept += 1
            
            candidates = retriever.get_candidate_details(idxs, scores)
            stats_cands_total += len(candidates)
            
            # 构建 Phenotype 对象
            phenotype = {
                "span_text": sp,
                "sent_id": s["sent_id"],
                "note_id": state["note_id"],
                "candidates": candidates,
                # 初始化后续阶段需要的字段
                "final_id": None,
                "assertion": "present",
                "modifiers": {}
            }
            extracted_phenotypes.append(phenotype)

    # 更新 State
    state['phenotypes'] = extracted_phenotypes
    state['logs'].append(f"Recall: Processed {len(sentence_map)} sentences. Found {len(extracted_phenotypes)} valid spans.")
    
    return state