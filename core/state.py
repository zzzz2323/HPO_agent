from typing import TypedDict, List, Dict, Any, Optional

# ============================================================
# 1. 单个表型实体的状态定义
# ============================================================
class Phenotype(TypedDict):
    """
    代表一个被提取出的临床表型实体。
    这个结构是动态的：B 阶段创建它，C/D/E 阶段不断填充它。
    """
    
    # --- 基础信息 (B阶段: Recall 产生) ---
    span_text: str          # 原文片段，例如 "high fever"
    char_start: int         # 在句子或全文中的起始字符位置
    char_end: int           # 结束字符位置
    sent_id: int            # 所在句子的索引 (对应 metadata['sentence_map'])
    note_id: str            # 所属病历 ID
    
    # --- 候选集 (B阶段: Recall 产生) ---
    # 结构: [{"id": "HP:0000001", "label": "...", "score": 0.95}, ...]
    candidates: List[Dict[str, Any]] 
    
    # --- 归一化结果 (C阶段: Normalization 产生) ---
    final_id: Optional[str] # 最终确定的 HPO ID (如 "HP:0000001")
    confidence: float       # 归一化置信度 (0.0 - 1.0)
    norm_method: str        # 方法来源: "baseline_confident", "llm", "no_candidate" 等
    
    # --- 断言状态 (D阶段: Assertion 产生) ---
    # 取值: "present", "absent", "uncertain", "family", "historical"
    assertion: str          
    assertion_conf: float   # 断言置信度
    assertion_method: str   # 方法来源: "rule", "llm"
    
    # --- 修饰词 (E阶段: Coordination 产生) ---
    # 结构: {"severity": "severe", "onset": "childhood", "frequency": "frequent"}
    modifiers: Dict[str, str]


# ============================================================
# 2. Agent 全局状态定义
# ============================================================
class AgentState(TypedDict):
    """
    Agent 的全局“记忆”。
    LangGraph 会在节点之间传递这个对象。
    """
    
    # --- 输入数据 ---
    note_id: str            # 当前处理的病历 ID
    raw_text: str           # 原始输入的病历文本
    
    # --- 中间过程数据 (A 阶段产出) ---
    processed_text: str     # 预处理/缩写展开后的完整文本
    
    # 元数据字典，用于存放辅助信息
    # 例如: {"sentence_map": [{"sent_id": 0, "text": "...", "start": 0}, ...]}
    metadata: Dict[str, Any] 
    
    # --- 核心结果列表 (B-E 阶段产出) ---
    # 随着流水线推进，这个列表里的 Phenotype 对象会被不断完善
    phenotypes: List[Phenotype]
    
    # --- 系统日志与追踪 ---
    # 简要日志：用于在控制台打印进度，例如 ["Step A done", "Step C: LLM triggered for 'pain'"]
    logs: List[str]         
    
    # 详细追踪：专门用于记录 LLM 的完整输入输出，用于后续分析和 debug
    # 结构: [{"stage": "C", "input": "...", "output": "..."}, ...]
    llm_trace: List[Dict[str, Any]]