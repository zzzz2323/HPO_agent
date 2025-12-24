from typing import Dict, Any
from langgraph.graph import StateGraph, END

# 导入共享状态定义
from core.state import AgentState

# 导入各个阶段的工具函数
# 确保您的 tools 文件夹下有 __init__.py，或者路径正确
from tools.text_processing import tool_preprocess
from tools.recall import tool_recall
from tools.normalization import tool_normalize
from tools.assertion import tool_assertion
from tools.coordination import tool_coordination

class HPOAgentGraph:
    def __init__(self):
        """
        初始化 HPO 提取流水线的图结构
        """
        # 1. 创建状态图，指定状态模式为 AgentState
        self.workflow = StateGraph(AgentState)
        
        # ============================================================
        # 2. 添加节点 (Nodes)
        # 节点名称可以自定义，这里与阶段名称保持一致
        # ============================================================
        
        # A 阶段: 预处理 (分句、缩写展开)
        self.workflow.add_node("preprocess", tool_preprocess)
        
        # B 阶段: 实体召回 (Spacy Noun Chunk + BM25/Embedding)
        self.workflow.add_node("recall", tool_recall)
        
        # C 阶段: 归一化 (BERT Rerank + LLM Disambiguation)
        self.workflow.add_node("normalize", tool_normalize)
        
        # D 阶段: 断言分类 (Rule + LLM Check)
        self.workflow.add_node("assertion", tool_assertion)
        
        # E 阶段: 修饰词绑定 (Syntax Analysis + LLM Binding)
        self.workflow.add_node("coordination", tool_coordination)
        
        # ============================================================
        # 3. 定义边 (Edges) - 构建流水线逻辑
        # ============================================================
        
        # 设置入口点：从预处理开始
        self.workflow.set_entry_point("preprocess")
        
        # 线性连接各个节点
        self.workflow.add_edge("preprocess", "recall")
        self.workflow.add_edge("recall", "normalize")
        self.workflow.add_edge("normalize", "assertion")
        self.workflow.add_edge("assertion", "coordination")
        
        # E 阶段结束后，流程结束
        self.workflow.add_edge("coordination", END)
        
        # ============================================================
        # 4. 编译图
        # ============================================================
        self.app = self.workflow.compile()

    def run(self, note_text: str, note_id: str = "unknown") -> AgentState:
        """
        执行流水线的对外接口
        
        Args:
            note_text: 原始病历文本
            note_id: 病历 ID (用于追踪)
            
        Returns:
            AgentState: 包含最终结果的完整状态对象
        """
        # 1. 构造初始状态
        initial_state: AgentState = {
            "note_id": note_id,
            "raw_text": note_text,
            # 以下字段由各阶段填充，初始化为空
            "processed_text": "",
            "metadata": {},
            "phenotypes": [],
            "logs": [],
            "llm_trace": []
        }
        
        # 2. 调用图执行 (Invoke)
        # LangGraph 会自动管理状态在节点间的传递
        try:
            final_state = self.app.invoke(initial_state)
            return final_state
        except Exception as e:
            # 简单的错误捕获，防止单条数据崩溃导致整个批处理停止
            print(f"Error processing note {note_id}: {str(e)}")
            # 返回包含错误日志的状态
            initial_state["logs"].append(f"CRITICAL ERROR: {str(e)}")
            return initial_state

# ============================================================
# 简单的测试代码 (当直接运行此文件时)
# ============================================================
if __name__ == "__main__":
    # 模拟一段文本
    test_text = "The patient presents with severe hp. and mild fever."
    
    print("Initializing Agent Graph...")
    agent = HPOAgentGraph()
    
    print(f"Running test on: '{test_text}'")
    result = agent.run(test_text, note_id="test_001")
    
    print("\n=== Execution Logs ===")
    for log in result["logs"]:
        print(f"- {log}")
        
    print("\n=== Extracted Phenotypes ===")
    import json
    print(json.dumps(result["phenotypes"], indent=2, ensure_ascii=False))