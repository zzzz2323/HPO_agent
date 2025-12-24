import os
from typing import List, Dict, Any

def parse_tsv_data(file_path: str) -> List[Dict[str, Any]]:
    """
    鲁棒解析 GSC / GSCplus 格式
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    docs = []
    
    # 临时变量
    current_id = None
    current_text_lines = []
    gold_spans = []
    
    # 状态标志：0=等待ID, 1=读取Text, 2=读取Span
    # GSC格式有点特殊，Text和Span之间没有明确分隔符，只能靠特征判断
    # 但是 ID 行通常是很明显的（纯数字）
    
    def flush():
        nonlocal current_id, current_text_lines, gold_spans
        if current_id is not None:
            docs.append({
                "note_id": current_id,  # 统一字段名
                "text": "\n".join(current_text_lines).strip(),
                "gold_phenotypes": gold_spans  # 统一字段名
            })
        current_id = None
        current_text_lines = []
        gold_spans = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 1. 尝试识别 ID 行
        # ID 通常是纯数字，且上一行是空的（或者是文件开头）
        # 或者这一行是纯数字，且下一行看起来像是文本
        is_id_line = False
        if line.isdigit():
            # 简单启发式：如果它是纯数字，且我们还没有 Text，那它肯定是 ID
            if not current_text_lines: 
                is_id_line = True
            # 或者如果已经有了 Text 和 Spans，那这可能是新 ID
            elif gold_spans: 
                is_id_line = True
            # 如果只有 Text 没有 Span，且遇到了数字行... 这种情况最危险
            # GSC 数据集中，ID 行通常紧跟 Text。如果 Text 里有数字行，通常不会独立成行。
            # 为了安全，我们假设 GSC 格式是标准的：ID -> Text -> Spans
            
        if is_id_line:
            if current_id is not None:
                flush()
            current_id = line
            i += 1
            continue
            
        # 2. 如果不是 ID，那要么是 Text，要么是 Span
        if not line:
            i += 1
            continue
            
        parts = line.split('\t') # 尝试 Tab 分割
        if len(parts) < 3:
            parts = line.split() # 回退到空格分割
            
        # 判定是否为 Gold Span 行
        # 特征：数字 数字 文本 HP:xxxx
        is_span = False
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            # 进一步检查最后一个部分是否像 HP ID
            # 有些数据可能没有 HP ID，只有 span，所以放宽条件
            is_span = True
            
        if is_span:
            try:
                start = int(parts[0])
                end = int(parts[1])
                # 提取 mention 和 hp_id
                # 假设格式：start end mention [HP:xxx]
                # mention 可能是中间的所有词
                
                # 寻找 HP ID
                hp_id = None
                mention_parts = []
                
                for p in parts[2:]:
                    if p.startswith("HP:") and len(p) >= 7:
                        hp_id = p
                    else:
                        mention_parts.append(p)
                
                mention = " ".join(mention_parts)
                
                gold_spans.append({
                    "start": start,
                    "end": end,
                    "mention": mention,
                    "hp_id": hp_id
                })
            except:
                # 解析失败，只好当做文本处理（虽然极少见）
                current_text_lines.append(line)
        else:
            # 既然不是 ID 也不是 Span，那就是 Text
            current_text_lines.append(line)
            
        i += 1

    # 循环结束，Flush 最后一个
    flush()
    
    print(f"[Loader] Loaded {len(docs)} docs from {file_path}")
    return docs