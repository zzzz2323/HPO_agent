import os
import json
import logging

# ==========================================
# 1. 新增：容错重试的辅助函数 (对应Markdown需求)
# ==========================================

def trim_modifiers_for_retry(hpo_data):
    """
    策略1：修剪修饰符 (用于Phenopacket导出失败时)
    逻辑：去除部分非必须的修饰符（如severity, onset），只保留核心ID。
    """
    logging.info("-> [Retry Strategy] Trimming modifiers...")
    # 这里的逻辑是创建一个副本，防止修改原始数据
    new_data = []
    for item in hpo_data:
        new_item = item.copy()
        # 移除可能导致校验失败的额外字段，只保留最稳的字段
        if 'modifiers' in new_item:
            # 简单粗暴：直接移除修饰符，确保能导出
            new_item.pop('modifiers', None) 
        if 'severity' in new_item:
            new_item.pop('severity', None)
        new_data.append(new_item)
    return new_data

def drop_all_modifiers_for_retry(hpo_data):
    """
    策略2：丢弃所有修饰符 (用于OWL导出失败时)
    逻辑：彻底为了合法性牺牲细节，只保留最核心的HPO ID和Label。
    """
    logging.info("-> [Retry Strategy] Dropping all modifiers (Keep ID only)...")
    new_data = []
    for item in hpo_data:
        # 只保留 id 和 label，其他全部丢弃
        clean_item = {
            "id": item.get("id"),
            "label": item.get("label")
        }
        new_data.append(clean_item)
    return new_data

# ==========================================
# 2. 导出函数 (如果你还没有定义导出逻辑，用这两个)
# ==========================================

def export_phenopacket(hpo_data, note_id, output_path):
    """生成简单的 Phenopacket 格式 JSON"""
    # 确保目录存在
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{note_id}.json")
    
    # 模拟 Phenopacket 结构
    packet = {
        "id": note_id,
        "phenotypicFeatures": []
    }
    
    for item in hpo_data:
        feature = {
            "type": {
                "id": item.get("id"),
                "label": item.get("label")
            }
        }
        # 如果有修饰符，加进去
        if "modifiers" in item and item["modifiers"]:
             feature["modifiers"] = item["modifiers"]
        
        packet["phenotypicFeatures"].append(feature)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(packet, f, indent=2)
    logging.info(f"Exported Phenopacket to {file_path}")

def export_to_owl(hpo_data, note_id, output_path):
    """生成简单的 Manchester Syntax OWL 格式"""
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{note_id}.owl")
    
    lines = []
    lines.append(f"Ontology: <http://example.org/{note_id}>")
    lines.append(f"Import: <http://purl.obolibrary.org/obo/hp.owl>")
    lines.append("")
    lines.append(f"Individual: patient_{note_id}")
    lines.append("    Types:")
    
    for item in hpo_data:
        # 简单的 OWL 转换
        lines.append(f"        {item.get('id')} ! {item.get('label')},")
    
    # 去掉最后一个逗号，改为分号或直接结束
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logging.info(f"Exported OWL to {file_path}")