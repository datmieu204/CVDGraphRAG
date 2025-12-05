"""
UMLS CSV 数据导入模块
专门处理 UMLS 格式的结构化数据（CSV文件）
不需要调用 LLM，直接从 CSV 提取实体和关系
"""

import os
import csv
from typing import Dict, List, Tuple
from utils import get_embedding, str_uuid

from logger_ import get_logger

logger = get_logger("import_umls_csv", log_file="logs/import_umls_csv.log")


def parse_umls_csv(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    解析 UMLS CSV 文件
    
    Args:
        file_path: CSV 文件路径
    
    Returns:
        (entities, relationships)
        entities: [{'id': CUI, 'name': name, 'type': type, 'description': def}, ...]
        relationships: [{'src': CUI1, 'tgt': CUI2, 'type': RELA, 'description': desc}, ...]
    """
    logger.info(f"  [解析 UMLS CSV] {os.path.basename(file_path)}")
    
    entities_dict = {}  # 使用字典避免重复
    relationships = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过表头或自动检测
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader, 1):
                try:
                    # 提取实体 1
                    cui1 = row.get('CUI1', '').strip()
                    name1 = row.get('name_1', '').strip()
                    def1 = row.get('def_1', '').strip()
                    
                    # 提取实体 2
                    cui2 = row.get('CUI2', '').strip()
                    name2 = row.get('name_2', '').strip()
                    def2 = row.get('def_2', '').strip()
                    
                    # 提取关系
                    rel = row.get('REL', '').strip()
                    rela = row.get('RELA', '').strip()
                    sab = row.get('SAB', '').strip()
                    
                    if not cui1 or not cui2:
                        continue
                    
                    # 添加实体 1
                    if cui1 not in entities_dict:
                        # 推断实体类型（基于名称的简单规则）
                        entity_type = _infer_entity_type(name1, def1)
                        entities_dict[cui1] = {
                            'id': cui1,
                            'name': name1.upper(),
                            'type': entity_type,
                            'description': def1
                        }
                    
                    # 添加实体 2
                    if cui2 not in entities_dict:
                        entity_type = _infer_entity_type(name2, def2)
                        entities_dict[cui2] = {
                            'id': cui2,
                            'name': name2.upper(),
                            'type': entity_type,
                            'description': def2
                        }
                    
                    # 添加关系
                    rel_type = _map_relation_type(rela, rel)
                    rel_desc = f"{rela} ({rel})" if rela else rel
                    
                    relationships.append({
                        'src': cui1,
                        'tgt': cui2,
                        'type': rel_type,
                        'description': rel_desc,
                        'source': sab
                    })
                    
                except Exception as e:
                    logger.warning(f"  ⚠️  跳过行 {idx}: {e}")
                    continue
        
        entities = list(entities_dict.values())
        
        logger.info(f"  ✅ 解析完成:")
        logger.info(f"     - 实体: {len(entities)}")
        logger.info(f"     - 关系: {len(relationships)}")
        
        return entities, relationships
        
    except Exception as e:
        logger.error(f"  ❌ 解析失败: {e}")
        return [], []


def _infer_entity_type(name: str, definition: str) -> str:
    """
    基于名称和定义推断实体类型
    """
    name_lower = name.lower()
    def_lower = definition.lower() if definition else ""
    
    # 疾病相关
    if any(keyword in name_lower for keyword in ['disease', 'syndrome', 'disorder', 'failure', 'infection']):
        return "DISEASE"
    if any(keyword in def_lower for keyword in ['disease', 'pathologic', 'disorder']):
        return "DISEASE"
    
    # 药物相关
    if any(keyword in name_lower for keyword in ['drug', 'medication', 'therapy']):
        return "MEDICATION"
    if any(keyword in def_lower for keyword in ['drug', 'therapeutic', 'inhibitor', 'antagonist']):
        return "MEDICATION"
    
    # 症状相关
    if any(keyword in name_lower for keyword in ['pain', 'symptom', 'ache', 'fever']):
        return "SYMPTOM"
    
    # 检查/程序
    if any(keyword in name_lower for keyword in ['test', 'procedure', 'examination', 'surgery']):
        return "PROCEDURE"
    if any(keyword in def_lower for keyword in ['procedure', 'surgical', 'examination']):
        return "PROCEDURE"
    
    # 解剖结构
    if any(keyword in name_lower for keyword in ['heart', 'lung', 'liver', 'kidney', 'organ']):
        return "ANATOMY"
    if any(keyword in def_lower for keyword in ['anatomy', 'anatomical', 'organ']):
        return "ANATOMY"
    
    # 默认为概念
    return "CONCEPT"


def _map_relation_type(rela: str, rel: str) -> str:
    """
    将 UMLS 关系映射到更友好的关系类型
    """
    rela_lower = rela.lower() if rela else ""
    
    # 治疗相关
    if 'treat' in rela_lower or 'therapy' in rela_lower:
        return "TREATS"
    if 'may_be_treated_by' in rela_lower:
        return "TREATED_BY"
    
    # 禁忌
    if 'contraindicated' in rela_lower:
        return "CONTRAINDICATED"
    
    # 因果关系
    if 'cause' in rela_lower:
        return "CAUSES"
    
    # 临床关联
    if 'clinically_associated' in rela_lower or 'associated' in rela_lower:
        return "ASSOCIATED_WITH"
    
    # 共现
    if 'co-occurs' in rela_lower or 'co_occurs' in rela_lower:
        return "CO_OCCURS_WITH"
    
    # 表现
    if 'manifestation' in rela_lower:
        return "MANIFESTATION_OF"
    
    # 诊断
    if 'diagnos' in rela_lower or 'indicate' in rela_lower:
        return "INDICATES"
    
    # ISA 关系
    if rel == 'CHD' or 'isa' in rela_lower:
        return "IS_A"
    
    # 部分关系
    if rel == 'PAR' or 'part' in rela_lower:
        return "PART_OF"
    
    # 默认为通用关系
    if rela:
        return rela.upper().replace(' ', '_')
    
    return "RELATED_TO"


def create_umls_nodes_and_relationships(n4j, entities: List[Dict], relationships: List[Dict], gid: str):
    """
    将 UMLS 实体和关系写入 Neo4j
    使用批处理避免内存溢出
    
    Args:
        n4j: Neo4j 连接
        entities: 实体列表
        relationships: 关系列表
        gid: 图ID
    """
    import gc
    
    BATCH_SIZE = 32  # 每批处理50个实体
    
    logger.info(f"  [Neo4j] 开始写入 {len(entities)} 个 UMLS 实体...")
    logger.info(f"  [策略] 使用批处理，每批 {BATCH_SIZE} 个实体")
    
    created_nodes = 0
    
    # 1. 分批创建实体节点
    total_batches = (len(entities) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(0, len(entities), BATCH_SIZE):
        batch_entities = entities[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        logger.info(f"  [批次 {batch_num}/{total_batches}] 处理 {len(batch_entities)} 个实体...")
        
        for entity in batch_entities:
            entity_id = entity['id']  # CUI
            entity_name = entity['name']
            entity_type = entity['type']
            description = entity['description']
            
            # 生成 embedding
            embedding_text = f"{entity_name}: {description}" if description else entity_name
            embedding = get_embedding(embedding_text)
            
            # 创建节点的 Cypher 查询
            create_node_query = """
            MERGE (n:`%s` {id: $id, gid: $gid})
            ON CREATE SET 
                n.cui = $cui,
                n.name = $name,
                n.description = $description,
                n.embedding = $embedding,
                n.source = 'UMLS',
                n.data_type = 'structured'
            ON MATCH SET
                n.description = CASE WHEN n.description IS NULL OR n.description = '' 
                                     THEN $description 
                                     ELSE n.description END,
                n.embedding = CASE WHEN n.embedding IS NULL 
                                   THEN $embedding 
                                   ELSE n.embedding END
            RETURN n
            """ % entity_type
            
            try:
                result = n4j.query(create_node_query, {
                    'id': entity_name,  # 使用 name 作为主 ID（去重）
                    'cui': entity_id,    # 保留 CUI 作为额外属性
                    'name': entity_name,
                    'gid': gid,
                    'description': description,
                    'embedding': embedding
                })
                if result:
                    created_nodes += 1
            except Exception as e:
                logger.warning(f"    ⚠️  创建节点失败: {entity_name} ({entity_id}) - {e}")
        
        # 每批次后强制垃圾回收
        gc.collect()
        logger.info(f"    ✅ 批次完成: {created_nodes}/{len(entities[:batch_idx + BATCH_SIZE])}")
    
    logger.info(f"  ✅ 实体节点创建完成: {created_nodes}/{len(entities)}")
    
    # 2. 分批创建关系
    logger.info(f"  [Neo4j] 开始创建 {len(relationships)} 个关系...")
    
    RELATION_BATCH_SIZE = 100  # 关系批次可以更大
    created_rels = 0
    total_rel_batches = (len(relationships) + RELATION_BATCH_SIZE - 1) // RELATION_BATCH_SIZE
    
    for batch_idx in range(0, len(relationships), RELATION_BATCH_SIZE):
        batch_rels = relationships[batch_idx:batch_idx + RELATION_BATCH_SIZE]
        batch_num = batch_idx // RELATION_BATCH_SIZE + 1
        
        logger.info(f"  [批次 {batch_num}/{total_rel_batches}] 处理 {len(batch_rels)} 个关系...")
        
        for rel in batch_rels:
            src_cui = rel['src']
            tgt_cui = rel['tgt']
            rel_type = rel['type']
            rel_desc = rel['description']
            
            create_rel_query = """
            MATCH (a {cui: $src_cui, gid: $gid})
            MATCH (b {cui: $tgt_cui, gid: $gid})
            MERGE (a)-[r:`%s`]->(b)
            ON CREATE SET 
                r.description = $description,
                r.source = $source
            RETURN r
            """ % rel_type
            
            try:
                result = n4j.query(create_rel_query, {
                    'src_cui': src_cui,
                    'tgt_cui': tgt_cui,
                    'gid': gid,
                    'description': rel_desc,
                    'source': rel.get('source', 'UMLS')
                })
                if result:
                    created_rels += 1
            except Exception as e:
                pass  # 有些关系可能找不到对应节点
        
        gc.collect()
        logger.info(f"    ✅ 批次完成: {created_rels}/{len(relationships[:batch_idx + RELATION_BATCH_SIZE])}")
    
    logger.info(f"  ✅ 关系创建完成: {created_rels}/{len(relationships)}")


def import_umls_csv_to_neo4j(file_path: str, gid: str, n4j) -> bool:
    """
    完整的 UMLS CSV 导入流程
    
    Args:
        file_path: CSV 文件路径
        gid: 图ID
        n4j: Neo4j 连接
    
    Returns:
        bool: 是否成功
    """
    logger.info(f"\n[UMLS CSV 导入] {os.path.basename(file_path)}")
    logger.info(f"GID: {gid[:8]}...")
    
    try:
        # 1. 解析 CSV
        entities, relationships = parse_umls_csv(file_path)
        
        if not entities:
            logger.warning("  ⚠️  没有找到有效实体，跳过")
            return False
        
        # 2. 写入 Neo4j
        create_umls_nodes_and_relationships(n4j, entities, relationships, gid)
        
        logger.info(f"  ✅ UMLS 数据导入完成")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
