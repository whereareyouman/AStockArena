"""
News Deduplicator - 新闻去重工具
使用 all-MiniLM-L6-v2 嵌入模型对新闻标题或摘要进行相似度计算并去重
"""
import os
import warnings
from typing import List, Dict, Any
import numpy as np

# 延迟导入，避免没有安装 sentence-transformers 时启动失败
_model = None
_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

def _get_model():
    """延迟加载句子嵌入模型"""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"🔧 加载句子嵌入模型: {_model_name}")
            _model = SentenceTransformer(_model_name)
            print(f"✅ 模型加载成功")
        except ImportError:
            print("⚠️ 未安装 sentence-transformers，请运行: pip install sentence-transformers")
            raise
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            raise
    return _model


def deduplicate_news_by_embedding(
    news_list: List[Dict[str, Any]],
    similarity_threshold: float = 0.85,
    field_to_compare: str = 'title'
) -> List[Dict[str, Any]]:
    """
    使用嵌入向量相似度对新闻进行去重
    
    Args:
        news_list: 新闻列表，每条新闻是一个字典
        similarity_threshold: 相似度阈值，默认0.85，超过此值的被视为重复
        field_to_compare: 用于比较的字段，默认'title'，也可以是'summary'或'content'
    
    Returns:
        去重后的新闻列表
    """
    if not news_list:
        return []
    
    # 检查字段是否存在
    if not all(field_to_compare in news for news in news_list):
        print(f"⚠️ 部分新闻缺少字段 '{field_to_compare}'，跳过去重")
        return news_list
    
    try:
        model = _get_model()

        # 提取需要比较的文本（优先 title，其次 summary，再次 content）
        texts: List[str] = []
        for news in news_list:
            val = news.get(field_to_compare)
            if not val:
                # 自动回退到 summary / content
                val = news.get('summary') or news.get('content') or ''
            texts.append(str(val).strip())

        # 过滤空文本
        valid_indices = [i for i, text in enumerate(texts) if text]
        if not valid_indices:
            return news_list

        valid_texts = [texts[i] for i in valid_indices]
        valid_news = [news_list[i] for i in valid_indices]

        print(f"🔍 开始对 {len(valid_texts)} 条新闻进行嵌入去重（阈值={similarity_threshold}）")

        # 优先使用 PyTorch，避免 NumPy 依赖导致的问题（如 torch<->numpy 桥接失败）
        use_torch = False
        try:
            # 抑制 torch 初始化阶段可能出现的与 NumPy 相关的警告
            warnings.filterwarnings(
                "ignore",
                message=".*Failed to initialize NumPy.*",
                module="torch.*",
            )
            import torch  # type: ignore
            use_torch = True
        except Exception:
            use_torch = False

        if use_torch:
            # 使用 torch 张量计算相似度，完全绕过 numpy
            embeddings = model.encode(valid_texts, convert_to_numpy=False, show_progress_bar=False)
            import torch  # type: ignore
            if isinstance(embeddings, list):
                embeddings = torch.stack(embeddings)
            elif not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings)

            embeddings = embeddings.to(dtype=torch.float32)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            sim = embeddings @ embeddings.T

            keep_indices = []
            duplicate_count = 0
            for i in range(len(valid_texts)):
                is_duplicate = False
                for j in keep_indices:
                    if sim[i, j].item() >= float(similarity_threshold):
                        is_duplicate = True
                        duplicate_count += 1
                        break
                if not is_duplicate:
                    keep_indices.append(i)
        else:
            # 回退到 numpy 路径（可能在部分环境下触发 numpy/torch 兼容问题）
            embeddings = model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            sim = np.dot(embeddings, embeddings.T)

            keep_indices = []
            duplicate_count = 0
            for i in range(len(valid_texts)):
                is_duplicate = False
                for j in keep_indices:
                    if float(sim[i, j]) >= float(similarity_threshold):
                        is_duplicate = True
                        duplicate_count += 1
                        break
                if not is_duplicate:
                    keep_indices.append(i)

        # 构建去重后的结果
        deduplicated = [valid_news[i] for i in keep_indices]

        print(f"✅ 去重完成: 原始 {len(valid_news)} 条 → 保留 {len(deduplicated)} 条（移除 {duplicate_count} 条重复）")

        # 将未参与比较的新闻（空文本）也加回去
        for i, news in enumerate(news_list):
            if i not in valid_indices:
                deduplicated.append(news)

        return deduplicated

    except Exception as e:
        print(f"⚠️ 嵌入去重失败: {e}，返回原始列表")
        import traceback
        traceback.print_exc()
        return news_list


def deduplicate_news_dataframe(df, similarity_threshold: float = 0.85, field_to_compare: str = 'title'):
    """
    对 pandas DataFrame 格式的新闻数据进行去重
    
    Args:
        df: pandas DataFrame，包含新闻数据
        similarity_threshold: 相似度阈值，默认0.85
        field_to_compare: 用于比较的字段，默认'title'
    
    Returns:
        去重后的 DataFrame
    """
    if df is None or df.empty:
        return df
    
    if field_to_compare not in df.columns:
        print(f"⚠️ DataFrame 缺少字段 '{field_to_compare}'，跳过去重")
        return df
    
    # 转换为字典列表
    news_list = df.to_dict('records')
    
    # 去重
    deduplicated_list = deduplicate_news_by_embedding(
        news_list, 
        similarity_threshold=similarity_threshold,
        field_to_compare=field_to_compare
    )
    
    # 转回 DataFrame
    import pandas as pd
    return pd.DataFrame(deduplicated_list)


if __name__ == "__main__":
    # 测试用例
    test_news = [
        {"title": "科创板平均股价39.44元，8股股价超300元", "symbol": "SH688008"},
        {"title": "科创板平均股价39.44元 8股股价超300元", "symbol": "SH688111"},  # 相似度很高
        {"title": "深沪北百元股数量达153只，电子行业占比最高", "symbol": "SH688008"},
        {"title": "中芯国际：终止出售中芯宁波股权", "symbol": "SH688981"},
        {"title": "中芯国际终止出售中芯宁波14.832%股权", "symbol": "SH688981"},  # 相似度高
    ]
    
    print("\n" + "="*80)
    print("测试新闻去重功能")
    print("="*80 + "\n")
    
    result = deduplicate_news_by_embedding(test_news, similarity_threshold=0.85)
    
    print("\n原始新闻:")
    for i, news in enumerate(test_news, 1):
        print(f"{i}. {news['title']}")
    
    print("\n去重后:")
    for i, news in enumerate(result, 1):
        print(f"{i}. {news['title']}")
