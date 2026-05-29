from db.chroma import get_collection
import jieba

# 获取 Chroma collection
collection = get_collection()

# 停用词
STOP_WORDS = {
    "什么",
    "是",
    "的",
    "了",
    "吗",
    "一个",
    "一种",
    " ",
}

# 分词函数
def split_query(query):
    words = jieba.cut(query)

    filtered_words = [
        w for w in words
        if w.strip() and w not in STOP_WORDS
    ]

    return filtered_words


# 关键词打分函数
def score_document(doc, words):
    score = 0

    for w in words:

        # 大小写统一
        if w.lower() in doc.lower():

            # 长词更重要
            if len(w) >= 2:
                score += 2
            else:
                score += 1

    return score


# 检索主函数
def retrieve(query):

    # LangGraph 传入的是 state dict
    if isinstance(query, dict):
        query_text = query.get("query", "")
    else:
        query_text = str(query)

    print("\n========== RETRIEVE ==========")

    # 用户问题
    print("用户问题:", query_text)

    # ===== 一、向量检索 =====
    vector_result = collection.query(
        query_texts=[query_text],
        n_results=2
    )["documents"][0]

    print("向量检索结果:", vector_result)

    # ===== 二、关键词检索 =====
    words = split_query(query_text)

    print("过滤后分词:", words)

    all_docs = collection.get(include=["documents"])["documents"]

    keyword_scores = []

    for doc in all_docs:

        score = score_document(doc, words)

        if score > 0:
            keyword_scores.append((doc, score))

    # 按分数排序
    keyword_scores = sorted(
        keyword_scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("关键词得分:", keyword_scores)

    # 取前2
    keyword_result = [
        x[0]
        for x in keyword_scores[:2]
    ]

    print("关键词检索结果:", keyword_result)

    # ===== 三、Hybrid 合并 =====
    merged_docs = vector_result + keyword_result

    # 去重
    merged_docs = list(dict.fromkeys(merged_docs))

    print("Hybrid 合并结果:", merged_docs)

    # ===== 四、拼接 context =====
    context_text = "\n\n".join(merged_docs)

    print("最终 context:", context_text)

    print("========== END ==========\n")

    # LangGraph 节点必须返回 dict
    return {
        "context": context_text
    }