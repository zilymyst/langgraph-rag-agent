from statistics import mode

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
def retrieve(state):    
    print("当前 state:", state)
    mode = state.get("mode", "rag")
    print("当前 mode:", mode)
    
    

    query_text = state["query"]    
    print("\n========== RETRIEVE ==========")    
    print("用户问题:", query_text)

    # 向量检索    
    vector_result = collection.query(query_texts=[query_text], n_results=2)["documents"][0]

    # 关键词检索    
    words = split_query(query_text)    
    all_docs = collection.get(include=["documents"])["documents"]    
    keyword_scores = [(doc, score_document(doc, words)) for doc in all_docs if score_document(doc, words) > 0]    
    keyword_result = [x[0] for x in sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:2]]

    # Hybrid 合并    
    enriched_context = []    
    for doc in set(vector_result + keyword_result):        
        index = all_docs.index(doc)        
        start = max(0, index - 1)        
        end = min(len(all_docs), index + 2)        
        enriched_context.extend(all_docs[start:end])

    # 去重    
    enriched_context = list(dict.fromkeys(enriched_context))    
    context = "\n\n".join(enriched_context)

    
    print("最终 context:", context)    
    print("========== END ==========")

    return {"context": context}