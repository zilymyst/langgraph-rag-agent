from http import client
from litellm import completion
import chromadb
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    query: str
    context: str
    answer: str

# 加载环境变量
load_dotenv()

# 1. 初始化 ChromaDB 客户端
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. 获取或创建集合
collection = chroma_client.get_or_create_collection(name="default")

# 3. 检查集合是否为空，如果为空则添加一些测试数据
if collection.count() == 0:
    print("正在为集合添加初始数据...")
    collection.add(
        documents=["LangGraph 是一个用于构建循环多智能体系统的库，它构建在 LangChain 之上。"],
        ids=["id1"]
    )

# 4. 执行查询
def retrieve(state):
    query = state["query"]
    search_result = collection.query(
        query_texts=[query],
        n_results=1
    )
    if search_result["documents"] and search_result["documents"][0]:
        context = search_result["documents"][0][0]
    else:
        context = "未找到相关资料。"
    return {"context": context}

def generate(state):
    query = state["query"]
    context = state["context"]
    
    response = completion(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个有用的助手。请基于提供的资料回答问题。"},
            {"role": "user", "content": f"资料：{context}\n\n问题：{query}"},
        ],
        api_key=os.environ.get('DEEPSEEK_API_KEY')
    )
    
    answer = response.choices[0].message.content
    return {"answer": answer}
# ===== 5. 构建 LangGraph 图 =====
workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# ===== 6. 运行 =====
result = app.invoke({"query": "什么是 LangGraph？"})
print(f"📚 检索到的资料：{result['context']}")
print(f"🤖 AI 回答：{result['answer']}")

# ===== 6. 运行 =====
result = app.invoke({"query": "什么是 LangGraph？"})
print(f"📚 检索到的资料：{result['context']}")
print(f"🤖 AI 回答：{result['answer']}")



