import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
from litellm import completion, query

import chromadb
from rag.retrieve import retrieve
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict
from markitdown import MarkItDown
from chromadb.utils import embedding_functions

class State(TypedDict):
    query: str
    context: str
    answer: str

# 加载环境变量
load_dotenv()

siliconflow_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("SILICONFLOW_API_KEY"),
    api_base="https://api.siliconflow.cn/v1",
    model_name="BAAI/bge-large-zh-v1.5"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

#results = retrieve("你的查询内容")


def generate(state):
    query = state["query"]
    context = state["context"]
    
    if context == "__chat__":
        response = completion(
            model="deepseek/deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "你是一个友好的助手。请直接回答用户的闲聊问题。"},
                {"role": "user", "content": query}
            ],
            api_key=os.environ.get('DEEPSEEK_API_KEY')
        )
    else:
        response = completion(
        model="deepseek/deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "你是一个擅长总结和提炼信息的助手。请根据提供的资料，用精炼的语言回答用户的问题。不要简单复述原文，要对信息进行归纳、总结和补充（如果资料充足）。如果资料不足以回答问题，请如实说明。"},
            {"role": "user", "content": f"资料：{context}\n\n问题：{query}"},
        ],
        api_key=os.environ.get('DEEPSEEK_API_KEY')
    )
    
    answer = response.choices[0].message.content
    return {"answer": answer}
# ===== 查询路由节点 =====
def router(state):
    query = state["query"]
    chat_words = ["你好", "谢谢", "再见", "怎么样", "你是谁", "哈哈", "早上好", "晚上好", "hi", "hello", "hey", "在吗", "在不在"]
    if any(word in query for word in chat_words):
        return {"context": "__chat__"}
    else:
        return state
# ===== 构建 LangGraph 图 =====
workflow = StateGraph(State)

# 注册三个节点
workflow.add_node("router", router)       # 路由节点：判断问题类型
workflow.add_node("retrieve", retrieve)   # 检索节点：从 Chroma 查资料
workflow.add_node("generate", generate)   # 生成节点：调 LLM 生成回答

# 设置入口为 router（不再是 retrieve）
workflow.set_entry_point("router")

# 条件分支：根据 state["context"] 决定下一步
# - 如果 context 是 "__chat__"，跳过检索，直接走 generate
# - 否则走 retrieve
workflow.add_conditional_edges(
    "router",
    lambda state: "retrieve" if state.get("context") == "" or state.get("context") is None else "generate",
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

workflow.add_edge("retrieve", "generate")  # 检索完 → 生成
workflow.add_edge("generate", END)         # 生成完 → 结束

app = workflow.compile()
# =====  运行 =====
while True:
    user_input = input("\n🙋 请输入问题（输入 q 退出）：")
    if user_input == "q":
        break
    result = app.invoke({"query": user_input})
    # 闲聊时不显示"检索到的资料"（因为是 __chat__ 标记）
    if result["context"] != "__chat__":
        print(f"📚 检索到的资料：{result['context']}")
    
    print(f"🤖 AI 回答：{result['answer']}")

