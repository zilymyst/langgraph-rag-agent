import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
from http import client
from litellm import completion

import litellm
litellm.set_verbose = False

import chromadb
import os
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
collection = chroma_client.get_or_create_collection(
    name="default",
    embedding_function=siliconflow_ef
)
# 3. 检查集合是否为空，如果为空则添加一些测试数据
if collection.count() == 0:
    print("正在从文件加载知识库...")

     # 读取 test.txt 文件
    md = MarkItDown()
    result = md.convert("test.txt")
    content = result.text_content

     # 按段落切分（空行分隔）
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    print(f"共提取到 {len(paragraphs)} 个段落")
    
    # 逐段存入 Chroma
    for i, p in enumerate(paragraphs):
        collection.add(
            documents=[p],
            ids=[f"doc_{i}"]
        )
    
    print("知识库加载完成！")
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
        model="deepseek/deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "你是一个擅长总结和提炼信息的助手。请根据提供的资料，用精炼的语言回答用户的问题。不要简单复述原文，要对信息进行归纳、总结和补充（如果资料充足）。如果资料不足以回答问题，请如实说明。"},
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
while True:
    user_input = input("\n🙋 请输入问题（输入 q 退出）：")
    if user_input == "q":
        break
    result = app.invoke({"query": user_input})
    print(f"📚 检索到的资料：{result['context']}")
    print(f"🤖 AI 回答：{result['answer']}")


