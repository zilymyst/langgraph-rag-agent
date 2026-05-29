# db/chroma.py

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from markitdown import MarkItDown
from dotenv import load_dotenv
import os

load_dotenv()

def get_collection():
    """
    返回 collection 实例，如果为空会自动初始化知识库
    """
    siliconflow_ef = OpenAIEmbeddingFunction(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        api_base="https://api.siliconflow.cn/v1",
        model_name="BAAI/bge-m3"
    )

    chroma_client = PersistentClient(path="./chroma_db")

    collection = chroma_client.get_or_create_collection(
        name="default",
        embedding_function=siliconflow_ef
    )

    if collection.count() == 0:
        print("正在从文件加载知识库...")

        md = MarkItDown()
        result = md.convert("test.txt")
        content = result.text_content

        paragraphs = [
            p.strip()
            for p in content.split("\n\n")
            if p.strip()
        ]

        print(f"共提取到 {len(paragraphs)} 个段落")

        for i, p in enumerate(paragraphs):
            collection.add(
                documents=[p],
                ids=[f"doc_{i}"]
            )

        print("知识库加载完成！")

    return collection