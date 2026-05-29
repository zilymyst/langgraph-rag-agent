# LangGraph RAG Agent

基于 LangGraph + Chroma + Hybrid Retrieval 的 AI Agent 项目。  
用户提问后，系统结合向量数据库和关键词检索，生成基于资料的准确回答。

## 功能

- 从本地 TXT 文件加载知识库，自动分段存入向量数据库
- 支持 CLI 交互式问答，可连续提问
- LangGraph 节点编排：Router → Retrieve → Generate，流程清晰可扩展
- Hybrid Retrieval：向量检索 + 关键词检索
- stop words 过滤、大小写统一、长词加权
- 可扩展 Router 分类（聊天/知识问答/任务请求）

## 技术栈

| 技术 | 用途 |
|:---|:---|
| LangGraph | Agent 流程编排 |
| Chroma | 向量数据库 |
| MarkItDown | 文档解析 |
| jieba | 中文分词与关键词匹配 |
| DeepSeek | 大语言模型生成回答 |
| SiliconFlow | Embedding API，生成向量 |
| uv | Python 项目管理 |

## 项目结构
agent-project/
├── main.py # 主程序
├── rag/ # 检索相关模块
│ └── retrieve.py
├── db/ # 数据库相关
│ └── chroma.py
├── test.txt # 知识库文件（用户自行准备）
├── .env # 环境变量配置
├── pyproject.toml # 项目配置
├── uv.lock # 依赖锁定文件
└── chroma_db/ # Chroma 向量数据库（自动生成）

## 运行方式

1. 安装 uv：`pip install uv`
2. 克隆项目：`git clone https://github.com/zilymyst/langgraph-rag-agent.git`
3. 进入项目目录：`cd langgraph-rag-agent`
4. 安装依赖：`uv sync`
5. 创建 `.env` 文件，配置以下环境变量：
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
SILICONFLOW_API_KEY=你的硅基流动_API_Key
复制
6. 在项目目录下创建 `test.txt`，写入知识库内容（段落用空行分隔）
7. 运行项目：`python main.py`
