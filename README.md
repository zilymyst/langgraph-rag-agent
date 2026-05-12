# Agent 方向任务 2：Master Stacks

基于 RAG（检索增强生成）的智能问答系统。用户提问后，系统从向量数据库中检索相关资料，结合 DeepSeek 大模型生成基于资料的准确回答。

## 功能

- 从本地 TXT 文件加载知识库，自动分段存入向量数据库
- 支持 CLI 交互式问答，可连续提问
- 基于 LangGraph 的节点编排，流程清晰可扩展
- 云端 Embedding API 加速向量检索

## 技术栈

| 技术 | 用途 |
|:---|:---|
| LangGraph | Agent 流程编排 |
| LiteLLM | LLM API 网关 |
| Chroma | 向量数据库 |
| MarkItDown | 文档解析 |
| DeepSeek | 大语言模型 |
| SiliconFlow | 云端 Embedding API |
| uv | Python 项目管理 |

## 项目结构
agent-master-stacks/
├── main.py # 主程序
├── test.txt # 知识库文件（用户自行准备）
├── .env # 环境变量配置（需自行创建）
├── pyproject.toml # 项目配置
├── uv.lock # 依赖锁定文件
└── chroma_db/ # Chroma 向量数据（自动生成）

text

## 运行方式

1. 安装 uv：`pip install uv`
2. 克隆项目：`git clone https://github.com/zilymyst/agent-master-stacks.git`
3. 进入项目：`cd agent-master-stacks`
4. 安装依赖：`uv sync`
5. 创建 `.env` 文件，配置以下环境变量：
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
SILICONFLOW_API_KEY=你的硅基流动_API_Key

text
6. 在项目目录下创建 `test.txt`，写入知识库内容（段落用空行分隔）
7. 运行：`python main.py`
