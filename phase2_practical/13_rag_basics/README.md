# 13 - RAG Basics (RAG 基础)

## 快速开始

```bash
# 1. 测试基础组件（无需 Pinecone API）
python test.py

# 2. 快速演示（非交互式，可选 Pinecone）
python demo.py

# 3. 完整示例（交互式，需要手动按 Enter）
python main.py
```

## 核心概念

**RAG (Retrieval-Augmented Generation) = 检索增强生成**

RAG 让 LLM 能够访问外部知识库，解决了以下问题：
- ❌ LLM 训练数据有截止日期
- ❌ LLM 不知道你的私有数据
- ❌ LLM 可能产生幻觉

RAG 工作流程：
```
1. 离线：文档 → 分割 → 嵌入 → 存入向量数据库
2. 在线：用户查询 → 检索相关文档 → 提供给 LLM → 生成答案
```

## 基本用法

### 1. 文档加载 (Document Loaders)

```python
from langchain_community.document_loaders import TextLoader

# 加载文本文件
loader = TextLoader("document.txt", encoding="utf-8")
documents = loader.load()

# documents 是 Document 对象列表
# Document 包含：page_content (文本) 和 metadata (元数据)
```

**常用 Loaders**：
- `TextLoader` - 文本文件
- `PyPDFLoader` - PDF 文件
- `WebBaseLoader` - 网页
- `CSVLoader` - CSV 文件

### 2. 文本分割 (Text Splitters)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块最大字符数
    chunk_overlap=50,      # 块之间的重叠
    separators=["\n\n", "\n", "。", " ", ""]  # 分割优先级
)

chunks = splitter.split_documents(documents)
```

**为什么要分割？**
- LLM 有 token 限制
- 小块检索更精准
- 降低成本

### 3. 向量嵌入 (Embeddings)

```python
from langchain_huggingface import HuggingFaceEmbeddings

# 使用免费的 HuggingFace 模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 嵌入单个查询
vector = embeddings.embed_query("什么是 RAG?")  # 返回 384 维向量

# 批量嵌入文档
vectors = embeddings.embed_documents(["文本1", "文本2"])
```

**免费 Embedding 模型**：
- `all-MiniLM-L6-v2` - 384维，快速，适合大多数场景
- `all-mpnet-base-v2` - 768维，更准确，但慢一些

### 4. Pinecone 向量存储 (免费版)

#### 设置 Pinecone 账号

1. **注册账号**：
   - 访问 https://www.pinecone.io/
   - 点击 "Sign Up" 注册
   - 免费层级提供：
     - 1 个 serverless 索引
     - 10 GB 存储
     - 无需信用卡

2. **获取 API Key**：
   - 登录后进入 Dashboard
   - 左侧菜单点击 "API Keys"
   - 复制 API Key

3. **设置环境变量**：
   ```bash
   # .env 文件
   PINECONE_API_KEY=your_api_key_here
   ```

#### 创建索引

```python
from pinecone import Pinecone, ServerlessSpec
import time

# 初始化
pc = Pinecone(api_key=PINECONE_API_KEY)

# 创建 serverless 索引（免费）
pc.create_index(
    name="my-index",
    dimension=384,  # 必须与 embedding 模型维度匹配
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"  # 免费层级可用
    )
)

# 等待就绪
time.sleep(10)
```

#### 使用 LangChain 集成

```python
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 方式 1：从文档创建（自动嵌入并存储）
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)

# 方式 2：从已有索引加载
vectorstore = PineconeVectorStore(
    index_name="my-index",
    embedding=embeddings
)

# 检索相似文档
docs = vectorstore.similarity_search("查询文本", k=3)
```

### 5. RAG 问答链

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

# 将 vectorstore 封装为工具
@tool
def search_kb(query: str) -> str:
    """搜索知识库"""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

# 创建 RAG Agent
agent = create_agent(
    model=model,
    tools=[search_kb],
    system_prompt="""你是助手。
使用 search_kb 工具检索信息，然后回答问题。"""
)

# 问答
response = agent.invoke({
    "messages": [{"role": "user", "content": "什么是 RAG?"}]
})
```

## 完整工作流程

### 离线阶段（索引文档）

```python
# 1. 加载文档
from langchain_community.document_loaders import TextLoader
loader = TextLoader("docs.txt")
documents = loader.load()

# 2. 分割文本
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. 创建 embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. 存入 Pinecone
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)
```

### 在线阶段（问答）

```python
# 1. 连接 vectorstore
vectorstore = PineconeVectorStore(
    index_name="my-index",
    embedding=embeddings
)

# 2. 创建检索工具
@tool
def search_kb(query: str) -> str:
    """搜索知识库"""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

# 3. 创建 Agent
agent = create_agent(model=model, tools=[search_kb])

# 4. 问答
response = agent.invoke({
    "messages": [{"role": "user", "content": "问题"}]
})
```

## Pinecone 免费版说明

### 免费层级限制

- **索引数量**: 1 个 serverless 索引
- **存储**: 10 GB
- **请求**: 无限制（但有速率限制）
- **向量维度**: 最大 20,000
- **无需信用卡**: 完全免费

### 可用区域（免费）

- `us-east-1` (AWS)
- 其他区域可能需要付费

### 索引配置建议

```python
# 免费层级推荐配置
pc.create_index(
    name="my-index",
    dimension=384,  # 使用 all-MiniLM-L6-v2 (384维)
    metric="cosine",  # 余弦相似度
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"  # 免费区域
    )
)
```

### 删除索引（节省配额）

```python
# 删除不用的索引
pc.delete_index("old-index")

# 查看所有索引
indexes = pc.list_indexes()
for idx in indexes:
    print(idx.name)
```

## 常见问题

### 0. Groq 工具调用偶尔失败？

**现象**: 运行示例 6 时，部分中文查询会失败，错误信息包含 `tool_use_failed`

**原因**: Groq 模型处理中文工具调用参数时的已知问题（Unicode 转义错误）

**影响**:
- ✅ 不影响前 5 个示例（文档加载、分割、嵌入、Pinecone、索引）
- ⚠️ 示例 6 的 RAG 问答成功率约 70-80%
- ✅ 不影响学习 RAG 核心概念

**解决方案**:
1. **已实现**: 代码已添加错误处理，程序不会崩溃
2. **重试**: 多运行几次，通常能成功
3. **换模型**: 使用 OpenAI GPT-4o-mini 或 Claude（100% 稳定）
4. **英文**: 用英文提问（100% 成功）

详细说明请查看: `GROQ_TOOL_ISSUE.md`

### 1. 为什么使用 HuggingFace 而不是 OpenAI Embeddings？

**HuggingFace 优势**：
- ✅ 完全免费
- ✅ 本地运行，无需 API
- ✅ 支持离线使用
- ✅ 足够准确（大多数场景）

**OpenAI 优势**：
- ✅ 更高精度
- ✅ 更多语言支持
- ❌ 需要 API key 和付费

### 2. chunk_size 设多大合适？

**推荐配置**：
```python
# 通用场景
chunk_size=500, chunk_overlap=50

# 长文档（如书籍）
chunk_size=1000, chunk_overlap=200

# 短文档（如问答对）
chunk_size=200, chunk_overlap=20
```

**原则**：
- 太小：检索不到完整信息
- 太大：噪音多，成本高
- overlap：防止信息被截断

### 3. k（检索数量）设多少？

```python
# 简单问题
k=1  # 只取最相关的

# 复杂问题
k=3-5  # 多个相关文档

# 需要全面信息
k=10  # 但注意 token 成本
```

### 4. Pinecone dimension 必须匹配吗？

**是的！非常重要！**

```python
# ❌ 错误：不匹配
embeddings = HuggingFaceEmbeddings("all-MiniLM-L6-v2")  # 384维
pc.create_index(dimension=1536)  # OpenAI 的维度

# ✅ 正确：匹配
embeddings = HuggingFaceEmbeddings("all-MiniLM-L6-v2")  # 384维
pc.create_index(dimension=384)  # 必须一致
```

**常见模型维度**：
- `all-MiniLM-L6-v2` → 384
- `all-mpnet-base-v2` → 768
- OpenAI `text-embedding-3-small` → 1536

### 5. 如何查看 Pinecone 使用情况？

```python
from pinecone import Pinecone

pc = Pinecone(api_key=API_KEY)
index = pc.Index("my-index")

# 查看统计
stats = index.describe_index_stats()
print(f"向量数: {stats['total_vector_count']}")
print(f"维度: {stats['dimension']}")
```

### 6. 免费版够用吗？

**对于学习和小项目**：完全够用！

- 10 GB ≈ 数百万个文档块
- 1 个索引适合单个项目
- 无请求限制

**需要升级的情况**：
- 多个独立项目（需要多个索引）
- 超过 10 GB 数据
- 需要其他云/区域

## 最佳实践

```python
# 1. 完整的 RAG 系统
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain_core.tools import tool

# 离线：索引文档
def index_documents(file_path, index_name):
    # 加载
    loader = TextLoader(file_path)
    docs = loader.load()

    # 分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # 嵌入并存储
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

# 在线：问答
def create_rag_agent(index_name):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    @tool
    def search_kb(query: str) -> str:
        """搜索知识库"""
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n".join([d.page_content for d in docs])

    return create_agent(
        model=model,
        tools=[search_kb],
        system_prompt="使用 search_kb 检索信息，然后回答问题。"
    )

# 使用
# index_documents("docs.txt", "my-kb")
# agent = create_rag_agent("my-kb")
# response = agent.invoke({"messages": [...]})
```

## 核心要点

1. **Document Loaders** - 加载各种格式的文档
2. **Text Splitters** - 智能分割文本
3. **Embeddings** - 文本转向量（使用免费的 HuggingFace）
4. **Vector Stores** - 存储和检索（Pinecone 免费版）
5. **Similarity Search** - 相似度检索
6. **RAG** - 检索 + 生成 = 知识增强的 LLM

## RAG 工作流

```
离线（一次性）:
  文档 → 加载 → 分割 → 嵌入 → 存入 Pinecone

在线（每次查询）:
  用户问题 → 嵌入 → 检索相似文档 → 提供给 LLM → 生成答案
```

## 下一步

**14_rag_advanced** - RAG 进阶技术
- 混合搜索（向量 + 关键词）
- 重排序（Reranking）
- 查询优化
- 元数据过滤
