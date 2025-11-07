# 14 - RAG Advanced (RAG 进阶)

## 快速开始

```bash
# 1. 安装额外依赖
pip install rank_bm25 chromadb langchain-classic

# 2. 运行完整示例
cd phase2_practical
python 14_rag_advanced/main.py

# 3. 运行测试（无需 API key）
python 14_rag_advanced/test.py
```

**重要提示（LangChain 1.0）**：
- EnsembleRetriever 在 LangChain 1.0 中已移至 `langchain-classic` 包
- 正确导入：`from langchain_classic.retrievers import EnsembleRetriever`
- ~~错误导入~~：~~`from langchain.retrievers import EnsembleRetriever`~~（已废弃）

## 核心概念

### 为什么需要进阶 RAG？

基础 RAG（只用向量搜索）的局限性：
- ❌ 精确匹配差：搜不到专有名词、版本号
- ❌ 关键词弱：对代码、配置查询效果不佳
- ❌ 鲁棒性低：查询表达不同，结果差异大

**混合检索解决方案**：向量搜索 + BM25 = 全面覆盖

## 1. 向量检索 vs BM25 ���索

### 向量检索（Vector Search / Semantic Search）

**原理**：将文本转为向量，计算余弦相似度

**优势**：
- ✅ 理解语义和同义词
- ✅ 处理概念性查询
- ✅ 跨语言查询

**劣势**：
- ❌ 精确匹配差
- ❌ 对专有名词不敏感

**示例**：
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = vector_retriever.invoke("LangChain 的核心功能")
```

### BM25 检索（Keyword Search）

**原理**：基于词频(TF-IDF的改进版)，计算词的重要性

**优势**：
- ✅ 精确匹配专有名词
- ✅ 代码、版本号查询准确
- ✅ 速度快，无需嵌入

**劣势**：
- ❌ 不理解语义
- ❌ 同义词无法匹配

**示例**：
```python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

results = bm25_retriever.invoke("langchain>=1.0.0")
```

### 对比测试

| 查询类型 | 查询示例 | 向量搜索 | BM25 | 混合检索 |
|---------|---------|---------|------|---------|
| 语义查询 | "LangChain 的主要功能" | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| 精确匹配 | "langchain>=1.0.0" | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 专有名词 | "BM25 算法" | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 概念查询 | "如何优化性能" | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| 代码查询 | "@tool def search" | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## 2. EnsembleRetriever（混合检索器）

### 什么是 Ensemble Retriever？

组合多个检索器，使用 **RRF (Reciprocal Rank Fusion)** 算法融合结果。

### RRF 算法原理

```
对于文档 d：
  BM25 排名: rank_bm25(d)
  向量排名: rank_vector(d)

  RRF 得分 = w1 / (k + rank_bm25) + w2 / (k + rank_vector)

  其中：
  - w1, w2 是权重
  - k 是常数（通常为 60）
```

### 基本用法

```python
from langchain_classic.retrievers import EnsembleRetriever

# 创建混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # 平衡权重
)

# 使用
results = ensemble_retriever.invoke("查询文本")
```

### 权重配置

```python
# 1. 平衡（默认推荐）
weights=[0.5, 0.5]

# 2. 偏向语义（适合：文章、对话）
weights=[0.4, 0.6]  # BM25 40%, 向量 60%

# 3. 偏向精确匹配（适合：代码、配置）
weights=[0.6, 0.4]  # BM25 60%, 向量 40%

# 4. 纯向量（基础 RAG）
weights=[0.0, 1.0]

# 5. 纯 BM25（传统搜索）
weights=[1.0, 0.0]
```

## 3. 完整实现

### 离线阶段：建立索引

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# 1. 加载和分割
loader = TextLoader("docs.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 2. 创建向量检索器
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 4. 创建混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # 稍偏向语义
)
```

### 在线阶段：RAG 问答

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

# 1. 创建检索工具
@tool
def search_docs(query: str) -> str:
    """在文档库中搜索相关信息"""
    docs = ensemble_retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# 2. 创建 Agent
agent = create_agent(
    model=model,
    tools=[search_docs],
    system_prompt="你是助手。使用 search_docs 搜索信息，然后回答问题。"
)

# 3. 问答
response = agent.invoke({
    "messages": [{"role": "user", "content": "LangChain 有什么特性？"}]
})
```

## 4. 性能优化

### 4.1 权重调整策略

```python
# 根据数据类型选择权重
def get_optimal_weights(data_type):
    weights_map = {
        "technical_docs": [0.4, 0.6],  # 偏向语义
        "code_base": [0.6, 0.4],       # 偏向精确
        "mixed": [0.5, 0.5],           # 平衡
        "conversation": [0.3, 0.7],    # 强语义
    }
    return weights_map.get(data_type, [0.5, 0.5])
```

### 4.2 检索数量 (k 值)

```python
# k 值选择
- k=1: 只要最相关的（快，但可能不全面）
- k=3: 推荐（平衡速度和覆盖率）
- k=5: 更全面（但增加噪音和 token 成本）
- k=10: 大量上下文（慢，成本高）
```

### 4.3 监控和评估

```python
# 评估检索质量
def evaluate_retrieval(retriever, query, expected_content):
    results = retriever.invoke(query)

    # 检查是否包含预期内容
    for doc in results:
        if expected_content in doc.page_content:
            return True, results[0].page_content

    return False, results[0].page_content if results else ""

# 测试
success, result = evaluate_retrieval(
    ensemble_retriever,
    "LangChain 1.0 新特性",
    "LangChain 1.0"
)
```

## 5. 常见问题

### Q1: 混合检索一定比单一检索好吗？

**A**: 大多数情况是，但不绝对。

**适合混合**：
- ✅ 查询类型多样（语义 + 精确）
- ✅ 文档包含代码、配置、术语
- ✅ 用户查询质量参差不齐

**可能不需要**：
- 纯对话场景（全语义）
- 纯代码搜索（全精确）
- 性能要求极高（BM25 更快）

### Q2: 如何选择权重？

**A**: 从 `[0.5, 0.5]` 开始，根据测试调整

```python
# 测试流程
1. 收集典型查询（10-20个）
2. 测试不同权重：[0.3,0.7], [0.5,0.5], [0.7,0.3]
3. 对比前3个结果的相关性
4. 选择最优配置
```

### Q3: BM25Retriever 需要额外依赖吗？

**A**: 是的，需要安装 `rank_bm25`

```bash
pip install rank_bm25
```

### Q4: Chroma vs Pinecone vs FAISS？

| 特性 | Chroma | Pinecone | FAISS |
|-----|--------|----------|-------|
| 部署 | 本地 | 云端 | 本地 |
| 速度 | 中 | 快 | 最快 |
| 扩展性 | 小规模 | 大规模 | 中规模 |
| 成本 | 免费 | 免费层 | 免费 |
| 易用性 | 高 | 高 | 中 |
| **推荐** | 开发 | 生产 | 离线/高性能 |

### Q5: 混合检索会慢吗？

**A**: 会慢一点，但影响小

```
单一向量搜索: ~50ms
单一 BM25:    ~10ms
混合检索:      ~60ms  (并行执行)

增加: ~10ms，可接受
```

### Q6: 如何处理大量文档？

```python
# 策略 1: 分层检索
1. 先用 BM25 快速过滤到 top-100
2. 再用向量搜索精选 top-3

# 策略 2: 预过滤
1. 用元数据过滤（日期、分类）
2. 在子集上做混合检索

# 策略 3: 缓存热门查询
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query):
    return ensemble_retriever.invoke(query)
```

## 6. 最佳实践

### 6.1 生产环境检查清单

```python
# ✅ 检查清单
1. [ ] 使用混合检索而非单一方法
2. [ ] 根据数据类型调整权重
3. [ ] 设置合适的 k 值（推荐 3-5）
4. [ ] 监控检索质量（定期评估）
5. [ ] 使用持久化向量存储
6. [ ] 缓存热门查询
7. [ ] 设置超时和重试机制
```

### 6.2 代码模板

```python
# 生产级混合 RAG 系统
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class HybridRAGSystem:
    def __init__(self, documents, weights=[0.5, 0.5]):
        # 初始化
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 向量存储
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

        # BM25
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 5

        # 混合检索
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=weights
        )

    @lru_cache(maxsize=100)
    def search(self, query: str, k: int = 3):
        """缓存的搜索"""
        try:
            results = self.ensemble_retriever.invoke(query)
            return results[:k]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def create_rag_agent(self, model):
        """创建 RAG Agent"""
        @tool
        def search_docs(query: str) -> str:
            """搜索文档"""
            docs = self.search(query)
            return "\n\n".join([doc.page_content for doc in docs])

        return create_agent(
            model=model,
            tools=[search_docs],
            system_prompt="你是助手。使用 search_docs 搜索信息。"
        )
```

### 6.3 测试套件

```python
# 测试混合检索效果
test_cases = [
    {
        "query": "LangChain 核心组件",
        "expected_keywords": ["Models", "Prompts", "Agents"],
        "type": "semantic"
    },
    {
        "query": "langchain>=1.0.0",
        "expected_keywords": ["版本", "1.0.0"],
        "type": "exact"
    },
    {
        "query": "BM25 算法原理",
        "expected_keywords": ["BM25", "词频", "TF-IDF"],
        "type": "hybrid"
    }
]

def run_tests(retriever):
    passed = 0
    for test in test_cases:
        results = retriever.invoke(test["query"])
        content = " ".join([doc.page_content for doc in results])

        if all(kw in content for kw in test["expected_keywords"]):
            passed += 1
            print(f"✓ {test['type']}: {test['query']}")
        else:
            print(f"✗ {test['type']}: {test['query']}")

    print(f"\n通过: {passed}/{len(test_cases)}")
```

## 7. 进一步学习

### 下一步主题

- **重排序 (Reranking)**: 使用 CrossEncoder 重新排序
- **查询优化**: Query rewriting, HyDE
- **元数据过滤**: 根据时间、分类过滤
- **多查询**: 生成多个查询变体
- **上下文压缩**: 减少无关信息

### 相关资源

- LangChain 文档: https://python.langchain.com/docs/how_to/ensemble_retriever/
- BM25 论文: https://en.wikipedia.org/wiki/Okapi_BM25
- RRF 算法: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

## 核心要点总结

1. **混合检索 = 向量 + BM25** - 结合两者优势
2. **EnsembleRetriever** - LangChain 的标准组合器
3. **RRF 算法** - 融合多个排名结果
4. **权重调整** - 根据数据类型优化
5. **生产就绪** - 监控、缓存、容错

混合检索是现代 RAG 系统的标准配置，能显著提升检索质量和鲁棒性！
