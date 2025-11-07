"""
LangChain 1.0 - RAG Advanced (RAG 进阶)
=======================================

本模块重点讲解：
1. 混合搜索 (Hybrid Search) - BM25 + 向量搜索
2. EnsembleRetriever - 组合多个检索器
3. 检索对比 - 向量 vs 关键词 vs 混合
4. 参数优化 - 权重调整和 k 值选择
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from langchain.agents import create_agent

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
CHROMA_DIR = SCRIPT_DIR / "chroma_db"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here_replace_this":
    raise ValueError("请先设置 GROQ_API_KEY")

model = init_chat_model("groq:llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


# ============================================================================
# 示例 1：准备测试数据
# ============================================================================
def example_1_prepare_data():
    """
    示例1：准备测试数据

    创建包含多种信息的测试文档，用于演示不同检索方法的效果
    """
    print("\n" + "="*70)
    print("示例 1：准备测试数据")
    print("="*70)

    # 创建测试文档 - 包含技术术语、概念、代码等
    documents_text = """
# LangChain 框架详解

## 核心组件

LangChain 提供以下核心组件：

1. Models (模型接口)
   - 支持 OpenAI GPT-4, GPT-3.5
   - 支持 Anthropic Claude
   - 支持 Groq Llama 模型
   - 版本号：langchain>=1.0.0

2. Prompts (提示词模板)
   - PromptTemplate 类
   - ChatPromptTemplate 类
   - 支持变量插值

3. Chains (链式调用)
   - 已在 1.0 中废弃
   - 建议使用 LCEL (LangChain Expression Language)

4. Agents (智能代理)
   - create_agent 函数
   - 工具调用机制
   - ReAct 模式

5. Memory (记忆管理)
   - InMemorySaver 类
   - SQLite checkpointer
   - 对话历史管理

## RAG 技术栈

### 基础 RAG
- 文档加载：TextLoader, PyPDFLoader
- 文本分割：RecursiveCharacterTextSplitter
- 向量嵌入：HuggingFaceEmbeddings
- 向量存储：Pinecone, Chroma, FAISS

### 进阶 RAG
- 混合搜索：BM25 + 向量搜索
- EnsembleRetriever：组合多个检索器
- 重排序：Reranking 模型
- 查询优化：Query rewriting

## 代码示例

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    \"\"\"搜索文档\"\"\"
    return "结果"

agent = create_agent(
    model=model,
    tools=[search_docs],
    system_prompt="你是助手"
)
```

## 性能优化

1. Chunk 大小：建议 500-1000 字符
2. Chunk 重叠：10-20%
3. 检索数量：k=3-5
4. 混合搜索权重：向量 0.6, BM25 0.4

## 常见问题

Q: LangChain 1.0 有什么新特性？
A: 更简洁的 API，内置 LangGraph，改进的中间件系统

Q: 如何选择向量数据库？
A: Pinecone 适合生产，Chroma 适合开发，FAISS 适合离线

Q: BM25 是什么？
A: Best Match 25，一种基于词频的检索算法，适合精确匹配
"""

    # 保存文档
    doc_path = DATA_DIR / "langchain_guide.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(documents_text)

    print(f"\n[OK] 创建测试文档: {doc_path}")
    print(f"  文档长度: {len(documents_text)} 字符")

    # 加载和分割
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"\n分割结果:")
    print(f"  原文档: {len(documents)} 个")
    print(f"  分割后: {len(chunks)} 块")
    print(f"\n前 3 块示例:")
    for i, chunk in enumerate(chunks[:3], 1):
        preview = chunk.page_content[:60].replace("\n", " ")
        print(f"  块 {i}: {preview}...")

    return chunks


# ============================================================================
# 示例 2：向量检索器 (语义搜索)
# ============================================================================
def example_2_vector_retriever(chunks):
    """
    示例2：向量检索器

    使用向量嵌入进行语义搜索
    """
    print("\n" + "="*70)
    print("示例 2：向量检索器 (语义搜索)")
    print("="*70)

    print("\n创建向量存储...")
    print("  使用: HuggingFaceEmbeddings (all-MiniLM-L6-v2)")
    print("  存储: Chroma (本地)")

    # 创建嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 创建 Chroma 向量存储
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # 创建检索器
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    print(f"\n[OK] 向量检索器已创建")
    print(f"  检索数量: k=3")

    # 测试查询
    test_queries = [
        "LangChain 有哪些核心组件？",  # 语义查询
        "如何优化 RAG 性能？",          # 概念查询
        "BM25 算法",                    # 精确术语
    ]

    print(f"\n测试查询:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n  查询 {i}: {query}")
        results = vector_retriever.invoke(query)
        print(f"  结果数: {len(results)}")
        if results:
            preview = results[0].page_content[:80].replace("\n", " ")
            print(f"  最相关: {preview}...")

    return vector_retriever, vectorstore


# ============================================================================
# 示例 3：BM25 检索器 (关键词搜索)
# ============================================================================
def example_3_bm25_retriever(chunks):
    """
    示例3：BM25 检索器

    使用 BM25 算法进行关键词匹配
    """
    print("\n" + "="*70)
    print("示例 3：BM25 检索器 (关键词搜索)")
    print("="*70)

    print("\n什么是 BM25？")
    print("  全称: Best Match 25")
    print("  类型: 基于词频的检索算法")
    print("  优势: 精确匹配专有名词、代码、版本号")
    print("  原理: TF-IDF 的改进版本")

    print("\n创建 BM25 检索器...")

    # 创建 BM25 检索器
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    print(f"\n[OK] BM25 检索器已创建")
    print(f"  检索数量: k=3")

    # 测试查询
    test_queries = [
        "LangChain 有哪些核心组件？",  # 语义查询
        "如何优化 RAG 性能？",          # 概念查询
        "BM25 算法",                    # 精确术语
    ]

    print(f"\n测试查询:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n  查询 {i}: {query}")
        results = bm25_retriever.invoke(query)
        print(f"  结果数: {len(results)}")
        if results:
            preview = results[0].page_content[:80].replace("\n", " ")
            print(f"  最相关: {preview}...")

    return bm25_retriever


# ============================================================================
# 示例 4：混合检索器 (Ensemble Retriever)
# ============================================================================
def example_4_ensemble_retriever(vector_retriever, bm25_retriever):
    """
    示例4：混合检索器

    组合向量搜索和 BM25，使用 RRF (Reciprocal Rank Fusion) 算法
    """
    print("\n" + "="*70)
    print("示例 4：混合检索器 (Ensemble Retriever)")
    print("="*70)

    print("\n混合检索原理:")
    print("  1. 向量搜索: 擅长语义理解、同义词、概念")
    print("  2. BM25 搜索: 擅长精确匹配、专有名词、代码")
    print("  3. RRF 算法: 融合两者的排名结果")

    print("\n权重说明:")
    print("  weights=[0.5, 0.5] - 平衡")
    print("  weights=[0.7, 0.3] - 偏向向量")
    print("  weights=[0.3, 0.7] - 偏向 BM25")

    # 创建混合检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]  # 稍微偏向向量搜索
    )

    print(f"\n[OK] 混合检索器已创建")
    print(f"  组合: BM25 (40%) + Vector (60%)")
    print(f"  算法: RRF (Reciprocal Rank Fusion)")

    # 对比测试
    test_queries = [
        ("语义查询", "LangChain 的主要功能是什么？"),
        ("精确匹配", "langchain>=1.0.0"),
        ("混合查询", "BM25 算法如何工作？"),
    ]

    print(f"\n对比测试:")
    for query_type, query in test_queries:
        print(f"\n  [{query_type}] {query}")

        # BM25 结果
        bm25_results = bm25_retriever.invoke(query)
        bm25_preview = bm25_results[0].page_content[:50].replace("\n", " ") if bm25_results else "无"

        # 向量结果
        vector_results = vector_retriever.invoke(query)
        vector_preview = vector_results[0].page_content[:50].replace("\n", " ") if vector_results else "无"

        # 混合结果
        ensemble_results = ensemble_retriever.invoke(query)
        ensemble_preview = ensemble_results[0].page_content[:50].replace("\n", " ") if ensemble_results else "无"

        print(f"    BM25:    {bm25_preview}...")
        print(f"    Vector:  {vector_preview}...")
        print(f"    Hybrid:  {ensemble_preview}...")

    print("\n关键点:")
    print("  - 混合检索结合了两者的优势")
    print("  - 对大多数查询都能获得更好的结果")
    print("  - 适用于生产环境")

    return ensemble_retriever


# ============================================================================
# 示例 5：权重优化实验
# ============================================================================
def example_5_weight_optimization(vector_retriever, bm25_retriever):
    """
    示例5：权重优化

    测试不同权重配置的效果
    """
    print("\n" + "="*70)
    print("示例 5：权重优化实验")
    print("="*70)

    print("\n测试不同的权重配置...")

    weight_configs = [
        (0.0, 1.0, "纯向量"),
        (0.3, 0.7, "偏向向量"),
        (0.5, 0.5, "平衡"),
        (0.7, 0.3, "偏向 BM25"),
        (1.0, 0.0, "纯 BM25"),
    ]

    test_query = "LangChain 1.0 有什么新特性？"

    print(f"\n测试查询: {test_query}")
    print("\n权重配置对比:")

    for bm25_weight, vector_weight, description in weight_configs:
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[bm25_weight, vector_weight]
        )

        results = ensemble.invoke(test_query)
        if results:
            preview = results[0].page_content[:60].replace("\n", " ")
            print(f"\n  {description} [{bm25_weight:.1f}, {vector_weight:.1f}]:")
            print(f"    最相关: {preview}...")

    print("\n推荐配置:")
    print("  - 技术文档: [0.4, 0.6] - 稍偏向语义")
    print("  - 代码搜索: [0.6, 0.4] - 稍偏向精确匹配")
    print("  - 通用场景: [0.5, 0.5] - 平衡")


# ============================================================================
# 示例 6：RAG Agent with Hybrid Search
# ============================================================================
def example_6_rag_agent_hybrid(ensemble_retriever):
    """
    示例6：使用混合检索的 RAG Agent

    将混合检索集成到 Agent 中
    """
    print("\n" + "="*70)
    print("示例 6：RAG Agent with Hybrid Search")
    print("="*70)

    print("\n创建混合检索工具...")

    # 创建检索工具
    @tool
    def search_knowledge_base(query: str) -> str:
        """在知识库中搜索相关信息（混合检索）"""
        docs = ensemble_retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs[:2]])  # 只取前2个

    print(f"[OK] 工具已创建: search_knowledge_base")

    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=[search_knowledge_base],
        system_prompt="""你是一个 LangChain 专家助手。

使用 search_knowledge_base 工具搜索相关信息，然后回答问题。

注意：
1. 优先使用检索到的信息
2. 如果信息不足，诚实告知
3. 回答要简洁准确"""
    )

    print(f"[OK] Agent 已创建\n")

    # 测试问答
    questions = [
        "LangChain 有哪些核心组件？",
        "如何优化 RAG 性能？",
    ]

    for question in questions:
        print(f"问题: {question}")
        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            print(f"回答: {response['messages'][-1].content}\n")
        except Exception as e:
            print(f"[错误] 查询失败（Groq 工具调用问题）\n")
        print("-" * 70 + "\n")

    print("关键点:")
    print("  - 混合检索提供更全面的上下文")
    print("  - 同时覆盖语义和精确匹配")
    print("  - 提高 RAG 系统的准确性和鲁棒性")


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - RAG Advanced (RAG 进阶)")
    print("="*70)

    try:
        # 1. 准备数据
        chunks = example_1_prepare_data()
        input("\n按 Enter 继续...")

        # 2. 向量检索器
        vector_retriever, vectorstore = example_2_vector_retriever(chunks)
        input("\n按 Enter 继续...")

        # 3. BM25 检索器
        bm25_retriever = example_3_bm25_retriever(chunks)
        input("\n按 Enter 继续...")

        # 4. 混合检索器
        ensemble_retriever = example_4_ensemble_retriever(
            vector_retriever, bm25_retriever
        )
        input("\n按 Enter 继续...")

        # 5. 权重优化
        example_5_weight_optimization(vector_retriever, bm25_retriever)
        input("\n按 Enter 继续...")

        # 6. RAG Agent
        example_6_rag_agent_hybrid(ensemble_retriever)

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  1. 向量搜索 - 语义理解")
        print("  2. BM25 搜索 - 精确匹配")
        print("  3. 混合检索 - 结合两者优势")
        print("  4. EnsembleRetriever - 使用 RRF 算法")
        print("  5. 权重调整 - 根据场景优化")
        print("\n生产建议：")
        print("  - 默认使用混合检索")
        print("  - 权重根据数据类型调整")
        print("  - 监控检索质量并持续优化")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
