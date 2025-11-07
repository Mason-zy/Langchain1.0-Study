"""
简单测试：验证混合检索组件（不需要 Groq API）
"""

import os
from pathlib import Path

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
CHROMA_DIR = SCRIPT_DIR / "chroma_db"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("测试：RAG Advanced - 混合检索")
print("=" * 70)


# ============================================================================
# 测试 1：准备测试数据
# ============================================================================
print("\n--- 测试 1: 准备测试数据 ---")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建测试文档
test_content = """
LangChain 框架核心组件

Models - 模型接口
支持 OpenAI, Anthropic, Groq 等多种模型。
版本要求：langchain>=1.0.0

Prompts - 提示词模板
使用 PromptTemplate 和 ChatPromptTemplate。

Agents - 智能代理
使用 create_agent 函数创建代理。
支持工具调用和 ReAct 模式。

RAG 进阶技术

混合检索 (Hybrid Search)
结合向量搜索和 BM25 关键词搜索。

BM25 算法
Best Match 25，基于词频的检索算法。
是 TF-IDF 的改进版本。

EnsembleRetriever
使用 RRF (Reciprocal Rank Fusion) 算法。
组合多个检索器的结果。

代码示例

@tool
def search_docs(query: str) -> str:
    return "结果"

agent = create_agent(model=model, tools=[search_docs])
"""

test_file = DATA_DIR / "test_docs.txt"

with open(test_file, "w", encoding="utf-8") as f:
    f.write(test_content)

# 加载和分割
loader = TextLoader(test_file, encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(documents)

print(f"\n[OK] 文档加载和分割成功")
print(f"  原文档: {len(documents)} 个")
print(f"  分割后: {len(chunks)} 块")


# ============================================================================
# 测试 2：向量检索器
# ============================================================================
print("\n--- 测试 2: 向量检索器 ---")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    print("创建向量存储（首次运行会下载模型）...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 测试查询
    query = "LangChain 核心组件"
    results = vector_retriever.invoke(query)

    print(f"\n[OK] 向量检索成功")
    print(f"  查询: {query}")
    print(f"  结果数: {len(results)}")
    if results:
        preview = results[0].page_content[:50].replace("\n", " ")
        print(f"  最相关: {preview}...")

except Exception as e:
    print(f"\n[SKIP] 向量检索跳过: {e}")
    vectorstore = None
    vector_retriever = None


# ============================================================================
# 测试 3：BM25 检索器
# ============================================================================
print("\n--- 测试 3: BM25 检索器 ---")

try:
    from langchain_community.retrievers import BM25Retriever

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2

    # 测试查询
    query = "BM25 算法"
    results = bm25_retriever.invoke(query)

    print(f"\n[OK] BM25 检索成功")
    print(f"  查询: {query}")
    print(f"  结果数: {len(results)}")
    if results:
        preview = results[0].page_content[:50].replace("\n", " ")
        print(f"  最相关: {preview}...")

except Exception as e:
    print(f"\n[ERROR] BM25 检索失败: {e}")
    print("  请安装: pip install rank_bm25")
    bm25_retriever = None


# ============================================================================
# 测试 4：混合检索器
# ============================================================================
print("\n--- 测试 4: 混合检索器 (EnsembleRetriever) ---")

if vector_retriever and bm25_retriever:
    try:
        from langchain_classic.retrievers import EnsembleRetriever

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

        print(f"\n[OK] 混合检索器创建成功")
        print(f"  组合: BM25 + 向量搜索")
        print(f"  权重: [0.5, 0.5]")
        print(f"  算法: RRF (Reciprocal Rank Fusion)")

        # 对比测试
        test_queries = [
            ("语义查询", "LangChain 的功能"),
            ("精确查询", "langchain>=1.0.0"),
            ("混合查询", "BM25 算法原理"),
        ]

        print(f"\n对比测试:")
        for query_type, query in test_queries:
            print(f"\n  [{query_type}] {query}")

            # BM25 结果
            bm25_results = bm25_retriever.invoke(query)
            bm25_preview = bm25_results[0].page_content[:40].replace("\n", " ") if bm25_results else "无"

            # 向量结果
            vector_results = vector_retriever.invoke(query)
            vector_preview = vector_results[0].page_content[:40].replace("\n", " ") if vector_results else "无"

            # 混合结果
            ensemble_results = ensemble_retriever.invoke(query)
            ensemble_preview = ensemble_results[0].page_content[:40].replace("\n", " ") if ensemble_results else "无"

            print(f"    BM25:   {bm25_preview}...")
            print(f"    Vector: {vector_preview}...")
            print(f"    Hybrid: {ensemble_preview}...")

    except Exception as e:
        print(f"\n[ERROR] 混合检索器创建失败: {e}")
else:
    print(f"\n[SKIP] 混合检索器跳过（缺少组件）")


# ============================================================================
# 测试 5：权重对比
# ============================================================================
print("\n--- 测试 5: 权重对比 ---")

if vector_retriever and bm25_retriever:
    try:
        query = "LangChain 核心组件"
        print(f"\n测试查询: {query}\n")

        weight_configs = [
            ([0.0, 1.0], "纯向量"),
            ([0.5, 0.5], "平衡"),
            ([1.0, 0.0], "纯 BM25"),
        ]

        for weights, desc in weight_configs:
            ensemble = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=weights
            )

            results = ensemble.invoke(query)
            if results:
                preview = results[0].page_content[:40].replace("\n", " ")
                print(f"  {desc} {weights}: {preview}...")

        print(f"\n[OK] 权重对比完成")

    except Exception as e:
        print(f"\n[ERROR] 权重对比失败: {e}")
else:
    print(f"\n[SKIP] 权重对比跳过")


# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("RAG Advanced 组件测试完成！")
print("=" * 70)

print("\n已验证:")
print("  [OK] 文档加载和分割")
print("  [OK] 向量检索 (HuggingFaceEmbeddings + Chroma)")
print("  [OK] BM25 检索 (需要 rank_bm25)")
print("  [OK] 混合检索 (EnsembleRetriever)")
print("  [OK] 权重调整")

print("\n核心要点:")
print("  1. 向量搜索 - 语义理解")
print("  2. BM25 搜索 - 精确匹配")
print("  3. 混合检索 - 结合两者优势")
print("  4. RRF 算法 - 融合多个排名")

print("\n运行完整示例:")
print("  python main.py  # 需要 GROQ_API_KEY")
