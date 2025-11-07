"""
LangChain 1.0 - RAG Basics (RAG 基础)
=====================================

本模块重点讲解：
1. 文档加载 (Document Loaders)
2. 文本分割 (Text Splitters)
3. 向量嵌入 (Embeddings)
4. 向量存储 (Vector Stores) - Pinecone 免费版
5. 检索 (Retrieval)
6. RAG 问答链
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from pinecone import Pinecone, ServerlessSpec
import time

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

# 确保 data 目录存在
DATA_DIR.mkdir(exist_ok=True)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here_replace_this":
    raise ValueError("请先设置 GROQ_API_KEY")

if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
    print("\n[警告] 未设置 PINECONE_API_KEY")
    print("如需运行 Pinecone 相关示例，请：")
    print("1. 访问 https://www.pinecone.io/ 注册免费账号")
    print("2. 获取 API Key")
    print("3. 在 .env 文件中设置 PINECONE_API_KEY=你的key")
    print("\n当前将跳过需要 Pinecone 的示例\n")

model = init_chat_model("groq:llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


# ============================================================================
# 示例 1：文档加载 - Document Loaders
# ============================================================================
def example_1_document_loaders():
    """
    示例1：文档加载

    Document Loaders 将各种数据源转换为 LangChain Document 对象
    """
    print("\n" + "="*70)
    print("示例 1：文档加载 - Document Loaders")
    print("="*70)

    # 创建示例文本文件
    sample_text = """LangChain 是一个用于构建 LLM 应用的框架。

它提供了以下核心组件：
1. Models - 语言模型接口
2. Prompts - 提示词模板
3. Chains - 链式调用
4. Agents - 智能代理
5. Memory - 记忆管理

LangChain 1.0 引入了重大改进，包括：
- 更简洁的 API
- 更好的性能
- 内置的 LangGraph 支持
- 强大的中间件系统

RAG (Retrieval-Augmented Generation) 是 LangChain 的核心应用场景之一。
它结合了检索和生成，让 LLM 能够访问外部知识库。"""

    # 保存到文件
    doc_path = DATA_DIR / "langchain_intro.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    print(f"\n[OK] 创建示例文档: {doc_path}")

    # 使用 TextLoader 加载
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()

    print(f"\n加载结果:")
    print(f"  文档数量: {len(documents)}")
    print(f"  第一个文档:")
    print(f"    内容长度: {len(documents[0].page_content)} 字符")
    print(f"    元数据: {documents[0].metadata}")
    print(f"    内容预览: {documents[0].page_content[:100]}...")

    print("\n关键点:")
    print("  - TextLoader 加载文本文件")
    print("  - 返回 Document 对象列表")
    print("  - Document 包含 page_content 和 metadata")
    print("\n其他常用 Loaders:")
    print("  - PyPDFLoader - 加载 PDF")
    print("  - WebBaseLoader - 爬取网页")
    print("  - CSVLoader - 加载 CSV")

    return documents


# ============================================================================
# 示例 2：文本分割 - Text Splitters
# ============================================================================
def example_2_text_splitters(documents):
    """
    示例2：文本分割

    将长文档分割成小块，便于嵌入和检索
    """
    print("\n" + "="*70)
    print("示例 2：文本分割 - Text Splitters")
    print("="*70)

    # 创建分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,        # 每块最大字符数
        chunk_overlap=50,      # 块之间的重叠字符数
        length_function=len,   # 计算长度的函数
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]  # 分割优先级
    )

    print("\n配置:")
    print(f"  chunk_size: 200 字符")
    print(f"  chunk_overlap: 50 字符（防止信息被截断）")
    print(f"  分割优先级: 段落 -> 行 -> 句子 -> 空格 -> 字符")

    # 分割文档
    chunks = splitter.split_documents(documents)

    print(f"\n分割结果:")
    print(f"  原文档数: {len(documents)}")
    print(f"  分割后: {len(chunks)} 块")

    print(f"\n前 3 块内容:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n  块 {i}:")
        print(f"    长度: {len(chunk.page_content)} 字符")
        print(f"    内容: {chunk.page_content[:80]}...")

    print("\n关键点:")
    print("  - chunk_size 控制块大小")
    print("  - chunk_overlap 防止信息被截断")
    print("  - separators 定义分割优先级")
    print("  - RecursiveCharacterTextSplitter 智能分割")

    return chunks


# ============================================================================
# 示例 3：向量嵌入 - Embeddings
# ============================================================================
def example_3_embeddings():
    """
    示例3：向量嵌入

    将文本转换为向量，用于相似度计算
    """
    print("\n" + "="*70)
    print("示例 3：向量嵌入 - Embeddings")
    print("="*70)

    print("\n使用 HuggingFace 免费模型:")
    print("  模型: sentence-transformers/all-MiniLM-L6-v2")
    print("  维度: 384")
    print("  特点: 小巧、快速、免费")

    # 创建嵌入模型（使用免费的 HuggingFace 模型）
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 嵌入单个文本
    text = "LangChain 是一个 LLM 应用框架"
    vector = embeddings.embed_query(text)

    print(f"\n嵌入示例:")
    print(f"  文本: {text}")
    print(f"  向量维度: {len(vector)}")
    print(f"  向量前 5 个值: {vector[:5]}")

    # 嵌入多个文本
    texts = [
        "LangChain 是一个框架",
        "Python 是一种编程语言",
        "LangChain 用于构建 LLM 应用"
    ]
    vectors = embeddings.embed_documents(texts)

    print(f"\n批量嵌入:")
    print(f"  文本数: {len(texts)}")
    print(f"  向量数: {len(vectors)}")
    print(f"  每个向量维度: {len(vectors[0])}")

    # 计算相似度（简单示例）
    import numpy as np

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    sim_01 = cosine_similarity(vectors[0], vectors[1])
    sim_02 = cosine_similarity(vectors[0], vectors[2])

    print(f"\n相似度计算:")
    print(f"  '{texts[0]}' vs '{texts[1]}': {sim_01:.4f}")
    print(f"  '{texts[0]}' vs '{texts[2]}': {sim_02:.4f}")
    print(f"  -> 相同主题的文本相似度更高")

    print("\n关键点:")
    print("  - embed_query() - 嵌入单个查询")
    print("  - embed_documents() - 批量嵌入文档")
    print("  - 使用免费的 HuggingFace 模型（无需 API key）")
    print("  - 向量可用于相似度搜索")

    return embeddings


# ============================================================================
# 示例 4：Pinecone 向量存储 - 创建索引
# ============================================================================
def example_4_pinecone_setup():
    """
    示例4：Pinecone 设置

    创建 Pinecone serverless 索引（免费层级）
    """
    print("\n" + "="*70)
    print("示例 4：Pinecone 向量存储 - 创建索引")
    print("="*70)

    if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
        print("\n[警告] 跳过：需要设置 PINECONE_API_KEY")
        return None, None

    # 初始化 Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 索引配置
    index_name = "langchain-rag-demo"
    dimension = 384  # 与 all-MiniLM-L6-v2 模型维度匹配

    print(f"\n索引配置:")
    print(f"  名称: {index_name}")
    print(f"  维度: {dimension}")
    print(f"  类型: Serverless (免费层级)")
    print(f"  区域: us-east-1 (AWS)")

    # 检查索引是否存在
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name in existing_indexes:
        print(f"\n[OK] 索引已存在，直接使用")
        index = pc.Index(index_name)
    else:
        print(f"\n创建新索引...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",  # 相似度度量
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # 免费层级可用区域
            )
        )

        # 等待索引就绪
        print("等待索引初始化...")
        time.sleep(10)
        index = pc.Index(index_name)
        print("[OK] 索引创建完成")

    # 获取索引统计
    stats = index.describe_index_stats()
    print(f"\n索引统计:")
    print(f"  向量数: {stats.get('total_vector_count', 0)}")
    print(f"  维度: {stats.get('dimension', 'N/A')}")

    print("\n关键点:")
    print("  - Pinecone 提供免费 serverless 层级")
    print("  - dimension 必须与 embedding 模型匹配")
    print("  - metric='cosine' 用于相似度计算")
    print("  - ServerlessSpec 配置云和区域")

    # 创建 embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return index_name, embeddings


# ============================================================================
# 示例 5：文档索引 - 存入向量数据库
# ============================================================================
def example_5_index_documents(index_name, embeddings, chunks):
    """
    示例5：文档索引

    将分割后的文档存入 Pinecone
    """
    print("\n" + "="*70)
    print("示例 5：文档索引 - 存入向量数据库")
    print("="*70)

    if not index_name or not embeddings:
        print("\n[警告] 跳过：需要 Pinecone 配置")
        return None

    print(f"\n准备索引 {len(chunks)} 个文档块...")

    # 使用 PineconeVectorStore.from_documents()
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    print(f"[OK] 文档已索引到 Pinecone")

    # 测试检索
    query = "LangChain 的核心组件是什么？"
    print(f"\n测试检索:")
    print(f"  查询: {query}")

    results = vectorstore.similarity_search(query, k=2)

    print(f"  返回 {len(results)} 个最相关的文档块:\n")
    for i, doc in enumerate(results, 1):
        print(f"  结果 {i}:")
        print(f"    内容: {doc.page_content[:100]}...")
        print()

    print("关键点:")
    print("  - from_documents() 自动嵌入并存储")
    print("  - similarity_search() 检索相似文档")
    print("  - k=2 返回最相关的 2 个结果")

    return vectorstore


# ============================================================================
# 示例 6：RAG 问答 - 使用检索工具
# ============================================================================
def example_6_rag_qa(vectorstore):
    """
    示例6：RAG 问答

    将向量存储转为工具，供 Agent 使用
    """
    print("\n" + "="*70)
    print("示例 6：RAG 问答 - 使用检索工具")
    print("="*70)

    if not vectorstore:
        print("\n[警告] 跳过：需要 Pinecone vectorstore")
        return

    # 创建检索工具
    @tool
    def search_knowledge_base(query: str) -> str:
        """在知识库中搜索相关信息"""
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])

    # 创建 Agent
    from langchain.agents import create_agent

    agent = create_agent(
        model=model,
        tools=[search_knowledge_base],
        system_prompt="""你是一个助手，可以访问知识库。
当用户提问时：
1. 使用 search_knowledge_base 工具搜索相关信息
2. 基于搜索结果回答问题
3. 如果知识库中没有相关信息，诚实告知"""
    )

    # 测试��答
    questions = [
        "LangChain 有哪些核心组件？",
        "RAG 是什么？",
        "LangChain 1.0 有什么改进？"
    ]

    for question in questions:
        print(f"\n问题: {question}")
        try:
            response = agent.invoke({"messages": [{"role": "user", "content": question}]})
            print(f"回答: {response['messages'][-1].content}")
        except Exception as e:
            print(f"[错误] 查询失败: {str(e)[:100]}...")
            print("提示: 这可能是 Groq 模型处理中文工具调用的偶发问题")
            print("解决方案: 1) 重试 2) 使用英文提问 3) 换用其他模型")
        print("-" * 70)

    print("\n关键点:")
    print("  - 将 vectorstore 封装为工具")
    print("  - Agent 自动调用工具检索")
    print("  - 基于检索结果生成答案")
    print("  - 这就是 RAG (检索增强生成)")


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - RAG Basics (RAG 基础)")
    print("="*70)

    try:
        # 1. 文档加载
        documents = example_1_document_loaders()
        input("\n按 Enter 继续...")

        # 2. 文本分割
        chunks = example_2_text_splitters(documents)
        input("\n按 Enter 继续...")

        # 3. 向量嵌入
        embeddings = example_3_embeddings()
        input("\n按 Enter 继续...")

        # 4. Pinecone 设置
        index_name, embeddings = example_4_pinecone_setup()
        input("\n按 Enter 继续...")

        # 5. 文档索引
        vectorstore = example_5_index_documents(index_name, embeddings, chunks)
        input("\n按 Enter 继续...")

        # 6. RAG 问答
        example_6_rag_qa(vectorstore)

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  1. Document Loaders - 加载文档")
        print("  2. Text Splitters - 分割文本")
        print("  3. Embeddings - 向量嵌入")
        print("  4. Vector Stores - 向量存储（Pinecone）")
        print("  5. Similarity Search - 相似度检索")
        print("  6. RAG - 检索增强生成")
        print("\nRAG 工作流:")
        print("  文档 -> 分割 -> 嵌入 -> 存储")
        print("  查询 -> 检索 -> 提供给 LLM -> 生成答案")
        print("\n下一步：")
        print("  14_rag_advanced - RAG 进阶（混合搜索、重排序）")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
