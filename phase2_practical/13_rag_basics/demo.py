"""
LangChain 1.0 - RAG Basics 演示（非交互式）
===========================================

快速演示所有 RAG 组件，无需按 Enter 确认
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
    print("Pinecone 相关示例将被跳过\n")
    PINECONE_API_KEY = None

model = init_chat_model("groq:llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - RAG Basics 快速演示")
    print("=" * 70)

    # 示例 1: 文档加载
    print("\n[1/6] 文档加载...")
    sample_text = """LangChain 是一个用于构建 LLM 应用的框架。

它提供了以下核心组件：
1. Models - 语言模型接口
2. Prompts - 提示词模板
3. Chains - 链式调用
4. Agents - 智能代理

RAG (Retrieval-Augmented Generation) 是 LangChain 的核心应用场景之一。"""

    doc_path = DATA_DIR / "langchain_intro.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()
    print(f"  [OK] 加载了 {len(documents)} 个文档")

    # 示例 2: 文本分割
    print("\n[2/6] 文本分割...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"  [OK] 分割为 {len(chunks)} 个块")

    # 示例 3: 向量嵌入
    print("\n[3/6] 向量嵌入 (首次运行会下载模型)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector = embeddings.embed_query("LangChain 是什么")
    print(f"  [OK] 向量维度: {len(vector)}")

    # 示例 4-6: Pinecone 相关
    if PINECONE_API_KEY:
        print("\n[4/6] Pinecone 设置...")
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index_name = "langchain-rag-demo"
            dimension = 384

            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if index_name in existing_indexes:
                print(f"  [OK] 索引已存在")
                index = pc.Index(index_name)
            else:
                print(f"  创建新索引...")
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(10)
                index = pc.Index(index_name)
                print(f"  [OK] 索引创建完成")

            print("\n[5/6] 文档索引...")
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=index_name
            )
            print(f"  [OK] {len(chunks)} 个文档块已索引")

            print("\n[6/6] RAG 问答...")
            @tool
            def search_knowledge_base(query: str) -> str:
                """在知识库中搜索相关信息"""
                docs = vectorstore.similarity_search(query, k=2)
                return "\n\n".join([doc.page_content for doc in docs])

            from langchain.agents import create_agent
            agent = create_agent(
                model=model,
                tools=[search_knowledge_base],
                system_prompt="你是一个助手，可以访问知识库。使用 search_knowledge_base 工具搜索相关信息，然后回答问题。"
            )

            question = "LangChain 有哪些核心组件？"
            print(f"\n  问题: {question}")
            try:
                response = agent.invoke({"messages": [{"role": "user", "content": question}]})
                print(f"  回答: {response['messages'][-1].content}")
                print(f"\n  [OK] RAG 问答完成")
            except Exception as e:
                print(f"  [错误] RAG 问答失败（Groq 工具调用问题）")
                print(f"  提示: 这是 Groq 处理中文工具调用的偶发问题，不影响其他功能")

        except Exception as e:
            print(f"  [错误] Pinecone 操作失败: {e}")
    else:
        print("\n[4-6] 跳过 Pinecone 相关示例（未设置 API key）")

    print("\n" + "=" * 70)
    print(" 演示完成！")
    print("=" * 70)
    print("\n完整功能请运行: python main.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
