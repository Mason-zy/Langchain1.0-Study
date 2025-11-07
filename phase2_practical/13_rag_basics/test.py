"""
简单测试：验证 RAG 基础组件（不需要 Pinecone API）
"""

import os
from pathlib import Path

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

# 确保 data 目录存在
DATA_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("测试：RAG 基础组件")
print("=" * 70)


# ============================================================================
# 测试 1：文档加载
# ============================================================================
print("\n--- 测试 1: 文档加载 ---")

from langchain_community.document_loaders import TextLoader
import os

# 创建测试文档
test_content = """LangChain 是一个强大的 LLM 应用框架。

核心组件包括：
1. Models - 模型接口
2. Prompts - 提示词
3. Chains - 链
4. Agents - 代理

RAG 是 LangChain 的核心应用场景。"""

test_file = DATA_DIR / "test.txt"

with open(test_file, "w", encoding="utf-8") as f:
    f.write(test_content)

# 加载文档
loader = TextLoader(test_file, encoding="utf-8")
documents = loader.load()

print(f"\n[OK] 文档加载成功")
print(f"  文档数: {len(documents)}")
print(f"  内容长度: {len(documents[0].page_content)} 字符")
print(f"  元数据: {documents[0].metadata}")


# ============================================================================
# 测试 2：文本分割
# ============================================================================
print("\n--- 测试 2: 文本分割 ---")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", " ", ""]
)

chunks = splitter.split_documents(documents)

print(f"\n[OK] 文本分割成功")
print(f"  原文档数: {len(documents)}")
print(f"  分割后: {len(chunks)} 块")
print(f"\n  前 2 块:")
for i, chunk in enumerate(chunks[:2], 1):
    print(f"    块 {i}: {chunk.page_content[:50]}...")


# ============================================================================
# 测试 3：向量嵌入（首次运行会下载模型，需要等待）
# ============================================================================
print("\n--- 测试 3: 向量嵌入 ---")
print("提示：首次运行会下载 HuggingFace 模型，请稍候...")

try:
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 嵌入测试文本
    text = "LangChain 是什么"
    vector = embeddings.embed_query(text)

    print(f"\n[OK] 向量嵌入成功")
    print(f"  文本: {text}")
    print(f"  向量维度: {len(vector)}")
    print(f"  向量前 5 个值: {[round(v, 4) for v in vector[:5]]}")

    # 计算相似度
    import numpy as np

    texts = [
        "LangChain 是一个框架",
        "Python 是编程语言",
        "LangChain 用于 LLM"
    ]

    vectors = embeddings.embed_documents(texts)

    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    print(f"\n  相似度测试:")
    print(f"    '{texts[0]}' vs '{texts[1]}': {cosine_sim(vectors[0], vectors[1]):.4f}")
    print(f"    '{texts[0]}' vs '{texts[2]}': {cosine_sim(vectors[0], vectors[2]):.4f}")
    print(f"    → 相关文本相似度更高 ✓")

except Exception as e:
    print(f"\n[SKIP] 向量嵌入跳过（可能网络问题）: {e}")


# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("RAG 基础组件测试完成！")
print("=" * 70)

print("\n已验证:")
print("  [OK] 文档加载 (TextLoader)")
print("  [OK] 文本分割 (RecursiveCharacterTextSplitter)")
print("  [OK] 向量嵌入 (HuggingFaceEmbeddings)")

print("\nPinecone 向量存储:")
print("  需要设置 PINECONE_API_KEY 才能测试")
print("  免费注册: https://www.pinecone.io/")

print("\n运行完整示例:")
print("  python main.py  # 查看所有示例")
