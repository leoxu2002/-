from typing import Any, List, Optional
import numpy as np
from src.libs.splitter.base_splitter import BaseSplitter


class SemanticSplitter(BaseSplitter):
    """基于语义相似度的智能分割器
    
    核心思想：
    1. 将文本分割为句子
    2. 计算每个句子的向量
    3. 计算相邻句子的语义相似度
    4. 在相似度低的位置（语义转折点）进行分割
    """
    
    def __init__(
        self,
        settings: Any,
        embedding_model: Optional[Any] = None,
        similarity_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.chunk_size = settings.ingestion.chunk_size
        self.chunk_overlap = settings.ingestion.chunk_overlap
        self.similarity_threshold = similarity_threshold
        
        # 加载嵌入模型（可插拔）
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            # 默认使用配置的嵌入模型
            from src.libs.embedding.embedding_factory import EmbeddingFactory
            self.embedding_model = EmbeddingFactory.create(settings)
    
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """基于语义相似度分割文本"""
        # 1. 验证输入
        self.validate_text(text)
        
        # 2. 分割为句子
        sentences = self._split_into_sentences(text)
        
        if len(sentences) == 1:
            return sentences
        
        # 3. 计算句子向量
        embeddings = self._compute_embeddings(sentences)
        
        # 4. 计算相邻句子相似度
        similarities = self._compute_similarities(embeddings)
        
        # 5. 找到分割点（相似度低的位置）
        split_points = self._find_split_points(similarities)
        
        # 6. 根据分割点合并句子
        chunks = self._merge_sentences(sentences, split_points)
        
        # 7. 验证输出
        self.validate_chunks(chunks)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """分割为句子"""
        import re
        
        # 中英文句子分割
        pattern = r'(?<=[。！？.!?])\s*'
        sentences = re.split(pattern, text)
        
        # 过滤空句子
        return [s.strip() for s in sentences if s.strip()]
    
    def _compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """计算句子向量"""
        # 批量编码
        embeddings = self.embedding_model.encode_batch(sentences)
        return np.array(embeddings)
    
    def _compute_similarities(self, embeddings: np.ndarray) -> List[float]:
        """计算相邻句子的余弦相似度"""
        from numpy.linalg import norm
        
        similarities = []
        for i in range(len(embeddings) - 1):
            # 余弦相似度
            cos_sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                norm(embeddings[i]) * norm(embeddings[i + 1])
            )
            similarities.append(cos_sim)
        
        return similarities
    
    def _find_split_points(self, similarities: List[float]) -> List[int]:
        """找到分割点（相似度低于阈值的位置）"""
        split_points = []
        
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_points.append(i + 1)  # 在低相似度后分割
        
        return split_points
    
    def _merge_sentences(
        self,
        sentences: List[str],
        split_points: List[int]
    ) -> List[str]:
        """根据分割点合并句子"""
        if not split_points:
            return [' '.join(sentences)]
        
        chunks = []
        start = 0
        
        for point in split_points:
            chunk = ' '.join(sentences[start:point])
            if len(chunk) <= self.chunk_size:
                chunks.append(chunk)
            else:
                # 如果块太大，回退到递归分割
                chunks.extend(self._fallback_split(chunk))
            start = point
        
        # 最后一块
        if start < len(sentences):
            chunk = ' '.join(sentences[start:])
            chunks.append(chunk)
        
        return chunks
    
    def _fallback_split(self, text: str) -> List[str]:
        """回退到递归分割"""
        from src.libs.splitter.recursive_splitter import RecursiveSplitter
        fallback = RecursiveSplitter(self.settings)
        return fallback.split_text(text)