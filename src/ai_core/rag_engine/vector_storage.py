"""
Vector Storage for NEXUS RAG Engine
Handles embedding and storage of knowledge in vector format for semantic retrieval
"""
import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStorage:
    """
    Vector storage for NEXUS that handles the embedding and storage of knowledge
    in a vector database for semantic retrieval. 
    
    This implementation supports multiple backends:
    1. ChromaDB (default) - Efficient local vector DB
    2. FAISS - Fast vector similarity search
    3. Simple in-memory (fallback) - Basic numpy-based vector storage
    """
    
    def __init__(self, storage_dir: str = "memory/vector_db", 
                backend: str = "auto", collection_name: str = "nexus_knowledge"):
        """
        Initialize the vector storage system
        
        Args:
            storage_dir: Directory to store vector database files
            backend: Vector DB backend to use ('chroma', 'faiss', 'memory', or 'auto')
            collection_name: Name of the collection/index to use
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.collection_name = collection_name
        self.backend_type = backend
        self.embedding_dim = 384  # Default for sentence-transformers
        
        # Will store the actual backend instance
        self.backend = None
        self.embedding_model = None
        
        # Initialize vector database backend
        self._initialize_backend(backend)
        
        logger.info(f"NEXUS Vector Storage initialized with backend: {self.backend_type}")
    
    def _initialize_backend(self, backend: str) -> None:
        """Initialize the vector database backend based on availability"""
        if backend == "auto":
            # Try each backend in order of preference
            backends_to_try = ["chroma", "faiss", "memory"]
            for b in backends_to_try:
                try:
                    success = self._setup_specific_backend(b)
                    if success:
                        self.backend_type = b
                        break
                except Exception as e:
                    logger.warning(f"Failed to initialize {b} backend: {e}")
        else:
            # Try to set up the specific requested backend
            success = self._setup_specific_backend(backend)
            if not success:
                logger.warning(f"Falling back to in-memory vector storage")
                self._setup_specific_backend("memory")
    
    def _setup_specific_backend(self, backend_type: str) -> bool:
        """Set up a specific vector database backend"""
        if backend_type == "chroma":
            try:
                import chromadb
                from chromadb.utils import embedding_functions
                
                # Set up embedding function
                self._initialize_embedding_model()
                
                # Initialize ChromaDB client
                self.chroma_client = chromadb.PersistentClient(path=str(self.storage_dir))
                
                # Get or create collection
                self.backend = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self._get_chroma_embedding_function()
                )
                
                logger.info(f"ChromaDB backend initialized with collection: {self.collection_name}")
                return True
                
            except ImportError:
                logger.warning("ChromaDB not available. Please install with: pip install chromadb")
                return False
                
        elif backend_type == "faiss":
            try:
                import faiss
                
                # Initialize embedding model
                self._initialize_embedding_model()
                
                # Check if an existing index exists
                index_path = self.storage_dir / f"{self.collection_name}_faiss.index"
                metadata_path = self.storage_dir / f"{self.collection_name}_metadata.json"
                
                if index_path.exists():
                    # Load existing index
                    self.backend = faiss.read_index(str(index_path))
                    
                    # Load metadata
                    with open(metadata_path, 'r') as f:
                        self.id_to_metadata = json.load(f)
                else:
                    # Create new index
                    self.backend = faiss.IndexFlatL2(self.embedding_dim)
                    self.id_to_metadata = {}
                
                self.faiss_index_path = index_path
                self.faiss_metadata_path = metadata_path
                
                logger.info("FAISS backend initialized")
                return True
                
            except ImportError:
                logger.warning("FAISS not available. Please install with: pip install faiss-cpu")
                return False
                
        elif backend_type == "memory":
            # Simple in-memory vector storage using numpy
            self._initialize_embedding_model()
            
            # Create empty arrays for vectors and metadata
            self.vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self.metadata = []
            self.ids = []
            
            # Load existing data if available
            memory_path = self.storage_dir / f"{self.collection_name}_memory.npz"
            if memory_path.exists():
                try:
                    data = np.load(memory_path, allow_pickle=True)
                    self.vectors = data['vectors']
                    self.metadata = data['metadata'].tolist()
                    self.ids = data['ids'].tolist()
                    logger.info(f"Loaded in-memory storage with {len(self.ids)} items")
                except Exception as e:
                    logger.error(f"Error loading in-memory storage: {e}")
            
            self.backend = {
                'vectors': self.vectors,
                'metadata': self.metadata,
                'ids': self.ids
            }
            
            self.memory_path = memory_path
            
            logger.info("In-memory vector storage initialized")
            return True
            
        else:
            logger.error(f"Unknown backend type: {backend_type}")
            return False
    
    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model for vector conversion"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use a lightweight, fast model for embeddings
            # all-MiniLM-L6-v2 is a good balance of quality and speed
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            logger.info(f"Initialized embedding model with dimension: {self.embedding_dim}")
            
        except ImportError:
            logger.warning("Sentence-transformers not available. Please install with: pip install sentence-transformers")
            
            # Create a simple fallback embedding function using character counts
            # This is ONLY for demonstration and should not be used in production
            def simple_embedding(text):
                # Very simplified embedding - just for fallback
                import hashlib
                hash_val = hashlib.md5(text.encode()).digest()
                np_array = np.frombuffer(hash_val, dtype=np.uint8)
                expanded = np.zeros(self.embedding_dim, dtype=np.float32)
                expanded[:min(len(np_array), self.embedding_dim)] = np_array[:min(len(np_array), self.embedding_dim)]
                # Normalize
                norm = np.linalg.norm(expanded)
                if norm > 0:
                    expanded = expanded / norm
                return expanded
            
            self.embedding_model = lambda texts: np.stack([simple_embedding(text) for text in texts])
            logger.warning("Using simplified fallback embedding function")
    
    def _get_chroma_embedding_function(self):
        """Get ChromaDB-compatible embedding function"""
        if self.backend_type == "chroma":
            try:
                from chromadb.utils import embedding_functions
                
                # If we have the sentence_transformer model, use it
                if hasattr(self.embedding_model, 'encode'):
                    return embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name='all-MiniLM-L6-v2'
                    )
                else:
                    # Otherwise use our custom function
                    class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
                        def __call__(self, texts):
                            return self.embedding_model(texts)
                    
                    return CustomEmbeddingFunction()
                    
            except ImportError:
                logger.error("Could not create ChromaDB embedding function")
                return None
        return None
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None, 
                ids: List[str] = None) -> List[str]:
        """
        Add texts to the vector storage
        
        Args:
            texts: List of text to add
            metadatas: Optional list of metadata dicts for each text
            ids: Optional list of IDs for each text
            
        Returns:
            List of IDs for the added texts
        """
        if not texts:
            return []
            
        # Create default metadata and IDs if not provided
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Make sure lists have the same length
        assert len(texts) == len(metadatas) == len(ids), "Texts, metadatas, and ids must have the same length"
        
        # Add to the appropriate backend
        if self.backend_type == "chroma":
            self.backend.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
        elif self.backend_type == "faiss":
            # Convert texts to embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Add embeddings to FAISS index
            faiss.normalize_L2(embeddings)
            self.backend.add(embeddings)
            
            # Store metadata
            for i, id_val in enumerate(ids):
                self.id_to_metadata[id_val] = {
                    "text": texts[i],
                    "metadata": metadatas[i]
                }
                
            # Save index and metadata
            faiss.write_index(self.backend, str(self.faiss_index_path))
            with open(self.faiss_metadata_path, 'w') as f:
                json.dump(self.id_to_metadata, f)
                
        elif self.backend_type == "memory":
            # Convert texts to embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Add to in-memory storage
            if len(self.vectors) == 0:
                self.vectors = embeddings
            else:
                self.vectors = np.vstack([self.vectors, embeddings])
                
            self.metadata.extend(metadatas)
            self.ids.extend(ids)
            
            # Update backend reference
            self.backend = {
                'vectors': self.vectors,
                'metadata': self.metadata,
                'ids': self.ids
            }
            
            # Save to disk
            np.savez(
                self.memory_path, 
                vectors=self.vectors, 
                metadata=np.array(self.metadata, dtype=object),
                ids=np.array(self.ids, dtype=object)
            )
        
        return ids
    
    def similarity_search(self, query: str, k: int = 4, 
                        filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar texts by semantic similarity
        
        Args:
            query: The query text to search for
            k: Number of results to return
            filter_dict: Optional filter to apply to the search
            
        Returns:
            List of dictionaries containing similar texts and their metadata
        """
        if not query:
            return []
            
        # Different search implementation per backend
        if self.backend_type == "chroma":
            # ChromaDB search
            filter_condition = filter_dict if filter_dict else None
            
            results = self.backend.query(
                query_texts=[query],
                n_results=k,
                where=filter_condition
            )
            
            # Format results
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            ids = results.get('ids', [[]])[0]
            distances = results.get('distances', [[]])[0]
            
            return [
                {
                    "text": doc,
                    "metadata": meta,
                    "id": id_val,
                    "score": 1.0 - (dist / 2.0)  # Convert distance to similarity score
                }
                for doc, meta, id_val, dist in zip(documents, metadatas, ids, distances)
            ]
            
        elif self.backend_type == "faiss":
            # FAISS search
            query_embedding = self.embedding_model.encode([query])[0]
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.backend.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.id_to_metadata):
                    continue
                    
                id_val = list(self.id_to_metadata.keys())[idx]
                item = self.id_to_metadata[id_val]
                
                # Apply filter if provided
                if filter_dict and not self._matches_filter(item["metadata"], filter_dict):
                    continue
                    
                results.append({
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "id": id_val,
                    "score": 1.0 - (distances[0][i] / 2.0)  # Convert distance to similarity score
                })
            
            return results
            
        elif self.backend_type == "memory":
            # In-memory search
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Compute distances to all vectors
            if len(self.vectors) > 0:
                distances = np.linalg.norm(self.vectors - query_embedding, axis=1)
                
                # Get indices of k nearest neighbors
                indices = np.argsort(distances)[:k]
                
                results = []
                for idx in indices:
                    item_metadata = self.metadata[idx]
                    
                    # Apply filter if provided
                    if filter_dict and not self._matches_filter(item_metadata, filter_dict):
                        continue
                        
                    results.append({
                        "text": item_metadata.get("text", ""),
                        "metadata": item_metadata,
                        "id": self.ids[idx],
                        "score": 1.0 - (distances[idx] / 2.0)  # Convert distance to similarity score
                    })
                
                return results
        
        return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter"""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete texts by ID
        
        Args:
            ids: List of IDs to delete
        """
        if not ids:
            return
            
        if self.backend_type == "chroma":
            self.backend.delete(ids=ids)
            
        elif self.backend_type == "faiss":
            # FAISS doesn't support deletion, so we need to rebuild the index
            # This is inefficient but works for now
            remaining_ids = [id_val for id_val in self.id_to_metadata if id_val not in ids]
            remaining_texts = [self.id_to_metadata[id_val]["text"] for id_val in remaining_ids]
            remaining_metadatas = [self.id_to_metadata[id_val]["metadata"] for id_val in remaining_ids]
            
            # Clear backend
            self.backend = faiss.IndexFlatL2(self.embedding_dim)
            self.id_to_metadata = {}
            
            # Re-add remaining items
            if remaining_texts:
                self.add_texts(remaining_texts, remaining_metadatas, remaining_ids)
                
        elif self.backend_type == "memory":
            # Get indices to keep
            keep_indices = [i for i, id_val in enumerate(self.ids) if id_val not in ids]
            
            if keep_indices:
                self.vectors = self.vectors[keep_indices]
                self.metadata = [self.metadata[i] for i in keep_indices]
                self.ids = [self.ids[i] for i in keep_indices]
            else:
                self.vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)
                self.metadata = []
                self.ids = []
                
            # Update backend reference
            self.backend = {
                'vectors': self.vectors,
                'metadata': self.metadata,
                'ids': self.ids
            }
            
            # Save to disk
            np.savez(
                self.memory_path, 
                vectors=self.vectors, 
                metadata=np.array(self.metadata, dtype=object),
                ids=np.array(self.ids, dtype=object)
            )
    
    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get texts by ID
        
        Args:
            ids: List of IDs to retrieve
            
        Returns:
            List of dictionaries containing texts and their metadata
        """
        if not ids:
            return []
            
        if self.backend_type == "chroma":
            results = self.backend.get(ids=ids)
            
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])
            result_ids = results.get('ids', [])
            
            return [
                {
                    "text": doc,
                    "metadata": meta,
                    "id": id_val
                }
                for doc, meta, id_val in zip(documents, metadatas, result_ids)
            ]
            
        elif self.backend_type == "faiss":
            return [
                {
                    "text": self.id_to_metadata[id_val]["text"],
                    "metadata": self.id_to_metadata[id_val]["metadata"],
                    "id": id_val
                }
                for id_val in ids if id_val in self.id_to_metadata
            ]
            
        elif self.backend_type == "memory":
            return [
                {
                    "text": self.metadata[i].get("text", ""),
                    "metadata": self.metadata[i],
                    "id": id_val
                }
                for i, id_val in enumerate(self.ids) if id_val in ids
            ]
            
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector storage"""
        if self.backend_type == "chroma":
            collection_info = self.backend.count()
            return {
                "backend": "ChromaDB",
                "count": collection_info,
                "collection": self.collection_name
            }
            
        elif self.backend_type == "faiss":
            return {
                "backend": "FAISS",
                "count": self.backend.ntotal,
                "dimension": self.embedding_dim
            }
            
        elif self.backend_type == "memory":
            return {
                "backend": "In-memory",
                "count": len(self.ids),
                "dimension": self.embedding_dim
            }
            
        return {"backend": "Unknown"}
