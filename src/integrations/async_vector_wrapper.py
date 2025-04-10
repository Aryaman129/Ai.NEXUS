"""
Async wrapper for Vector Storage
Provides async-friendly methods to interact with the vector database
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from functools import wraps

# Import the base vector storage
from src.ai_core.rag_engine.vector_storage import VectorStorage

logger = logging.getLogger(__name__)

class AsyncVectorWrapper:
    """
    Async wrapper for VectorStorage that properly handles async operations
    
    This wrapper adapts the synchronous VectorStorage methods to be properly awaitable,
    solving the 'object list can't be used in await expression' errors.
    """
    
    def __init__(self, vector_storage: VectorStorage):
        """
        Initialize with an existing VectorStorage instance
        
        Args:
            vector_storage: The VectorStorage instance to wrap
        """
        self.vector_storage = vector_storage
    
    async def add_texts(self, texts: List[str], metadatas: List[Dict] = None, 
                        ids: List[str] = None) -> List[str]:
        """
        Add texts to the vector storage asynchronously
        
        Args:
            texts: List of text to add
            metadatas: Optional list of metadata dicts for each text
            ids: Optional list of IDs for each text
            
        Returns:
            List of IDs for the added texts
        """
        # Make sure we have lists, not coroutines or other objects
        texts_list = list(texts) if texts else []
        metadatas_list = list(metadatas) if metadatas else None
        ids_list = list(ids) if ids else None
        
        # Run the synchronous method in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None,  # Default executor
                lambda: self.vector_storage.add_texts(
                    texts=texts_list,
                    metadatas=metadatas_list,
                    ids=ids_list
                )
            )
        except Exception as e:
            logger.error(f"Error in async add_texts: {e}")
            return []
    
    async def similarity_search(self, query: str, k: int = 4, 
                                filter_dict: Dict[str, Any] = None) -> List[Dict]:
        """
        Search for similar texts by semantic similarity asynchronously
        
        Args:
            query: The query text to search for
            k: Number of results to return
            filter_dict: Optional filter to apply to the search
            
        Returns:
            List of dictionaries containing similar texts and their metadata
        """
        # Run the synchronous method in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,  # Default executor
                lambda: self.vector_storage.similarity_search(
                    query=query,
                    k=k,
                    filter_dict=filter_dict
                )
            )
            
            # Make sure we're returning a list
            if not isinstance(results, list):
                return []
            
            return results
        except Exception as e:
            logger.error(f"Error in async similarity_search: {e}")
            return []
    
    async def delete(self, ids: List[str]):
        """
        Delete texts by ID asynchronously
        
        Args:
            ids: List of IDs to delete
        """
        # Make sure we have a list, not a coroutine
        ids_list = list(ids) if ids else []
        
        # Run the synchronous method in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,  # Default executor
                lambda: self.vector_storage.delete(ids=ids_list)
            )
            return True
        except Exception as e:
            logger.error(f"Error in async delete: {e}")
            return False
    
    async def get(self, ids: List[str]) -> List[Dict]:
        """
        Get texts by ID asynchronously
        
        Args:
            ids: List of IDs to retrieve
            
        Returns:
            List of dictionaries containing texts and their metadata
        """
        # Make sure we have a list, not a coroutine
        ids_list = list(ids) if ids else []
        
        # Run the synchronous method in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,  # Default executor
                lambda: self.vector_storage.get(ids=ids_list)
            )
            
            # Make sure we're returning a list
            if not isinstance(results, list):
                return []
            
            return results
        except Exception as e:
            logger.error(f"Error in async get: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector storage
        This is a pass-through method since it's synchronous and fast
        """
        return self.vector_storage.get_stats()
