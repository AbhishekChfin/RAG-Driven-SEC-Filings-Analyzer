import gc
from typing import List, Any
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = 'all-MPNet-base-v2', device: str = "mps"):
        # Note: 'mps' is for Mac Silicon. Change to 'cuda' for Nvidia or 'cpu' if neither.
        print(f"Loading embedding model: {model_name} on {device}...")
        try:
            self.model = SentenceTransformer(f"sentence-transformers/{model_name}", device=device)
        except Exception:
            print("⚠️ Device not found, falling back to CPU")
            self.model = SentenceTransformer(f"sentence-transformers/{model_name}", device="cpu")

    def generate_embeddings(self, texts: List[str], batch_size: int = 4) -> List[List[float]]:
        """
        Generates vector embeddings for a list of text strings.
        """
        all_embeddings = []
        chunk_size = 50 # Processing chunk size (not text chunk)
        
        print(f"Generating embeddings for {len(texts)} items...")
        
        for i in tqdm(range(0, len(texts), chunk_size), desc="Embedding"):
            batch_texts = texts[i:i + chunk_size]
            try:
                batch_embs = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=batch_size
                )
                all_embeddings.extend(batch_embs)
            except Exception as e:
                print(f"❌ Embedding error at index {i}: {e}")
                # Return partial or empty on fatal error depending on strictness needed
                return []
            
            # Memory cleanup for large loops
            gc.collect()
            
        return all_embeddings