import faiss
import numpy as np
import os

class FaissIndexer:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.index = None
        self.is_gpu_index = False

    def create_index(self, embeddings):
        """Create a FAISS index from embeddings."""
        if embeddings.size == 0:
            print("Empty embeddings array. Cannot initialize FAISS index.")
            return

        dim = embeddings.shape[1]
        try:
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    config = faiss.GpuIndexFlatConfig()
                    config.device = 0
                    gpu_index = faiss.GpuIndexFlatL2(res, dim, config)
                    batch_size = 100
                    for i in range(0, embeddings.shape[0], batch_size):
                        batch = embeddings[i:i + batch_size]
                        gpu_index.add(batch)
                    self.index = gpu_index
                    self.is_gpu_index = True
                    print("FAISS GPU index initialized and embeddings added.")
                except Exception as e:
                    print(f"GPU indexing failed: {e}")
                    print("Falling back to CPU indexing.")
                    self._create_cpu_index(embeddings, dim)
            else:
                self._create_cpu_index(embeddings, dim)
        except Exception as e:
            print(f"Error initializing FAISS index: {e}")

    def _create_cpu_index(self, embeddings, dim):
        """Create a CPU-based FAISS index."""
        cpu_index = faiss.IndexFlatL2(dim)
        cpu_index.add(embeddings)
        self.index = cpu_index
        print("FAISS CPU index initialized and embeddings added.")

    def save_index(self, file_path="faiss_index.bin"):
        """Save the FAISS index to a file."""
        if self.index:
            faiss.write_index(self.index if not self.is_gpu_index else faiss.index_gpu_to_cpu(self.index), file_path)
            print(f"FAISS index saved to {file_path}")
        else:
            print("No index to save.")

    def load_index(self, file_path="faiss_index.bin"):
        """Load the FAISS index from a file."""
        if os.path.exists(file_path):
            self.index = faiss.read_index(file_path)
            self.is_gpu_index = False  # Assume CPU index unless GPU is explicitly set later
            print(f"FAISS index loaded from {file_path}")
            return True
        print(f"No FAISS index found at {file_path}")
        return False