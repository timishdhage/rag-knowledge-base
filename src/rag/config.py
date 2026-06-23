from pydantic import BaseModel

class Settings(BaseModel):
    chroma_path: str = './chroma'
    embedding_model: str = 'text-embedding-3-small'
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5

settings = Settings()
