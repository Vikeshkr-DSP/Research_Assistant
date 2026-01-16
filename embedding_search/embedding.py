from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

class NomicEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(
            model_name_or_path="D:/Study Material/models/nomic-ai/nomic-embed-text-v2-moe",
            trust_remote_code=True
        )

    def embed_documents(self, texts):
        return self.model.encode(
            sentences=texts,
            prompt_name="passage",
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [text],
            prompt_name="query",
            normalize_embeddings=True
        )[0].tolist()


## functions for embedding and retrieve most similar chunk based on user query
def chunk_retriever(chunks:list, query:str)->str:
    embedding_model = NomicEmbeddings()

    ## plain string to Langchain Document
    docs  = [Document(page_content=str(item)) for item in chunks]

    ## vector db and retriever using FAISS
    vector_store = FAISS.from_documents(documents=docs, embedding=embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":1, "search_with_scores": False})
    matched_chunk = retriever.invoke(input=query)

    return [item.page_content for item in matched_chunk][0].strip('{}').replace("'", "")

