from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from config import DOCUMENT_FILEPATH, COLLECTION_NAME, TOP_K_RETRIEVAL

def initialize_and_load_vector_db(file_path: str = None, collection_name: str = COLLECTION_NAME):
    """
    Initializes a ChromaDB client and an embedding model,
    then semantically chunks a give PDF document, and saves
    the chunks and their embeddings to ChromaDB.
    """

    # Initialize Langchain embedding model
    print("Loading embedding model...")
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name=collection_name)
    if collection:
        return collection, embeddings_model
    collection = chroma_client.create_collection(name=collection_name)

    # Load document for retrieval
    print(f"Loading document from {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Chunk document via semantic chunking
    print("Chunking text semantically...")
    text_splitter = SemanticChunker(embeddings_model)
    docs = text_splitter.split_documents(documents)
    
    doc_contents = [doc.page_content for doc in docs]
    doc_ids = [f"id_{i}" for i in range(len(docs))]

    # Add the chunks to the collection
    print(f"Adding {len(docs)} chunks to the collection...")
    collection.add(
        documents=doc_contents,
        ids=doc_ids
    )
    print("Vector DB initialization and loading completed.")
    return collection, embeddings_model

if __name__ == "__main__":
    # Run script to create vectorDB for PDF document
    print("Running vectordb.py script...")
    collection, embeddings_model = initialize_and_load_vector_db(DOCUMENT_FILEPATH, COLLECTION_NAME)
    
    # Example of how to query the collection
    query = "How many days of hospitalization leave?"
    query_embedding = embeddings_model.embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_RETRIEVAL
    )
    
    print(f"\n--- Example Query ---")
    print(f"Query: '{query}'")
    print("Found similar documents in ChromaDB:")
    if results['documents']:
        for doc in results['documents'][0]:
            print(f"- {doc[:150]}...") # Print first 150 characters