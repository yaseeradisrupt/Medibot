import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from app.config.config import DATA_PATH, PROCESSED_PATH, OPENAI_API_KEY, EMBEDDING_MODEL
from app.utils.milvusCollection import get_milvus_collection

# Ensure processed folder exists
os.makedirs(PROCESSED_PATH, exist_ok=True)
llm = OpenAI(api_key=OPENAI_API_KEY)
try:
    collection= get_milvus_collection()
    # Step 1: Load Raw PDFs
    def load_pdf_files(data):
        loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        return documents

    documents = load_pdf_files(DATA_PATH)

    # Step 2: Create Chunks
    def create_chunks(extracted_data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(extracted_data)
        return text_chunks
    
    print("creating text_chunks")

    text_chunks = create_chunks(documents)
    print("Length of text_chunks:", len(text_chunks))

    # Step 3: Create Vector Embeddings
    def generate_embeddings_and_insert(chunks, batch_size=100):
        embeddings_data = []
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

        for batch in batches:
            texts = [chunk.page_content for chunk in batch]
            try:
                print(f"Generating embeddings for batch: {len(texts)}")
                response = llm.embeddings.create(input=texts, model=EMBEDDING_MODEL)
                embeddings = [data.embedding for data in response.data]
                for i, chunk in enumerate(batch):
                    embeddings_data.append({"embedding": embeddings[i], "content": chunk.page_content})
            except Exception as e:
                print(f"Error generating embeddings for batch: {e}")
                # Handle the error, possibly retry or log the failed batch.

        embeddings = [item["embedding"] for item in embeddings_data]
        contents = [item["content"] for item in embeddings_data]
        collection.insert([embeddings, contents])
        print("Embeddings generated and inserted into Milvus.")

    generate_embeddings_and_insert(text_chunks)



    # Step 5: Move Files to "Processed" Folder (Only on Success)
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            shutil.move(os.path.join(DATA_PATH, file), os.path.join(PROCESSED_PATH, file))

    print("Processing completed. Files moved to 'processed' folder.")

except Exception as e:
    print(f"Error occurred: {e}")
    print("Files remain in 'unprocessed' folder.")
