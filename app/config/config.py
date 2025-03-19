from dotenv import dotenv_values
config = dotenv_values(".env")

OPENAI_API_KEY = config.get("OPENAI_API_KEY", "OPENAI_API_KEY")
OPEN_AI_MODEL = "gpt-4o"
MILVUS_HOST = config.get("MILVUS_HOST", "localhost")
MILVUS_PORT = config.get("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = config.get("MILVUS_COLLECTION_NAME", "medical_knowledge")
DIMENSION=1536
EMBEDDING_MODEL = "text-embedding-ada-002"
DATA_PATH = "data/unprocessed/"
PROCESSED_PATH="data/processed/"
