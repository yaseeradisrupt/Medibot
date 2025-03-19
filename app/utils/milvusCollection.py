from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from app.config.config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION_NAME, DIMENSION

def get_milvus_collection():
    print(MILVUS_HOST ,MILVUS_PORT, MILVUS_COLLECTION_NAME)

    # Connect to Milvus server
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    # Check if the collection exists
    if utility.has_collection(MILVUS_COLLECTION_NAME):
        collection = Collection(MILVUS_COLLECTION_NAME)
    else:
        # Define the schema for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields, "Medical Knowledge Collection")
        
        # Create the collection
        collection = Collection(name=MILVUS_COLLECTION_NAME, schema=schema)
        
        # Create an index for the embedding field
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        # Load the collection into memory
        collection.load()
    
    return collection