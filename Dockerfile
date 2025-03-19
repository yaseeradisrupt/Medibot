FROM python:3.10-slim

WORKDIR /app 

# Copy requirements and install dependencies
COPY requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# Copy the entire project
COPY . .

# Check if the vector store folder exists, and if not, run createMemoryForLLM.py
# RUN python3 app/utils/createMemoryForLLM.py
    
# Expose ports for FastAPI and Streamlit
EXPOSE 8005 8501

# Command to run both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload & streamlit run app/medibot.py --server.port 8501"]