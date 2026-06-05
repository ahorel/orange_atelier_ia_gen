FROM python:3.11-slim

WORKDIR /app

COPY RAG_POC/requirements_rag.txt .
RUN pip install --no-cache-dir -r requirements_rag.txt

COPY . .

WORKDIR /app/RAG_POC

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
