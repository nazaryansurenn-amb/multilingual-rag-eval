FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV RAG_DIR=/data \
    LM_STUDIO_URL=http://host.docker.internal:1234/v1 \
    PYTHONUNBUFFERED=1

CMD ["python", "-c", "print('water-docs RAG image. Run a script from src/')"]