# base image
FROM python:3.10-slim

# ish papkasi
WORKDIR /app

# requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# projectni COPY qilamiz
COPY . .

# ports
EXPOSE 8000
EXPOSE 7860

# CMD: FastAPI + Gradio
CMD ["sh", "-c", "PYTHONPATH=/app uvicorn app.main:app --host 0.0.0.0 --port 8000 & python gradio_app.py"]
