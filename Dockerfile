FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="${PYTHONPATH}:/app/Models"

EXPOSE 8000

CMD ["sh", "-c", "cd /app/Models && uvicorn api:app --host 0.0.0.0 --port 8000"]
