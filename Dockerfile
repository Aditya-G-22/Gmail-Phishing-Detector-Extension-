FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Open your Dockerfile, delete everything, paste only the above (no ` ``` `, no extra text), save it, and push:
```
git add Dockerfile
git commit -m "Fix Dockerfile"
git push origin main