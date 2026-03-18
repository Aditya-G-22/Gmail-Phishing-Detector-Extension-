FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Then **create the `.dockerignore`** file:
```
New-Item -Path .dockerignore -ItemType File
```

Paste this inside:
```
__pycache__
*.pyc
.git
.github
Phishing_extension
docs
README.md
.gitignore
```

**One important thing** — make sure `main:app` in the CMD line matches your actual FastAPI file. If your main Python file is called something different, like `app.py` or `server.py`, adjust it accordingly. For example if it's `app.py`, change it to `"app:app"`.

**Test it locally** (make sure Docker Desktop is installed and running):
```
docker build -t phishing-detector .
docker run -p 8000:8000 phishing-detector
```

Then visit `http://localhost:8000/docs` in your browser — if you see the FastAPI Swagger page, it's working.

After that, push it to GitHub:
```
git add Dockerfile .dockerignore
git commit -m "Add Dockerfile and dockerignore"
git push origin main