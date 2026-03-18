name: CI/CD Pipeline

on:
  push:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t phishing-detector .

      - name: Run Docker container (smoke test)
        run: |
          docker run -d -p 8000:8000 --name test-container phishing-detector
          sleep 60
          curl -f http://localhost:8000/docs || exit 1
          docker stop test-container