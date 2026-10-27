FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[model]"
COPY models ./models
ENV WINDQC_MODEL_DIR=/app/models

EXPOSE 8000
CMD ["uvicorn", "windqc.api:app", "--host", "0.0.0.0", "--port", "8000"]
