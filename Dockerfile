FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/   ./src/
COPY config/ ./config/
COPY web/    ./web/

# Install the soci package (handles the src/ layout)
RUN pip install --no-cache-dir -e .

# Hugging Face Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "soci.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
