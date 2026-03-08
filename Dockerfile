FROM python:3.10-slim

# HF Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install dependencies first (layer cache)
COPY --chown=user requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=user src/   ./src/
COPY --chown=user config/ ./config/
COPY --chown=user web/    ./web/

# Add src/ to Python path (avoids needing setuptools editable install)
ENV PYTHONPATH=/home/user/app/src

# Hugging Face Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "soci.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
