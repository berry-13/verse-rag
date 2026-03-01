FROM python:3.12-slim

WORKDIR /app

# System deps for crawl4ai (playwright, etc.)
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    # Playwright deps
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 libx11-6 libx11-xcb1 libxcb1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Clone the original repo and install deps before copying patched files
# so that code changes don't bust the install cache
RUN git clone https://github.com/coleam00/mcp-crawl4ai-rag.git /app/original

WORKDIR /app/original

# Install dependencies (cached independently of src/ changes)
RUN UV_HTTP_TIMEOUT=300 uv pip install --system -e .

# Install playwright browsers
RUN python -m playwright install chromium --with-deps

# Install ollama client (not in original deps)
RUN pip install ollama httpx

# Copy patched files last — only invalidates this layer on code changes
COPY src/ /app/original/src/

CMD ["python", "src/crawl4ai_mcp.py"]
