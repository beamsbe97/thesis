FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip

RUN pip install --no-cache-dir -U git+https://github.com/huggingface/transformers

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torchcodec

COPY . .
CMD ["python", "test.py", "-i", "~/EPIC-KITCHENS", "-o", "embeddings"]
