FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3-dev libxml2-dev libz-dev \
        libopenblas-dev gfortran libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Poetry のインストールと依存解決
RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock* /app/
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --without dev --no-interaction --no-ansi

COPY . /app
RUN mkdir -p /app/data

CMD ["python", "-u", "-m", "src"]
