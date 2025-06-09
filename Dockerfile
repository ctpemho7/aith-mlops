FROM python:3.11-slim

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.8.4  python3 - && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml poetry.lock  ./
RUN poetry config installer.max-workers 4 && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi

ENV TZ="Europe/Moscow"

WORKDIR /app
COPY . .

CMD ["bash"]
