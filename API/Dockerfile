# Базовый образ с поддержкой CUDA
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка нужной версии Python
RUN apt-get update && apt-get install -y python3.10

# Установка зависимостей
COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt

# Копирование кода приложения
COPY . /app
WORKDIR /app

# Открытие порта
EXPOSE 8000

# Запуск сервера
CMD ["bash", "-c", "uvicorn llama_api:app --host 0.0.0.0 --port 8000 --reload"]
