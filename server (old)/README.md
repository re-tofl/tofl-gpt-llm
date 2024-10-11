# LLaMA API with FastAPI, Celery, and Redis

Представляет API для взаимодействия с моделью LLaMA, который использует FastAPI для обработки запросов, Celery для обработки фоновых задач, и Redis как брокер и хранилище результатов.

## Установка

### Требования:
- Python 3.8 или выше
- Redis

### Шаги установки:

1. Установите необходимые библиотеки:
    ```bash
    pip install -r requirements.txt
    ```

2. Установите Redis:

   - **На Ubuntu:**
     ```bash
     sudo apt update
     sudo apt install redis-server
     ```
     Для проверки, что Redis работает:
     ```bash
     sudo systemctl status redis
     ```
     Если Redis не запущен, используйте:
     ```bash
     sudo systemctl start redis
     ```

   - **На MacOS (через Homebrew):**
     ```bash
     brew install redis
     brew services start redis
     ```

   - **На Windows:**
     1. Скачайте Redis с [официального сайта](https://github.com/MicrosoftArchive/redis/releases).
     2. Установите и запустите Redis, используя `redis-server.exe`.

3. Запустите Redis сервер:
    ```bash
    redis-server
    ```

4. Запустите Celery:
    ```bash
    celery -A celery_worker.app worker --loglevel=info
    ```

5. Запустите сервер FastAPI:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

6. Тестирование клиента:

   В файле `client.py` есть простой пример использования API. Для отправки вопросов модели используйте:
   ```bash
   python client.py
   ```