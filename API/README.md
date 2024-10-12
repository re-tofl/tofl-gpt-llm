# LLaMA API

Представляет API для взаимодействия с моделью LLaMA

## Требования:
- Python 3.8 или выше
- Docker (для одного из типов установки)

## Установка:
### Без Docker:
1. Установите необходимые библиотеки:
    ```bash
    pip install -r requirements.txt
    ```
2. Запустите сервер FastAPI:
    ```bash
    uvicorn llama_api:app --host 0.0.0.0 --port 8000
    ```
### С Docker:
1. Соберите build (если не работает, то с sudo):
   ```bash
   docker-compose build
   ```
2. Запустите контейнер (если не работает, то с sudo):
   ```bash
   docker-compose up
   ```

## Использование
1. В файле `client.py` есть простой пример использования API:
   ```bash
   python client.py
   ```
2. Пример curl:
   ```bash
    curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Что такое конечный автомат?"}'
    ```