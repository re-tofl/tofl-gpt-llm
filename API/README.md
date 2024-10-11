# LLaMA API

Представляет API для взаимодействия с моделью LLaMA

## Установка

### Требования:
- Python 3.8 или выше

### Шаги установки:

1. Установите необходимые библиотеки:
    ```bash
    pip install -r requirements.txt
    ```

2. Запустите сервер FastAPI:
    ```bash
    uvicorn llama_api:app --host 0.0.0.0 --port 8000
    ```

### Использование

1. В файле `client.py` есть простой пример использования API. Для отправки вопросов модели используйте:
   ```bash
   python client.py
   ```

2. Пример curl:
   ```bash
    curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Что такое конечный автомат?"}'
    ```