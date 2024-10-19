from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch

app = FastAPI()

# Модель и токенизатор
model_name = "c9llm3bones/Llama-3.1-8B-Instruct-bnb-4bit-TFL"
max_seq_length = 2048
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=load_in_4bit
)

FastLanguageModel.for_inference(model)


# Модель данных для входящего запроса
class QueryModel(BaseModel):
    prompt: str


# Маршрут для получения ответа от модели
@app.post("/generate")
async def generate_text(query: QueryModel):
    user_input = query.prompt
    alpaca_prompt = """Ниже представлено задание, которое описывает задачу. Напишите ответ, который соответствующим образом завершает запрос.

    ### Инструкция:
    {}

    ### Ответ:
    {}"""

    try:
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    user_input,
                    ""
                )
            ], return_tensors="pt").to("cuda")
        response = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        generated_text = tokenizer.batch_decode(response, skip_special_tokens=True)
        return {"response": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск: команда uvicorn llama_api:app --host 0.0.0.0 --port <port> --reload
