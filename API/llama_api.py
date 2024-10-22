from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch
from deep_translator import GoogleTranslator


def translate_text(text, source_language, target_language):
    try:
        return GoogleTranslator(source=source_language, target=target_language).translate(text)
    except Exception as e:
        print(f"Error during translation: {e}")
        return None


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
    user_input = translate_text(query.prompt, source_language='ru', target_language='en')
    alpaca_prompt = """Below is a task that describes the problem. Write an answer that completes the query appropriately.

    ### Instructions:
    {}

    ### Answer:
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
        response = translate_text(generated_text[0], source_language='en', target_language='ru')
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск: команда uvicorn llama_api:app --host 0.0.0.0 --port <port> --reload
