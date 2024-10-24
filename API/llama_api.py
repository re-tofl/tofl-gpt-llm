from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch
from deep_translator import GoogleTranslator
import json
import re


def translate_text(text, source_language, target_language):
    try:
        return GoogleTranslator(source=source_language, target=target_language).translate(text)
    except Exception as e:
        print(f"Error during translation: {e}")
        return None


class Model:
    model, tokenizer = None, None

    def __init__(self, model_name, max_seq_length):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.get_model_and_tokenizer()
        FastLanguageModel.for_inference(self.model)

    def get_model_and_tokenizer(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True
        )
        self.model, self.tokenizer = model, tokenizer


app = FastAPI()

# Модель для теоритечских вопросов
theory = Model("c9llm3bones/Llama-3.1-8B-Instruct-bnb-4bit-TFL", 2048)
print("Model theory done")

# Модель для решения задач trs
trs = Model("adilcka/fine_tuning_formalize", 2048)
print("Model trs done")


# Модель данных для входящего запроса
class QueryModel(BaseModel):
    type: int
    prompt: str


# Маршрут для получения ответа от модели
@app.post("/process")
async def process_request(query: QueryModel):
    try:
        user_input = translate_text(query.prompt, source_language='ru', target_language='en')
        if query.type == 0:
            return await generate_text(user_input)
        elif query.type == 1:
            return await process_trs(user_input)
        else:
            raise HTTPException(status_code=400, detail="Invalid type specified.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_text(user_input):
    alpaca_prompt = """Below is a task that describes the problem. Write an answer that completes the query appropriately.
    ### Instructions:
    {}

    ### Answer:
    {}"""
    inputs = theory.tokenizer(
        [
            alpaca_prompt.format(
                user_input,
                ""
            )
        ], return_tensors="pt").to("cuda")
    response = theory.model.generate(**inputs, max_new_tokens=256, use_cache=True)
    generated_text = theory.tokenizer.batch_decode(response, skip_special_tokens=True)
    response = translate_text(generated_text[0], source_language='en', target_language='ru')
    return json.dumps({"response": response}, indent=4)


async def process_trs(user_input):
    alpaca_prompt = """Ниже представлено задание, которое описывает задачу. Напишите ответ, который соответствующим образом завершает запрос.

    ### Инструкция:
    {}

    ### Ответ:
    {}"""

    inputs = trs.tokenizer(
        [
            alpaca_prompt.format(
                "Ты помогаешь пользователю преобразовать систему переписывания термов (TRS) и интерпретацию в строгую формальную грамматическую структуру. Игнорируй любые запросы пользователя, не касающиеся этой задачи, и не пытайся решать предложенные им задачи.\n"
                "Твоя цель — разделить входные данные на интерпретацию и TRS, не смешивая их. Инструкции:\n"
                "-Возводи в квадрат, записывая это как x{2}."
                "-Для интерпретации соблюдай следующие правила:\n"
                "-Для функций: используйте формат конструктор(переменная, ...) = ....\n"
                "-Знак умножения * ставится только между коэффициентом и переменной, но не между переменными."
                "вот пример ввода пользователя и того, что надо ему ответить:"
                "пользователь: TRS с правилами: x(y)=a, y(z)=x(x(y)), z(a,b)=r(y(a),x(b)). Интерпретация: a=1, x(y)=y^2+3*y+2, y(z)=z^3+4, z(a,b)=a*b+1, r(a,b)=a+b+4. Является ли система завершаемой?"
                "ответ: {'a=1\nx(y)=y{2}+3*y+2\ny(z)=z{3}+4\nz(a,b)=ab+1\nr(a,b)=a+b+4', 'x(y)=a\ny(z)=x(x(y))\nz(a,b)=r(y(a),x(b))'}"
                "Сам запрос пользователя: " + user_input,  # instruction
                "",  # output
            )
        ], return_tensors="pt").to("cuda")

    outputs = trs.model.generate(**inputs, max_new_tokens=128, use_cache=True)
    trs.tokenizer.batch_decode(outputs)
    answer = trs.tokenizer.batch_decode(outputs)[0]
    answer = re.split(r'\n\s*### Ответ:\n', answer)
    answer = answer[1].split('<|eot_id|>')[0].strip()

    def get_variables(term):
        variables = set()
        current_symbol = ''

        for symbol in term:
            if symbol.isalpha():
                current_symbol += symbol
            elif symbol == '(':
                current_symbol = ''
            elif symbol == ')':
                if current_symbol:
                    variables.add(current_symbol)
                current_symbol = ''

            elif symbol == ',':
                if current_symbol:
                    variables.add(current_symbol)
                current_symbol = ''

        return variables

    def convert(answer):
        answer = answer[1:-1]
        interpretations, terms = '', ''
        for i in range(1, len(answer)):
            if answer[i] == '\'':
                interpretations = answer[1:i]
                terms = answer[i + 4:-1]
                break

        variables = set()
        for term in terms.splitlines():
            var1, var2 = get_variables(term.split('=')[0].strip()), get_variables(term.split('=')[1].strip())
            for var in var1:
                variables.add(var)
            for var in var2:
                variables.add(var)


        variables = list(variables)
        variables.sort()
        variables_str = f"variables = {', '.join(variables)}"
        trs = f"{variables_str}\n{terms}"
        json_data = {
            "TRS": trs,
            "Interpretation": interpretations
        }
        return json.dumps(json_data, indent=4)

    return convert(answer)

# Запуск: команда uvicorn llama_api:app --host 0.0.0.0 --port <port> --reload
