import torch
from celery import Celery
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

app = Celery('tasks',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/0')

# Вход в huggingface
login(token='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')

# Переменные для хранения модели и токенизатора
model = None
tokenizer = None


# Функция для загрузки модели (вызывается только в Celery воркере)
def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        model_id = "meta-llama/Meta-Llama-3-8B"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type='nf4',
            bnb_8bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model.eval()

    # Функция для взаимодействия с моделью


def ask_model(model, tokenizer, question, device='cuda', max_length=200):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.5,  # Контролирует креативность: чем ниже, тем детерминированнее
            top_k=30,  # Сокращает набор токенов для выбора
            top_p=0.8,  # Ограничивает сумму вероятностей для выбора токенов
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


@app.task
def process_question(question):
    load_model()
    return ask_model(model, tokenizer, question)
