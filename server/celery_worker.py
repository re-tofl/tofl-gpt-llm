from celery import Celery
from test_func import ask_model, model, tokenizer

app = Celery('tasks',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/0')

@app.task
def process_question(question):
    return ask_model(model, tokenizer, question)
