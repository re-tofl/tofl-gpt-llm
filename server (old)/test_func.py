model = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = "token"

def ask_model(model, tokenizer, question, device='cuda', max_length=200):
    answer = f'ответ для {model}, {tokenizer} на вопрос: {question}'
    return answer