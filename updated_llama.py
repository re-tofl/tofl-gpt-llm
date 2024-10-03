import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

login(token='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')

model_path = "F:/huggingface_cache/hub/finetuned_llama"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is: {device}")

offload_dir = "F:/huggingface_cache/offload"

os.makedirs(offload_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_dir=offload_dir,
)

model.eval()

def ask_model(model, tokenizer, question, device='cuda', max_length=200):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.5,
            top_k=30,
            top_p=0.8,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Fine-tuned LLaMA Model is ready! Type your question below or 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        response = ask_model(model, tokenizer, user_input, device=device)
        print(f"LLaMA: {response}")
