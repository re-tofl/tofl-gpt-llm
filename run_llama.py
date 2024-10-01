import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login


login(token='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')


model_id = "meta-llama/Meta-Llama-3-8B"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type='nf4',
    bnb_8bit_compute_dtype=torch.float16
)


tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="F:/huggingface")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir="F:/huggingface"
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
    print("LLaMA Model is ready! Type your question below or 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        response = ask_model(model, tokenizer, user_input, device=device)
        print(f"LLaMA: {response}")
