import requests
import json

API_URL = "http://127.0.0.1:8000/process"


def send_request(prompt):
    headers = {"Content-Type": "application/json"}
    type = 0
    data = {
        "type": type,
        "prompt": prompt
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"


if __name__ == "__main__":
    print("Chat with the model. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        model_response = send_request(user_input)
        print(f"Model: {model_response}")
