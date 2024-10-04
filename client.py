import requests
import time

API_URL = "http://localhost:8000"

def ask_question(question):
    response = requests.post(
        f"{API_URL}/ask/",
        json={"question": question},
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        task_id = response.json().get("task_id")
        print(f"ID задачи: {task_id}")
        return task_id
    else:
        print(f"Ошибка при отправке запроса: {response.status_code}")
        print(response.text)
        return None


def get_result(task_id):
    response = requests.get(f"{API_URL}/result/{task_id}")

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Ошибка при получении результата: {response.status_code}")
        return None


if __name__ == "__main__":
    print("Введите exit для выхода.")
    while True:
        question = input("Введите вопрос для нейросети: ")

        if question == 'exit':
            break

        task_id = ask_question(question)

        if task_id:
            while True:
                time.sleep(3)
                result = get_result(task_id)

                if result:
                    status = result.get("status")
                    if status == "Completed":
                        print(f"Ответ нейросети: {result.get('result')}")
                        break
                    elif status == "Processing":
                        print("Задача все еще выполняется...")
                    else:
                        print(f"Состояние задачи: {status}")
                else:
                    print("Не удалось получить результат.")
                    break
