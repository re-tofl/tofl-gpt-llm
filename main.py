from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
from celery.result import AsyncResult
from celery_worker import process_question

class QuestionRequest(BaseModel):
    question: str

app = FastAPI()

@app.post("/ask/")
async def ask_llama(request: QuestionRequest):
    task = process_question.apply_async(args=[request.question])
    return {"task_id": task.id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task_result = AsyncResult(task_id)
    if task_result.state == 'PENDING':
        return {"status": "Processing"}
    elif task_result.state == 'SUCCESS':
        return {"status": "Completed", "result": task_result.result}
    else:
        return {"status": task_result.state}
