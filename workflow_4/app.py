# app.py

from fastapi import FastAPI, File, UploadFile
from workflow_4 import analyze_stability
import tempfile
import shutil

app = FastAPI()

@app.post("/model-stability")
async def model_stability(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
        shutil.copyfileobj(file.file, temp)
        temp.flush()
        result = analyze_stability(temp.name)
    return result
