from fastapi import FastAPI, File, UploadFile
import pandas as pd
import tempfile
import shutil
import override_detect

app = FastAPI()

@app.post("/override-detection")
async def override_detection(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
        shutil.copyfileobj(file.file, temp)
        temp.flush()
        result = override_detect.analyze_overrides(temp.name)
    return result
