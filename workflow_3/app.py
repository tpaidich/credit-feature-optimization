from fastapi import FastAPI, File, UploadFile
from threshold_summary import analyze_thresholds
import tempfile
import shutil

app = FastAPI()

@app.post("/threshold-summary")
async def threshold_summary(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
        shutil.copyfileobj(file.file, temp)
        temp.flush()
        result = analyze_thresholds(temp.name)
    return result
