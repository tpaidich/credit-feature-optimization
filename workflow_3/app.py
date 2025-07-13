from fastapi import FastAPI, File, UploadFile
from threshold_summary import analyze_thresholds
import tempfile
import shutil
import traceback

app = FastAPI()

@app.post("/threshold-summary")
async def threshold_summary(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
            shutil.copyfileobj(file.file, temp)
            temp.flush()
            print(f"Temp file path: {temp.name}")
            result = analyze_thresholds(temp.name)
        return result
    except Exception as e:
        print("Exception occurred in /threshold-summary")
        traceback.print_exc()
        return {"error": str(e)}
