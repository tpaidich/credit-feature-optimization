from fastapi import FastAPI, File, UploadFile
from downturn_summary import analyze_downturn
import tempfile
import shutil

app = FastAPI()

@app.post("/downturn-summary")
async def downturn_summary(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
            shutil.copyfileobj(file.file, temp)
            temp.flush()
            result = analyze_downturn(temp.name)
        return result
    except Exception as e:
        # Print error to container logs
        print(f"ERROR in analyze_downturn: {str(e)}")
        return {"error": str(e)}
