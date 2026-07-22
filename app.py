from __future__ import annotations

import base64
import binascii
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = FastAPI(
    title="Chest Disease Classification API",
    description="FastAPI web interface for the existing chest disease classifier.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    image: str


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self._classifier: Any | None = None

    @property
    def classifier(self):
        if self._classifier is None:
            from cnnClassifier.pipeline.predict import PredictionPipeline

            self._classifier = PredictionPipeline(filename=self.filename)
        return self._classifier


clAPP = ClientApp()


def decode_request_image(imgstring: str, file_name: str) -> None:
    try:
        imgdata = base64.b64decode(imgstring, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload.") from exc

    with open(file_name, "wb") as image_file:
        image_file.write(imgdata)


@app.get("/", response_class=FileResponse)
async def home():
    return FileResponse(PROJECT_ROOT / "templates" / "index.html", media_type="text/html")


@app.api_route("/train", methods=["GET", "POST"], response_class=PlainTextResponse)
async def trainroute():
    completed_process = subprocess.run(
        ["dvc", "repro"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed_process.returncode != 0:
        detail = completed_process.stderr or completed_process.stdout or "Training failed."
        raise HTTPException(status_code=500, detail=detail)
    return "Training done successfully!!"


@app.post("/predict", response_class=JSONResponse)
async def predict_route(payload: PredictionRequest) -> List[dict]:
    image_data = payload.image.strip()
    if not image_data:
        raise HTTPException(status_code=400, detail="Image data is required.")

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        decode_request_image(image_data, clAPP.filename)
        return clAPP.classifier.predict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
