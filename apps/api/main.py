from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from io import BytesIO

from PIL import Image
from pathlib import Path

from .model import load_model, predict_image

MODEL_PATH = Path(__file__).parent / "models" / "best_cat_cnn.pth"

# Define the device and model states of the application before the run
# and before shutting down
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    app.state.model, app.state.device = load_model(MODEL_PATH)

    yield

    app.state.device = None
    app.state.model = None

app = FastAPI(lifespan=lifespan)

# Add the CORS allow list so that the frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# @app.get() creates a route
# A route means that when that route is entered, run this function
@app.get("/")
def home():
    return {"message": "Cat or Not Cat API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: Request, file: UploadFile):
    model = getattr(request.app.state, "model", None)
    device = getattr(request.app.state, "device", None)

    if model is None or device is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    image_bytes = await file.read()

    try:
        # BytesIO takes in a raw image file and wraps it in a file-like object
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    prediction_result = predict_image(image=image, model=model, device=device)
    return prediction_result
