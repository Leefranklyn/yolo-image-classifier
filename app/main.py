from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.predictor import predictor

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

templates = Jinja2Templates(directory="app/templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="SCENE//CLASSIFIER", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def _context(request: Request) -> dict:
    return {
        "request": request,
        "model_state": predictor.state,
        "classes": predictor.classes,
        "weights_file": predictor.weights_file,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", _context(request))


async def _predict_response(request: Request, prediction, exclude=None):
    """Render the result partial and return wrapper JSON used by the client."""
    payload = {**_context(request), **(exclude or {})}
    if prediction:
        payload["prediction"] = prediction
    html = templates.TemplateResponse(request, "partials/result.html", payload)
    data = prediction.model_dump() if prediction else {"error": exclude.get("error")}
    return JSONResponse({"html": html.body.decode("utf-8"), "data": data})


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if predictor.state != "ready":
        return await _predict_response(request, None, {"error": "offline"})
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="UNSUPPORTED MEDIA // IMAGE REQUIRED")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PAYLOAD EXCEEDS 10.0MB LIMIT")
    try:
        prediction = predictor.predict(data)
    except Exception:
        raise HTTPException(status_code=422, detail="UNREADABLE IMAGE STREAM")
    return await _predict_response(request, prediction)


@app.get("/health")
async def health():
    return {
        "model_state": predictor.state,
        "weights_file": predictor.weights_file,
        "n_classes": len(predictor.classes),
    }
