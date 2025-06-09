# InstantID API Server Implementation
# This creates REST endpoints similar to Replicate's InstantID service

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import cv2
import numpy as np
import base64
from PIL import Image
import io
import uuid
import asyncio
from datetime import datetime
import logging
from pathlib import Path

# Import InstantID dependencies
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="InstantID API",
    description="Zero-shot Identity-Preserving Generation API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class InstantIDRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(
        "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful",
        description="Negative prompt"
    )
    controlnet_conditioning_scale: float = Field(0.8, ge=0.0, le=2.0, description="ControlNet conditioning scale")
    ip_adapter_scale: float = Field(0.8, ge=0.0, le=2.0, description="IP adapter scale")
    num_inference_steps: int = Field(50, ge=1, le=100, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    width: int = Field(1024, description="Output image width")
    height: int = Field(1024, description="Output image height")

class PredictionResponse(BaseModel):
    id: str
    status: str
    output: Optional[str] = None  # Base64 encoded image
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class PredictionStatus(BaseModel):
    id: str
    status: str
    output: Optional[str] = None
    error: Optional[str] = None
    logs: Optional[str] = None

# Global variables for model management
pipeline = None
face_app = None
predictions = {}  # In-memory storage for demo purposes

class ModelManager:
    """Manages InstantID model loading and inference"""
    
    def __init__(self):
        self.pipeline = None
        self.face_app = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
    
    async def load_models(self):
        """Load InstantID models asynchronously"""
        if self.model_loaded:
            return
        
        try:
            logger.info("Loading InstantID models...")
            
            # Initialize face analysis
            self.face_app = FaceAnalysis(
                name='antelopev2', 
                root='./models/', 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Load ControlNet and pipeline
            face_adapter = './checkpoints/ip-adapter.bin'
            controlnet_path = './checkpoints/ControlNetModel'
            base_model = 'wangqixun/YamerMIX_v8'
            
            controlnet = ControlNetModel.from_pretrained(
                controlnet_path, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.pipeline = StableDiffusionXLInstantIDPipeline.from_pretrained(
                base_model,
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to("cuda")
            
            # Load IP adapter
            self.pipeline.load_ip_adapter_instantid(face_adapter)
            
            self.model_loaded = True
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

    def extract_face_features(self, face_image: Image.Image):
        """Extract face embeddings and keypoints from image"""
        try:
            # Convert PIL to OpenCV format
            face_array = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
            # Get face info
            face_info = self.face_app.get(face_array)
            if not face_info:
                raise ValueError("No face detected in the image")
            # Use the largest face
            face_info = sorted(
                face_info, 
                key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1])
            )[-1]
            face_emb = face_info['embedding']
            face_kps = draw_kps(face_image, face_info['kps'])
            return face_emb, face_kps
        except Exception as e:
            logger.error(f"Error extracting face features: {str(e)}")
            raise ValueError(f"Failed to process face image: {str(e)}")

    async def generate_image(self, face_image: Image.Image, request: InstantIDRequest) -> Image.Image:
        """Generate image using InstantID"""
        if not self.model_loaded:
            await self.load_models()
        try:
            face_emb, face_kps = self.extract_face_features(face_image)
            if request.seed is not None:
                torch.manual_seed(request.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(request.seed)
            result = self.pipeline(
                request.prompt,
                negative_prompt=request.negative_prompt,
                image_embeds=face_emb,
                image=face_kps,
                controlnet_conditioning_scale=request.controlnet_conditioning_scale,
                ip_adapter_scale=request.ip_adapter_scale,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
            )
            return result.images[0]
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise ValueError(f"Failed to generate image: {str(e)}")

model_manager = ModelManager()

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def base64_to_image(base64_str: str) -> Image.Image:
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

async def process_prediction(prediction_id: str, face_image: Image.Image, request: InstantIDRequest):
    try:
        predictions[prediction_id]["status"] = "processing"
        predictions[prediction_id]["logs"] = "Starting image generation..."
        result_image = await model_manager.generate_image(face_image, request)
        output_base64 = image_to_base64(result_image)
        predictions[prediction_id].update({
            "status": "succeeded",
            "output": output_base64,
            "completed_at": datetime.utcnow().isoformat(),
            "logs": "Image generation completed successfully"
        })
    except Exception as e:
        logger.error(f"Prediction {prediction_id} failed: {str(e)}")
        predictions[prediction_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat(),
            "logs": f"Error: {str(e)}"
        })

@app.on_event("startup")
async def startup_event():
    logger.info("Starting InstantID API server...")

@app.get("/")
async def root():
    return {
        "message": "InstantID API Server",
        "status": "running",
        "model_loaded": model_manager.model_loaded
    }

@app.post("/predictions", response_model=PredictionResponse)
async def create_prediction(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Face reference image"),
    prompt: str = Form(..., description="Text prompt"),
    negative_prompt: Optional[str] = Form(None),
    controlnet_conditioning_scale: float = Form(0.8),
    ip_adapter_scale: float = Form(0.8),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.5),
    seed: Optional[int] = Form(None),
    width: int = Form(1024),
    height: int = Form(1024)
):
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        image_data = await image.read()
        face_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        request = InstantIDRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height
        )
        prediction_id = str(uuid.uuid4())
        predictions[prediction_id] = {
            "id": prediction_id,
            "status": "starting",
            "output": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "logs": "Prediction created"
        }
        background_tasks.add_task(process_prediction, prediction_id, face_image, request)
        return PredictionResponse(**predictions[prediction_id])
    except Exception as e:
        logger.error(f"Error creating prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{prediction_id}", response_model=PredictionStatus)
async def get_prediction(prediction_id: str):
    if prediction_id not in predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return PredictionStatus(**predictions[prediction_id])

@app.post("/predictions/{prediction_id}/cancel")
async def cancel_prediction(prediction_id: str):
    if prediction_id not in predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")
    prediction = predictions[prediction_id]
    if prediction["status"] in ["succeeded", "failed"]:
        raise HTTPException(status_code=400, detail="Prediction already completed")
    predictions[prediction_id].update({
        "status": "canceled",
        "completed_at": datetime.utcnow().isoformat(),
        "logs": "Prediction canceled by user"
    })
    return {"message": "Prediction canceled"}

@app.get("/predictions")
async def list_predictions():
    return {"predictions": list(predictions.values())}

@app.delete("/predictions/{prediction_id}")
async def delete_prediction(prediction_id: str):
    if prediction_id not in predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")
    del predictions[prediction_id]
    return {"message": "Prediction deleted"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_manager.model_loaded,
        "device": model_manager.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "active_predictions": len([p for p in predictions.values() if p["status"] == "processing"])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
