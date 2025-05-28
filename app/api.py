import base64
import io
import os
import numpy as np
import torch
import uvicorn
import cv2
from PIL import Image
from fastapi import FastAPI, Request
from pydantic import BaseModel
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

# ---------- Configs ----------
CHECKPOINTS_DIR = "./checkpoints"
FACE_ENCODER_DIR = "./models/antelopev2"
BASE_MODEL = "wangqixun/YamerMIX_v8"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- FastAPI Init ----------
app = FastAPI()

# ---------- Load Models Once ----------
print("🔄 Loading face encoder...")
face_app = FaceAnalysis(name='antelopev2', root="./", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

print("🔄 Loading ControlNet...")
controlnet_path = os.path.join(CHECKPOINTS_DIR, "ControlNetModel")
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

print("🔄 Loading StableDiffusionXLInstantID pipeline...")
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.cuda()

print("🔄 Loading IP Adapter...")
pipe.load_ip_adapter_instantid(os.path.join(CHECKPOINTS_DIR, "ip-adapter.bin"))

pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

# ---------- Input Schema ----------
class FaceSwapRequest(BaseModel):
    face_image: str
    target_image: str
    prompt: str
    negative_prompt: str = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"
    controlnet_conditioning_scale: float = 0.8
    ip_adapter_scale: float = 0.8

# ---------- Helper: Base64 to PIL ----------
def base64_to_pil(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

# ---------- Helper: PIL to Base64 ----------
def pil_to_base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ---------- API Route ----------
@app.post("/swap-face")
def swap_face(req: FaceSwapRequest):
    print("⚙️ Processing request...")

    # Convert images
    face_image = base64_to_pil(req.face_image)
    target_image = base64_to_pil(req.target_image)

    # Get face embedding
    face_info = face_app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    if not face_info:
        return {"error": "No face detected in face_image"}

    face_info = sorted(face_info, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]))[-1]
    face_emb = face_info['embedding']
    face_kps = draw_kps(target_image, face_info['kps'])

    # Generate
    result_image = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=req.controlnet_conditioning_scale,
        ip_adapter_scale=req.ip_adapter_scale,
        num_inference_steps=30,
        guidance_scale=5,
    ).images[0]

    # Return image
    return {"output": pil_to_base64(result_image)}
