import base64
import io
import os
import requests
import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
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
    face_image: Optional[str] = None
    face_image_url: Optional[str] = None
    target_image: Optional[str] = None
    target_image_url: Optional[str] = None
    prompt: str
    negative_prompt: str = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"
    controlnet_conditioning_scale: float = 0.8
    ip_adapter_scale: float = 0.8

# ---------- Helpers ----------
def base64_to_pil(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def pil_to_base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def load_image_input(base64_str: Optional[str], url: Optional[str]):
    if base64_str:
        return base64_to_pil(base64_str)
    elif url:
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch image from URL: {url}")
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        raise ValueError("No valid image input provided")

# ---------- API Route ----------
@app.post("/swap-face")
def swap_face(req: FaceSwapRequest):
    print("⚙️ Processing request...")

    try:
        face_image = load_image_input(req.face_image, req.face_image_url)
        target_image = load_image_input(req.target_image, req.target_image_url)

        # Detect face and extract embedding
        face_info = face_app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        if not face_info:
            return {"error": "No face detected in face_image"}

        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        face_emb = face_info['embedding']
        face_kps = draw_kps(target_image, face_info['kps'])

        print("🎨 Generating image with prompt:", req.prompt)

        # Generate image
        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=req.controlnet_conditioning_scale,
            ip_adapter_scale=req.ip_adapter_scale,
            num_inference_steps=30,
            guidance_scale=5,
        )

        result_image = result.images[0]
        return {"output": pil_to_base64(result_image)}

    except Exception as e:
        print("❌ Error during face swap:", e)
        return {"error": str(e)}
