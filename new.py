import sys
sys.path.append('./')

import os
import torch
import cv2
import random
import numpy as np
from PIL import Image
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64

from insightface.app import FaceAnalysis
from diffusers.models import ControlNetModel
from diffusers import EulerDiscreteScheduler, LCMScheduler
from diffusers.utils import load_image

from style_template import styles, DEFAULT_STYLE_NAME
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl, get_torch_device, torch_gc

# --- Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = get_torch_device()
dtype = torch.float16 if "cuda" in str(device) else torch.float32
MAX_SEED = np.iinfo(np.int32).max

STYLE_NAMES = list(styles.keys())
face_adapter = './checkpoints/ip-adapter.bin'
controlnet_path = './checkpoints/ControlNetModel'

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype).to(device)
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "wangqixun/YamerMIX_v8",
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None,
    feature_extractor=None,
).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter_instantid(face_adapter)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.disable_lora()

face_app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


# --- Utilities ---
def convert_to_cv2(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def convert_to_image(img_cv2: np.ndarray):
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

def resize_img(input_image: Image.Image, max_side=1280, min_side=1024):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    w, h = round(ratio * w), round(ratio * h)
    return input_image.resize((w, h), Image.BILINEAR)

def apply_style(style_name: str, positive: str, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative


# --- Routes ---

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/styles")
async def get_styles():
    return {"styles": STYLE_NAMES}

@app.post("/generate")
async def generate(
    face_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    style: str = Form(DEFAULT_STYLE_NAME),
    adapter_strength: float = Form(0.8),
    identitynet_strength: float = Form(0.8),
    steps: int = Form(30),
    guidance: float = Form(5.0),
    seed: int = Form(42),
    enable_lcm: bool = Form(False),
    enhance_face_region: bool = Form(True),
    pose_image: Optional[UploadFile] = File(None)
):
    if enable_lcm:
        pipe.enable_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.disable_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    image = Image.open(BytesIO(await face_image.read())).convert("RGB")
    image = resize_img(image)
    image_cv2 = convert_to_cv2(image)
    height, width, _ = image_cv2.shape

    face_info = face_app.get(image_cv2)
    if len(face_info) == 0:
        return JSONResponse({"error": "No face found in face image"}, status_code=400)

    face_info = sorted(face_info, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    face_emb = face_info['embedding']
    kps = face_info['kps']

    if pose_image:
        pose = Image.open(BytesIO(await pose_image.read())).convert("RGB")
        pose = resize_img(pose)
        pose_cv2 = convert_to_cv2(pose)
        face_info = face_app.get(pose_cv2)
        if len(face_info) == 0:
            return JSONResponse({"error": "No face found in pose image"}, status_code=400)
        kps = face_info[-1]['kps']

    # Create control image
    control_img = np.zeros((height, width, 3), dtype=np.uint8)
    for (x, y) in kps:
        cv2.circle(control_img, (int(x), int(y)), 5, (255, 0, 0), -1)
    control_pil = convert_to_image(control_img)

    control_mask = None
    if enhance_face_region:
        x1, y1, x2, y2 = map(int, face_info['bbox'])
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(mask)

    # Apply style
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

    generator = torch.Generator(device=device).manual_seed(seed)
    pipe.set_ip_adapter_scale(adapter_strength)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=control_pil,
        control_mask=control_mask,
        controlnet_conditioning_scale=float(identitynet_strength),
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=generator
    ).images[0]

    buffer = BytesIO()
    result.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return {"image_base64": img_str}


# Optional root redirect
@app.get("/")
def root():
    return {"message": "InstantID API is running."}
