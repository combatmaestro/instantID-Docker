# instantid_api.py
import sys
sys.path.append("./")

import os
import math
import torch
import random
import numpy as np
from PIL import Image
from io import BytesIO

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from diffusers import LCMScheduler, EulerDiscreteScheduler
from diffusers.utils import load_image
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
from style_template import styles
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import get_torch_device
from controlnet_util import openpose, get_depth_map, get_canny_image

# Init
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = get_torch_device()
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Models
face_adapter = "./checkpoints/ip-adapter.bin"
controlnet_identitynet = ControlNetModel.from_pretrained(
    "./checkpoints/ControlNetModel", torch_dtype=dtype
)
controlnet_map = {
    "pose": ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=dtype).to(device),
    "canny": ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=dtype).to(device),
    "depth": ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0-small", torch_dtype=dtype).to(device),
}
controlnet_map_fn = {
    "pose": openpose,
    "canny": get_canny_image,
    "depth": get_depth_map,
}

STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"
MAX_SEED = np.iinfo(np.int32).max

face_app = FaceAnalysis(name="antelopev2", root="./", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "wangqixun/YamerMIX_v8",
    controlnet=[controlnet_identitynet],
    torch_dtype=dtype,
    safety_checker=None,
    feature_extractor=None,
).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter_instantid(face_adapter)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.disable_lora()

# Utilities
def resize_img(input_image: Image.Image, max_side=1024):
    w, h = input_image.size
    ratio = min(1024 / min(h, w), max_side / max(h, w))
    return input_image.resize((round(w * ratio), round(h * ratio)), Image.BILINEAR)

def apply_style(style_name: str, positive: str, negative: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + " " + negative

def convert_to_image(img_cv2):
    import cv2
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

# API Route
@app.post("/generate")
async def generate(
    face_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    style: str = Form(DEFAULT_STYLE_NAME),
    identitynet_strength: float = Form(0.8),
    adapter_strength: float = Form(0.8),
    pose_strength: float = Form(0.4),
    canny_strength: float = Form(0.3),
    depth_strength: float = Form(0.5),
    controlnets: str = Form("pose"),
    steps: int = Form(30),
    guidance: float = Form(5.0),
    seed: int = Form(42),
    enable_lcm: bool = Form(False),
    enhance_face_region: bool = Form(True),
    pose_image: UploadFile = File(None)
):
    try:
        if enable_lcm:
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.enable_lora()
        else:
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            pipe.disable_lora()

        # Prep face image
        face = Image.open(BytesIO(await face_image.read())).convert("RGB")
        face = resize_img(face)
        face_np = np.array(face)
        face_cv2 = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        height, width, _ = face_cv2.shape

        face_info = face_app.get(face_cv2)
        if not face_info:
            return JSONResponse({"error": "No face found"}, status_code=400)

        face_info = sorted(face_info, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        face_emb = face_info["embedding"]

        if pose_image:
            pose = Image.open(BytesIO(await pose_image.read())).convert("RGB")
            pose = resize_img(pose)
            control_img = pose
        else:
            control_img = face

        bbox = list(map(int, face_info["bbox"]))
        if enhance_face_region:
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
            control_mask = Image.fromarray(mask)
        else:
            control_mask = None

        selected_cn = controlnets.split(",") if controlnets else []
        control_images = [face]
        control_scales = [identitynet_strength]

        if selected_cn:
            pipe.controlnet = MultiControlNetModel(
                [controlnet_identitynet] + [controlnet_map[c] for c in selected_cn]
            )
            control_scales += [
                pose_strength if "pose" in selected_cn else 0,
                canny_strength if "canny" in selected_cn else 0,
                depth_strength if "depth" in selected_cn else 0,
            ]
            control_images += [
                controlnet_map_fn[c](control_img).resize((width, height)) for c in selected_cn
            ]
        else:
            pipe.controlnet = controlnet_identitynet

        prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

        generator = torch.Generator(device=device).manual_seed(seed)
        pipe.set_ip_adapter_scale(adapter_strength)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=control_images,
            control_mask=control_mask,
            controlnet_conditioning_scale=control_scales,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        buffer = BytesIO()
        result.save(buffer, format="PNG")
        return {"image_base64": base64.b64encode(buffer.getvalue()).decode()}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/styles")
def get_styles():
    return {"styles": STYLE_NAMES}


@app.get("/")
def root():
    return {"message": "InstantID API with multi-controlnet is running."}
