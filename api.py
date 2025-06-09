# instantid_api.py

import io
import os
import cv2
import math
import base64
import torch
import random
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Union
from .model_util import load_models_xl, get_torch_device
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from .controlnet_util import openpose, get_depth_map, get_canny_image
from diffusers import ControlNetModel, EulerDiscreteScheduler, LCMScheduler
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from .style_template import styles

# Setup
app = FastAPI()
MAX_SEED = np.iinfo(np.int32).max
DEFAULT_STYLE_NAME = "Watercolor"
device = get_torch_device()
dtype = torch.float16 if "cuda" in str(device) else torch.float32
STYLE_NAMES = list(styles.keys())

# Load Face Analyzer
face_analyzer = FaceAnalysis(name="antelopev2", root="./", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Load ControlNets
controlnet_identitynet = ControlNetModel.from_pretrained("./checkpoints/ControlNetModel", torch_dtype=dtype, use_safetensors=True).to(device)
controlnet_pose = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=dtype).to(device)
controlnet_canny = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=dtype).to(device)
controlnet_depth = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0-small", torch_dtype=dtype).to(device)

controlnet_map = {"pose": controlnet_pose, "canny": controlnet_canny, "depth": controlnet_depth}
controlnet_map_fn = {"pose": openpose, "canny": get_canny_image, "depth": get_depth_map}

# Load Pipeline
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "wangqixun/YamerMIX_v8",
    controlnet=[controlnet_identitynet],
    torch_dtype=dtype,
    safety_checker=None,
    feature_extractor=None
).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter_instantid("./checkpoints/ip-adapter.bin")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.disable_lora()

# Utils
def decode_image(input_data: str) -> Image.Image:
    if input_data.startswith("http://") or input_data.startswith("https://"):
        return load_image(input_data)
    return Image.open(io.BytesIO(base64.b64decode(input_data)))

def resize_img(input_image: Image.Image, max_side=1280, min_side=1024) -> Image.Image:
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    w, h = round(ratio * w), round(ratio * h)
    w -= w % 64
    h -= h % 64
    input_image = input_image.resize((w, h), Image.BILINEAR)
    return input_image

def convert_to_cv2(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_kps(img: Image.Image, kps):
    from pipeline_stable_diffusion_xl_instantid_full import draw_kps
    return draw_kps(img, kps)

def apply_style(style_name, positive, negative=""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + " " + negative

# Request Model
class BlendInput(BaseModel):
    face_image: str
    pose_image: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = ""
    style_name: Optional[str] = DEFAULT_STYLE_NAME
    num_steps: Optional[int] = 30
    identitynet_strength_ratio: Optional[float] = 0.8
    adapter_strength_ratio: Optional[float] = 0.8
    pose_strength: Optional[float] = 0.4
    canny_strength: Optional[float] = 0.4
    depth_strength: Optional[float] = 0.4
    controlnet_selection: Optional[List[str]] = ["pose"]
    guidance_scale: Optional[float] = 5.0
    seed: Optional[int] = 1016821440
    scheduler: Optional[str] = "EulerDiscreteScheduler"
    randomize_seed: Optional[bool] = True
    enable_LCM: Optional[bool] = False
    enhance_face_region: Optional[bool] = True

@app.post("/blend")
def blend_face(data: BlendInput):
    if data.randomize_seed:
        data.seed = random.randint(0, MAX_SEED)

    prompt, negative_prompt = apply_style(data.style_name, data.prompt, data.negative_prompt)

    face_image = resize_img(decode_image(data.face_image))
    face_cv2 = convert_to_cv2(face_image)
    height, width, _ = face_cv2.shape

    face_info = face_analyzer.get(face_cv2)
    if not face_info:
        raise HTTPException(status_code=400, detail="No face detected in face_image")

    face_info = sorted(face_info, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]))[-1]
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])

    img_controlnet = face_image
    if data.pose_image:
        pose_image = resize_img(decode_image(data.pose_image))
        pose_cv2 = convert_to_cv2(pose_image)
        pose_info = face_analyzer.get(pose_cv2)
        if not pose_info:
            raise HTTPException(status_code=400, detail="No face detected in pose_image")
        face_kps = draw_kps(pose_image, pose_info[-1]['kps'])
        img_controlnet = pose_image
        width, height = face_kps.size

    control_mask = None
    if data.enhance_face_region:
        mask = np.zeros([height, width, 3])
        x1, y1, x2, y2 = map(int, face_info["bbox"])
        mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(mask.astype(np.uint8)).resize((width, height))

    if data.enable_LCM:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_lora()
    else:
        pipe.disable_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if data.controlnet_selection:
        pipe.controlnet = MultiControlNetModel([
            controlnet_identitynet
        ] + [controlnet_map[k] for k in data.controlnet_selection])

        control_scales = [float(data.identitynet_strength_ratio)] + [
            {"pose": data.pose_strength, "canny": data.canny_strength, "depth": data.depth_strength}[k] for k in data.controlnet_selection
        ]
        face_kps = face_kps.resize((width, height)) 
        control_images = [face_kps] + [
            controlnet_map_fn[k](img_controlnet).resize((width, height)) for k in data.controlnet_selection
        ]
    else:
        pipe.controlnet = controlnet_identitynet
        control_scales = float(data.identitynet_strength_ratio)
        control_images = face_kps.resize((width, height))

    pipe.set_ip_adapter_scale(data.adapter_strength_ratio)
    generator = torch.Generator(device=device).manual_seed(data.seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=control_images,
        control_mask=control_mask,
        controlnet_conditioning_scale=control_scales,
        num_inference_steps=data.num_steps,
        guidance_scale=data.guidance_scale,
        height=height,
        width=width,
        generator=generator
    ).images[0]

    buffer = io.BytesIO()
    result = result.convert("RGB")
    result.save(buffer, format="JPEG", quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"image": img_str}
