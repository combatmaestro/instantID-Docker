import io
import base64
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from diffusers import EulerDiscreteScheduler, LCMScheduler
from diffusers.utils import load_image
import cv2
import random

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from controlnet_aux import MidasDetector
from diffusers.models import ControlNetModel

# Setup
app = FastAPI()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load InsightFace
face_analyzer = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Load ControlNet and pipeline
controlnet = ControlNetModel.from_pretrained("./checkpoints/ControlNetModel", torch_dtype=dtype)
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "wangqixun/YamerMIX_v8",
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None,
    feature_extractor=None,
).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_ip_adapter_instantid("./checkpoints/ip-adapter.bin")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.disable_lora()

# Midas for depth (optional)
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

# Utility functions
def decode_base64_image(base64_str):
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))

def resize_img(input_image, max_side=1280, min_side=1024, base_pixel_number=64):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    w, h = round(ratio * w), round(ratio * h)
    w = (w // base_pixel_number) * base_pixel_number
    h = (h // base_pixel_number) * base_pixel_number
    return input_image.resize([w, h], Image.BILINEAR)

def convert_pil_to_cv2(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_kps(image_pil, kps):
    from pipeline_stable_diffusion_xl_instantid_full import draw_kps
    return draw_kps(image_pil, kps)

# Request model
class InferenceInput(BaseModel):
    face_image: str
    pose_image: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = ""
    style_name: Optional[str] = "Watercolor"
    num_steps: Optional[int] = 30
    identitynet_strength_ratio: Optional[float] = 0.8
    adapter_strength_ratio: Optional[float] = 0.8
    guidance_scale: Optional[float] = 5.0
    seed: Optional[int] = 42
    randomize_seed: Optional[bool] = True
    enable_LCM: Optional[bool] = False
    enhance_face_region: Optional[bool] = True

@app.post("/blend")
def blend_face(data: InferenceInput):
    if data.randomize_seed:
        data.seed = random.randint(0, 2**31 - 1)

    face_img = decode_base64_image(data.face_image)
    face_img = resize_img(face_img)
    face_cv2 = convert_pil_to_cv2(face_img)

    face_info = face_analyzer.get(face_cv2)
    if not face_info:
        raise HTTPException(status_code=400, detail="No face detected in source image.")

    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_img, face_info['kps'])

    if data.pose_image:
        pose_img = decode_base64_image(data.pose_image)
        pose_img = resize_img(pose_img)
        pose_cv2 = convert_pil_to_cv2(pose_img)
        pose_info = face_analyzer.get(pose_cv2)
        if not pose_info:
            raise HTTPException(status_code=400, detail="No face detected in pose image.")
        pose_info = pose_info[-1]
        face_kps = draw_kps(pose_img, pose_info['kps'])
        face_kps = face_kps.resize(pose_img.size)  # Force same size
        width, height = face_kps.size
    else:
        face_kps = face_kps.resize(face_img.size)  # Force same size
        width, height = face_kps.size

    control_mask = None
    if data.enhance_face_region:
        control_mask_np = np.zeros([height, width, 3], dtype=np.uint8)
        x1, y1, x2, y2 = map(int, face_info["bbox"])
        control_mask_np[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask_np)

        if control_mask.size != face_kps.size:
            control_mask = control_mask.resize(face_kps.size, resample=Image.BILINEAR)

    if data.enable_LCM:
        pipe.enable_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.disable_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.set_ip_adapter_scale(data.adapter_strength_ratio)
    generator = torch.Generator(device=device).manual_seed(data.seed)

    result = pipe(
        prompt=data.prompt,
        negative_prompt=data.negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        control_mask=control_mask,
        controlnet_conditioning_scale=float(data.identitynet_strength_ratio),
        num_inference_steps=data.num_steps,
        guidance_scale=data.guidance_scale,
        height=height,
        width=width,
        generator=generator
    ).images[0]

    buffered = io.BytesIO()
    result.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"image": f"data:image/png;base64,{img_str}"}

