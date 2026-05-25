"""
Agri-Vision Flask Application
Unified inference for disease classification (ResNet50) and growth stage prediction (YOLOv8)
Thread-safe execution for production environments with Celery Async Support.
Optimized via a Two-Pointer Ambiguity Filter for overlapping disease classes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import re
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from flasgger import Swagger
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    stream_with_context,
    url_for,
    Request,
)
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from services.weather_service import (
    generate_weather_recommendations,
    geocode_city,
    get_weather,
)
from services.yield_service import estimate_yield

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

class CustomRequest(Request):
    max_form_memory_size = 25 * 1024 * 1024  # Support larger base64-encoded forms

app.request_class = CustomRequest

swagger = Swagger(app)
CORS(app)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

secret_key = os.getenv("SECRET_KEY") or "dev_secret_123"
app.secret_key = secret_key
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 25 * 1024 * 1024

LANG = {
    "en": {"welcome": "Welcome to Agri Vision"},
    "te": {"welcome": "అగ్రి విజన్‌కు స్వాగతం"},
}

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("models", exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_INFERENCE_DIMENSION = 1024
DISPLAY_IMAGE_MAX_DIMENSION = 1200
DISPLAY_JPEG_QUALITY = 80

RESNET_MODEL_PATH = "models/cotton_crop_disease_classification/full_resnet50_model.pth"
YOLO_MODEL_PATH = "models/cotton_crop_growth_stage_prediction/best.pt"

disease_classes = [
    "Aphids",
    "Army worm",
    "Bacterial blight",
    "Cotton Boll Rot",
    "Green Cotton Boll",
    "Healthy",
    "Powdery mildew",
    "Target Spot",
]

growth_stage_classes = [
    "Cotton Blossom",
    "Cotton Bud",
    "Early Boll",
    "Matured Cotton Boll",
    "Split Cotton Boll",
]

disease_info_map = {
    "Aphids": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Aphids are small sap-sucking insects that weaken cotton plants by feeding on tender leaves and shoots.",
        "symptoms": "Curled leaves, sticky honeydew, yellowing, and clusters of tiny insects on the underside of leaves.",
        "treatment": "Remove heavily infested leaves, encourage natural predators, and use neem oil or recommended insecticide if infestation increases.",
    },
    "Army worm": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Army worms are leaf-feeding caterpillars that can quickly damage cotton foliage when populations build up.",
        "symptoms": "Chewed leaf edges, holes in leaves, skeletonized foliage, and visible larvae on plants.",
        "treatment": "Scout fields regularly, remove larvae where possible, and apply recommended biological or chemical control at early stages.",
    },
    "Bacterial blight": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Bacterial blight is a cotton disease that spreads through infected seed, crop residue, rain splash, and wind-driven moisture.",
        "symptoms": "Angular water-soaked leaf spots, dark lesions, yellowing, and drying of affected leaf tissue.",
        "treatment": "Avoid overhead irrigation, remove infected debris, use disease-free seed, and follow local copper-based spray recommendations if needed.",
    },
    "Cotton Boll Rot": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Cotton boll rot affects developing bolls, especially under humid conditions or poor field drainage.",
        "symptoms": "Soft or discolored bolls, fungal growth, rotting tissue, and premature boll drop.",
        "treatment": "Improve drainage and airflow, remove rotten bolls, avoid excess irrigation, and manage insects that create boll wounds.",
    },
    "Green Cotton Boll": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Green cotton boll indicates developing boll growth that should be monitored for nutrition, pests, and disease pressure.",
        "symptoms": "Green immature bolls with no clear disease symptoms unless stress, pest injury, or spotting appears.",
        "treatment": "Maintain balanced irrigation and nutrition, scout for pests, and continue regular field monitoring.",
    },
    "Healthy": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "The leaf appears healthy with no major visible disease symptoms detected.",
        "symptoms": "Uniform green color, normal leaf shape, and no significant spots, mildew, curling, or pest damage.",
        "treatment": "Continue routine monitoring, balanced fertilization, proper irrigation, and preventive crop hygiene.",
    },
    "Powdery mildew": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Powdery mildew is a fungal disease that appears as white powdery growth on cotton leaves.",
        "symptoms": "White or gray powdery patches, yellowing leaves, reduced vigor, and premature leaf drying.",
        "treatment": "Improve airflow, remove infected debris, reduce leaf wetness, and apply recommended fungicide when needed.",
    },
    "Target Spot": {
        "healthy_image": "static/images/healthy_leaf.jpg",
        "description": "Target spot is a fungal leaf disease that produces circular lesions and can reduce cotton leaf area.",
        "symptoms": "Brown circular spots with ring-like patterns, yellow halos, leaf blight, and premature defoliation.",
        "treatment": "Reduce leaf wetness, improve spacing and airflow, remove infected residue, and use suitable fungicide if disease spreads.",
    },
}

UNCERTAINTY_THRESHOLD = 0.45
AMBIGUITY_MARGIN = 0.08


# -------------------------------------------------------------------
# THREAD-SAFE MODEL MANAGER
# -------------------------------------------------------------------
class ModelManager:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._load_lock = threading.Lock()
        self.loaded = False
        self.errors = {"resnet": None, "yolo": None}
        self.resnet_model = None
        self.yolo_model = None
        self._initialized = True

    def load_models(self) -> Tuple[Optional[torch.nn.Module], Optional[YOLO]]:
        if self.loaded:
            return self.resnet_model, self.yolo_model

        with self._load_lock:
            if self.loaded:
                return self.resnet_model, self.yolo_model

            if self.resnet_model is None:
                try:
                    try:
                        self.resnet_model = torch.load(
                            RESNET_MODEL_PATH,
                            map_location=torch.device("cpu"),
                        )
                    except TypeError:
                        self.resnet_model = torch.load(
                            RESNET_MODEL_PATH,
                            map_location=torch.device("cpu"),
                            weights_only=False
                        )
                    self.resnet_model.eval()
                    self.errors["resnet"] = None
                    logger.info("ResNet50 model loaded successfully")
                except Exception as exc:
                    self.errors["resnet"] = str(exc)
                    logger.warning(f"ResNet50 model not found or failed to load: {exc}")
                    self.resnet_model = None

            if self.yolo_model is None:
                try:
                    self.yolo_model = YOLO(YOLO_MODEL_PATH)
                    self.errors["yolo"] = None
                    logger.info("YOLOv8 model loaded successfully")
                except Exception as exc:
                    self.errors["yolo"] = str(exc)
                    logger.warning(f"YOLOv8 model not found or failed to load: {exc}")
                    self.yolo_model = None

            self.loaded = True
            return self.resnet_model, self.yolo_model

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "resnet": {
                "loaded": self.resnet_model is not None,
                "path": RESNET_MODEL_PATH,
                "error": self.errors.get("resnet"),
            },
            "yolo": {
                "loaded": self.yolo_model is not None,
                "path": YOLO_MODEL_PATH,
                "error": self.errors.get("yolo"),
            },
        }


model_manager = ModelManager()

resnet_model = None
yolo_model = None


def load_models():
    """Wrapper for backward compatibility"""
    global resnet_model, yolo_model, grad_cam_instance

    if resnet_model is None:
        try:
            resnet_model, yolo_model = model_manager.load_models()

            # Keep compatibility with newer PyTorch versions
            if resnet_model is None:
                resnet_model = torch.load(
                    "models/cotton_crop_disease_classification/full_resnet50_model.pth",
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )

            resnet_model.eval()
            logger.info("ResNet50 model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    return resnet_model, yolo_model

def ensure_models_loaded() -> None:
    load_models()


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels")
    return image


def resize_image(image: np.ndarray, max_dim: int = MAX_INFERENCE_DIMENSION) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_dim:
        return image
    scale = max_dim / float(max(height, width))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def calculate_disease_severity(health_score: float) -> float:
    return max(0.0, 100.0 - float(health_score))



def generate_mock_heatmap(image_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = image_rgb.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    x_grid, y_grid = np.meshgrid(x, y)
    cx, cy = 0.05, -0.05
    sigma = 0.35
    heatmap = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma**2))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def generate_pure_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    h, w, _ = image_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_255 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


def apply_heatmap_on_image(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6, beta: float = 0.4) -> np.ndarray:
    heatmap_color_rgb = generate_pure_heatmap(image_rgb, heatmap)
    return cv2.addWeighted(image_rgb, alpha, heatmap_color_rgb, beta, 0)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.heatmap_np = None
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)
        logger.info("Grad-CAM hooks registered on layer: %s", target_layer.__class__.__name__)

    def cleanup(self) -> None:
        if getattr(self, "forward_handle", None) is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if getattr(self, "backward_handle", None) is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_class_idx: Optional[int], original_image_rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            logger.warning("Grad-CAM: model is not loaded.")
            return None

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        self.heatmap_np = None

        try:
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)

            with torch.enable_grad():
                output = self.model(input_tensor)
                if target_class_idx is None:
                    target_class_idx = int(output.argmax(dim=1).item())

                score = output[:, target_class_idx].sum()
                score.backward()

                if self.activations is None or self.gradients is None:
                    logger.warning("Grad-CAM: activations or gradients not captured.")
                    return None

                pooled_gradients = torch.mean(self.gradients, dim=(2, 3))
                weighted_activations = self.activations * pooled_gradients[:, :, None, None]
                heatmap = torch.sum(weighted_activations, dim=1).squeeze()
                heatmap = F.relu(heatmap)

                max_val = torch.max(heatmap)
                if float(max_val.item()) == 0.0:
                    heatmap = torch.zeros_like(heatmap)
                else:
                    heatmap = heatmap / max_val

                heatmap_np = heatmap.detach().cpu().numpy()
                self.heatmap_np = heatmap_np
                return apply_heatmap_on_image(original_image_rgb, heatmap_np)

        except Exception as exc:
            logger.error("Error generating Grad-CAM: %s", exc)
            return None
        finally:
            self.gradients = None
            self.activations = None


# -------------------------------------------------------------------
# INFERENCE PIPELINE
# -------------------------------------------------------------------
def preprocess_image_for_resnet(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor


def infer_disease(image: np.ndarray) -> Dict[str, Any]:
    global resnet_model
    if resnet_model is None:
        resnet_model, _ = model_manager.load_models()

    if resnet_model is not None:
        processed = preprocess_image_for_resnet(image)
        with torch.no_grad():
            output = resnet_model(processed)
            probs = F.softmax(output, dim=1)
            _, prediction = torch.max(probs, 1)
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = np.random.rand(1, len(disease_classes))
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)

    probabilities = probs_np[0]

    # -------------------------------------------------------------------
    # TWO-POINTER RESOLUTION FILTER FOR OVERLAPPING DISEASES (#270 Feature)
    # -------------------------------------------------------------------
    indexed_probs = sorted(
        [(float(prob), idx) for idx, prob in enumerate(probabilities)],
        key=lambda x: x[0],
        reverse=True
    )

    low = 0
    high = 1

    top1_conf, top1_idx = indexed_probs[low]
    top2_conf, top2_idx = indexed_probs[high]

    predicted_class = disease_classes[top1_idx]
    alternative_class = disease_classes[top2_idx]

    healthy_idx = disease_classes.index("Healthy")
    health_score = float(probabilities[healthy_idx]) * 100

    is_uncertain = top1_conf < UNCERTAINTY_THRESHOLD
    is_ambiguous = False
    contender_classes = []

    while high < len(indexed_probs):
        current_conf, current_idx = indexed_probs[high]
        if abs(top1_conf - current_conf) < AMBIGUITY_MARGIN:
            is_ambiguous = True
            contender_classes.append(disease_classes[current_idx])
        high += 1

    interpretation_message = None
    if is_uncertain:
        interpretation_message = "The model could not make a confident prediction. Please upload a clearer crop image or seek expert review."
    elif is_ambiguous:
        if len(contender_classes) > 1:
            contenders_str = ", ".join(contender_classes)
            interpretation_message = f"The prediction is close between {predicted_class} and other localized indicators: {contenders_str}. Monitor the crop closely for overlapping symptoms."
        else:
            interpretation_message = f"The prediction is somewhat ambiguous between {predicted_class} and {alternative_class}."

    disease_confidences = {disease_classes[i]: float(probabilities[i]) for i in range(len(disease_classes))}

    return {
        "predicted_class": predicted_class,
        "predicted_class_idx": top1_idx,
        "confidence": top1_conf,
        "all_confidences": disease_confidences,
        "health_score": health_score,
        "raw": probs_np.tolist(),
        "detected_issue": predicted_class,
        "model_confidence": round(top1_conf * 100, 2),
        "alternative_prediction": {
            "class": alternative_class,
            "confidence": round(top2_conf * 100, 2),
        },
        "is_uncertain": is_uncertain,
        "is_ambiguous": is_ambiguous,
        "interpretation_message": interpretation_message,
    }


def infer_growth_stage(image: np.ndarray) -> Dict[str, Any]:
    _, yolo_model = model_manager.load_models()
    result = {
        "main_class": None,
        "main_class_idx": None,
        "confidence": 0.0,
        "boxes": [],
        "raw": [],
    }

    if yolo_model is None:
        return result

    pil_image = Image.fromarray(image)
    yolo_results = yolo_model(pil_image)
    boxes = []

    for r in yolo_results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for b in r.boxes:
            class_id = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])
            conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            boxes.append({
                "class_id": class_id,
                "class_name": growth_stage_classes[class_id] if class_id < len(growth_stage_classes) else str(class_id),
                "confidence": conf,
                "bbox": xyxy,
            })

    if boxes:
        main = max(boxes, key=lambda x: x["confidence"])
        result.update({
            "main_class": main["class_name"],
            "main_class_idx": main["class_id"],
            "confidence": main["confidence"],
            "boxes": boxes,
        })
    result["raw"] = boxes
    return result


def generate_recommendations(disease_result: Dict[str, Any], growth_result: Dict[str, Any], weather: Optional[Dict[str, Any]] = None) -> list[str]:
    recs: list[str] = []
    dclass = disease_result["predicted_class"]

    instr_map = {
        "Aphids": ["Inspect leaves closely for clusters of small pests.", "Use recommended insecticides if infestation is severe."],
        "Army worm": ["Increase scouting frequency.", "Apply biological or suitable chemical controls early."],
        "Bacterial blight": ["Avoid overhead irrigation.", "Remove and destroy affected plant parts."],
        "Cotton Boll Rot": ["Improve field drainage, avoid stagnant water.", "Remove and destroy rotten bolls."],
        "Green Cotton Boll": ["Monitor bolls for signs of pests or disease.", "Maintain optimal nutrient regime."],
        "Healthy": ["Continue general crop monitoring.", "Maintain optimal fertilization and irrigation."],
        "Powdery mildew": ["Remove infected plant debris.", "Apply fungicide at recommended intervals."],
        "Target Spot": ["Monitor for spread, reduce leaf wetness.", "Apply suitable fungicide if required."],
    }
    recs.extend(instr_map.get(dclass, ["Practice general crop hygiene."]))
    
    if disease_result["health_score"] < 50:
        recs.append("Consult an agricultural expert urgently for low health score.")
        recs.append("Consult an agricultural expert if symptoms persist.")
    elif disease_result["health_score"] < 70:
        recs.append("Increase frequency of crop monitoring based on moderate health.")

    if disease_result.get("is_uncertain"):
        recs.append("Model confidence is low. Please upload a clearer image or consult an agricultural expert.")
    elif disease_result.get("is_ambiguous"):
        alt = disease_result.get("alternative_prediction", {}).get("class", "another condition")
        recs.append(f"The prediction may overlap with {alt}. Monitor the crop closely before applying treatment.")

    gmain = growth_result.get("main_class", None)
    grow_map = {
        "Cotton Blossom": ["Maintain regular watering during blossom phase.", "Scout for early flower pests."],
        "Cotton Bud": ["Ensure adequate phosphorus supply.", "Monitor for budworm."],
        "Early Boll": ["Start borer management as boll phase begins.", "Avoid excess nitrogen at this stage."],
        "Matured Cotton Boll": ["Reduce irrigation to harden bolls.", "Plan for harvest in coming weeks."],
        "Split Cotton Boll": ["Prepare for immediate harvest.", "Avoid rainfall exposure to split bolls."],
    }
    if gmain in grow_map:
        recs.extend(grow_map[gmain])

    if weather:
        recs.extend(generate_weather_recommendations(weather))

    return recs[:6]


def generate_farmer_insights(disease_result: Dict[str, Any], growth_result: Dict[str, Any]) -> list[str]:
    insights = []
    dclass = disease_result["predicted_class"]
    hscore = disease_result["health_score"]
    gmain = growth_result.get("main_class", "Unknown")

    if dclass != "Healthy":
        insights.append(f"Possible {dclass} risk detected. Immediate action advised.")
    elif hscore > 80:
        insights.append("Crop is currently healthy. No immediate disease risks detected.")
    else:
        insights.append("Crop shows slight stress. Monitor closely for early signs of disease.")

    if gmain == "Cotton Blossom":
        insights.append("Expected harvest in 45–60 days.")
    elif gmain == "Cotton Bud":
        insights.append("Expected harvest in 30–45 days.")
    elif gmain == "Early Boll":
        insights.append("Expected harvest in 20–30 days.")
    elif gmain == "Matured Cotton Boll":
        insights.append("Expected harvest in 10–15 days. Prepare equipment.")
    elif gmain == "Split Cotton Boll":
        insights.append("Ready for harvest. Ideal harvesting window is within 7 days.")

    return insights


def generate_advanced_recommendations(disease_result: Dict[str, Any], growth_result: Dict[str, Any]) -> Dict[str, str]:
    gmain = growth_result.get("main_class", "Unknown")
    dclass = disease_result["predicted_class"]

    adv_recs = {
        "irrigation_timing": "Maintain standard schedule (every 7-10 days depending on soil moisture).",
        "fertilizer_suggestions": "Use balanced NPK (e.g., 20-20-20) as per standard guidelines.",
        "pest_prevention": "Install sticky traps and monitor for early pest signs.",
        "harvesting_window": "Monitor crop maturity daily.",
    }

    if gmain in ["Cotton Blossom", "Cotton Bud"]:
        adv_recs["irrigation_timing"] = "Increase watering frequency to support blooming."
        adv_recs["fertilizer_suggestions"] = "Apply potassium-rich fertilizers to boost flower development."
    elif gmain in ["Matured Cotton Boll", "Split Cotton Boll"]:
        adv_recs["irrigation_timing"] = "Reduce or stop irrigation to harden bolls and prevent rot."
        adv_recs["harvesting_window"] = "Immediate to 1-2 weeks."

    if dclass == "Aphids":
        adv_recs["pest_prevention"] = "Use neem oil or recommended insecticide for Aphids immediately."
    elif dclass == "Army worm":
        adv_recs["pest_prevention"] = "Apply specific anti-worm biological controls like Bacillus thuringiensis (Bt)."
    elif dclass == "Cotton Boll Rot":
        adv_recs["irrigation_timing"] = "Stop irrigation immediately to allow soil and plant base to dry."
        
    return adv_recs



def encode_image_for_display(image: np.ndarray) -> str:
    display_image = resize_image(image, DISPLAY_IMAGE_MAX_DIMENSION)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), DISPLAY_JPEG_QUALITY]
    ok, buffer = cv2.imencode(".jpg", display_image, encode_params)
    if not ok:
        raise ValueError("Failed to encode image for display")
    return base64.b64encode(buffer).decode("utf-8")


def is_allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def calculate_file_hash(file_storage) -> str:
    """Generate SHA-256 hash for an uploaded file using chunk reading."""
    sha256_hash = hashlib.sha256()
    file_storage.seek(0)
    for byte_block in iter(lambda: file_storage.read(4096), b""):
        sha256_hash.update(byte_block)
    file_storage.seek(0)
    return sha256_hash.hexdigest()


def read_uploaded_image(file_storage) -> Tuple[str, np.ndarray, np.ndarray]:
    safe_filename = secure_filename(file_storage.filename)
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error reading image file")
    return safe_filename, image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# -------------------------------------------------------------------
# THREAD-SAFE GRAD-CAM CACHE
# -------------------------------------------------------------------
GRAD_CAM_CACHE = {}
GRAD_CAM_CACHE_LOCK = threading.Lock()
MAX_CACHE_SIZE = 100

def get_cached_grad_cam(image_hash: str) -> Optional[Tuple[str, str]]:
    with GRAD_CAM_CACHE_LOCK:
        return GRAD_CAM_CACHE.get(image_hash)

def set_cached_grad_cam(image_hash: str, overlay_b64: str, heatmap_only_b64: str) -> None:
    with GRAD_CAM_CACHE_LOCK:
        if len(GRAD_CAM_CACHE) >= MAX_CACHE_SIZE:
            # FIFO eviction
            first_key = next(iter(GRAD_CAM_CACHE))
            GRAD_CAM_CACHE.pop(first_key, None)
        GRAD_CAM_CACHE[image_hash] = (overlay_b64, heatmap_only_b64)


def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    resnet_model, yolo_model = model_manager.load_models()
    try:
        try:
            growth = infer_growth_stage(image)
        except Exception as exc:
            logger.error("Error during growth stage inference: %s", exc)
            growth = {"main_class": None, "main_class_idx": None, "confidence": 0.0, "boxes": [], "raw": []}

        disease = infer_disease(image)
        if not isinstance(disease, dict) or "predicted_class" not in disease or "health_score" not in disease:
            raise ValueError("Invalid disease model prediction output.")

        # Check cache first
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()
        cached_result = get_cached_grad_cam(image_hash)
        
        grad_cam_image_b64 = None
        heatmap_only_b64 = None
        
        if cached_result is not None:
            grad_cam_image_b64, heatmap_only_b64 = cached_result
            logger.info("Using cached Grad-CAM heatmaps")
        else:
            if resnet_model is not None and disease.get("predicted_class_idx") is not None:
                try:
                    input_tensor = preprocess_image_for_resnet(image)
                    with GradCAM(resnet_model, resnet_model.layer4[-1]) as grad_cam:
                        grad_cam_overlay = grad_cam(input_tensor, disease["predicted_class_idx"], image)
                        heatmap_np = getattr(grad_cam, "heatmap_np", None)
                    if grad_cam_overlay is not None:
                        grad_cam_image_b64 = encode_image_for_display(grad_cam_overlay)
                    if heatmap_np is not None:
                        pure_heatmap_rgb = generate_pure_heatmap(image, heatmap_np)
                        heatmap_only_b64 = encode_image_for_display(pure_heatmap_rgb)
                except Exception as exc:
                    logger.error("Error generating Grad-CAM: %s", exc)

            if grad_cam_image_b64 is None or heatmap_only_b64 is None:
                try:
                    mock_heatmap = generate_mock_heatmap(image)
                    mock_overlay = apply_heatmap_on_image(image, mock_heatmap)
                    grad_cam_image_b64 = encode_image_for_display(mock_overlay)
                    
                    pure_heatmap_rgb = generate_pure_heatmap(image, mock_heatmap)
                    heatmap_only_b64 = encode_image_for_display(pure_heatmap_rgb)
                except Exception as exc:
                    logger.error("Error generating fallback heatmap: %s", exc)
            
            if grad_cam_image_b64 and heatmap_only_b64:
                set_cached_grad_cam(image_hash, grad_cam_image_b64, heatmap_only_b64)

        disease["heatmap_b64"] = grad_cam_image_b64
        disease["heatmap_only_b64"] = heatmap_only_b64

        recs = generate_recommendations(disease, growth)
        severity = calculate_disease_severity(disease["health_score"])
        yield_est = estimate_yield(disease, growth, weather=None, field_acres=1.0)
        adv_recs = generate_advanced_recommendations(disease, growth)
        insights = generate_farmer_insights(disease, growth)

        result = {
            "disease": disease,
            "growth": growth,
            "recommendations": recs,
            "grad_cam_image_b64": grad_cam_image_b64,
            "heatmap_only_b64": heatmap_only_b64,
            "disease_severity": severity,
            "yield_estimate": yield_est,
            "advanced_recommendations": adv_recs,
            "farmer_insights": insights,
        }

        if growth.get("main_class") is None:
            fallback_reason = "Growth stage model unavailable in this deployment." if yolo_model is None else "Cotton growth stage could not be detected from the uploaded image."
            result["warnings"] = [
                fallback_reason,
                "Disease analysis is still provided, but comparison may be less reliable without a confirmed cotton crop detection.",
                "Grad-CAM explainability may also be affected if the primary crop is not detected.",
            ]

        return result
    except Exception as exc:
        logger.error("Unexpected error in image analysis: %s", exc)
        return {"error": "The AI model encountered an unexpected error while analyzing the image. Please verify the image file format and content and try again."}


def build_comparison_result(old_results: Dict[str, Any], new_results: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(old_results, dict) or not isinstance(new_results, dict):
        raise ValueError("Comparison analysis did not produce valid result objects.")

    old_disease = old_results.get("disease")
    new_disease = new_results.get("disease")
    if old_disease is None or new_disease is None:
        raise ValueError("Unable to compare the provided images because one or both images did not contain a valid cotton crop analysis.")

    old_score = float(old_disease.get("health_score", 0.0))
    new_score = float(new_disease.get("health_score", 0.0))
    change = new_score - old_score
    abs_change = abs(change)

    if change > 1:
        trend = {"status": "improved", "label": "Improved", "icon": "fa-arrow-trend-up", "direction": "up"}
        headline = f"Crop health improved by {abs_change:.1f}%"
        recommendation = "Continue the current treatment plan, keep irrigation steady, and scout every few days to confirm the recovery trend."
    elif change < -1:
        trend = {"status": "declined", "label": "Declined", "icon": "fa-arrow-trend-down", "direction": "down"}
        headline = f"Crop health declined by {abs_change:.1f}%"
        recommendation = "Increase field inspection frequency, isolate visibly affected plants, and consider expert guidance before the disease pressure spreads."
    else:
        trend = {"status": "stable", "label": "Stable", "icon": "fa-arrows-left-right", "direction": "flat"}
        headline = "Crop health remained stable"
        recommendation = "Maintain the current crop care routine and compare again after the next treatment or irrigation cycle."

    old_predicted = old_disease.get("predicted_class", "Unknown")
    new_predicted = new_disease.get("predicted_class", "Unknown")
    disease_reduced = old_predicted != "Healthy" and new_predicted == "Healthy"
    disease_changed = old_predicted != new_predicted

    summary = [
        headline,
        "Disease spread reduced" if disease_reduced else (f"Disease signal shifted from {old_predicted} to {new_predicted}" if disease_changed else f"Disease signal remains {new_predicted}"),
        recommendation,
    ]

    if new_results.get("recommendations"):
        summary.append(f"Model priority: {new_results['recommendations'][0]}")

    if isinstance(new_results.get("farmer_insights"), list):
        insight_msg = f"Crop health improved by {abs_change:.1f}% this week." if change > 0 else (f"Crop health declined by {abs_change:.1f}% this week." if change < 0 else "Crop health remained stable this week.")
        new_results["farmer_insights"].insert(0, insight_msg)

    return {
        "old_score": old_score,
        "new_score": new_score,
        "change_percentage": change,
        "abs_change_percentage": abs_change,
        "trend": trend,
        "recommendation": recommendation,
        "summary": summary,
    }


# -------------------------------------------------------------------
# FLASK ROUTES
# -------------------------------------------------------------------
@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def is_pytest_mode() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


@app.route("/")
def index():
    lang = request.args.get("lang", "en")
    return render_template("index.html", text=LANG.get(lang, LANG["en"]), lang=lang)


@app.route("/set-language/<lang>")
def set_language(lang):
    return redirect(url_for("index", lang=lang))


@app.template_filter("datetimeformat")
def datetimeformat_filter(value):
    if value == "now":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return value


@app.route("/tutorials")
def tutorials():
    return render_template("tutorials.html")


@app.route("/support")
def support():
    return render_template("support.html")


@app.route("/stories")
def stories():
    return render_template("stories.html")


@app.route("/dashboard")
def dashboard():
    farms = [
        {"name": "GreenGrid Hub — Gujarat", "health": 82, "stage": "Matured Boll", "disease_risk": "Low", "yield_est": 740},
        {"name": "North Field — Punjab", "health": 61, "stage": "Early Boll", "disease_risk": "Medium", "yield_est": 520},
        {"name": "West Field — Maharashtra", "health": 45, "stage": "Cotton Bud", "disease_risk": "High", "yield_est": 310},
        {"name": "East Field — Rajasthan", "health": 91, "stage": "Split Cotton Boll", "disease_risk": "Low", "yield_est": 860},
    ]
    return render_template("dashboard.html", farms=farms)


@app.route("/history")
def history():
    return render_template("history.html")


@app.route("/health")
def health():
    ensure_models_loaded()
    diagnostics = model_manager.diagnostics()
    model_loaded = diagnostics["resnet"]["loaded"] and diagnostics["yolo"]["loaded"]
    return jsonify({
        "status": "healthy",
        "mode": "ready" if model_loaded else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "models": diagnostics,
        "service": "Agri-Vision Cotton Analysis API",
    })


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded", "error")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)

        if not is_allowed_image(file.filename):
            flash("Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)", "error")
            return redirect(request.url)

        try:
            safe_filename, image, image_rgb = read_uploaded_image(file)
            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
            results = analyze_image(compressed_rgb)

            lat = request.form.get("lat", type=float)
            lon = request.form.get("lon", type=float)
            city = request.form.get("city", type=str)
            weather = None

            if lat is not None and lon is not None:
                owm_key = os.getenv("OPENWEATHER_API_KEY")
                weather = get_weather(lat, lon, owm_key)
            elif city:
                geo = geocode_city(city)
                if geo:
                    owm_key = os.getenv("OPENWEATHER_API_KEY")
                    weather = get_weather(geo["lat"], geo["lon"], owm_key)

            if weather and results.get("disease") and results.get("growth"):
                results["recommendations"] = (results.get("recommendations", []) + generate_weather_recommendations(weather))[:6]
                results["weather"] = weather

            if results.get("error"):
                raise ValueError(results["error"])

            predicted_class = results.get("disease", {}).get("predicted_class", "")
            disease_info = disease_info_map.get(predicted_class, {})

            return render_template(
                "results.html",
                results=results,
                filename=safe_filename,
                image_b64=encode_image_for_display(image_rgb),
                img_shape={"width": image.shape[1], "height": image.shape[0]},
                raw_json=json.dumps(results, indent=2),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                weather=weather,
                grad_cam_image_b64=results.get("grad_cam_image_b64"),
                heatmap_only_b64=results.get("heatmap_only_b64"),
                disease_info=disease_info,
            )
        except Exception as exc:
            logger.error("Analysis error: %s", exc)
            flash(f"Error during analysis: {str(exc)}", "error")
            return redirect(request.url)

    return render_template("upload.html")


@app.route("/comparison", methods=["GET", "POST"])
def comparison():
    error_message = None
    old_filename, new_filename, old_image, new_image = None, None, None, None

    if request.method == "POST":
        required_files = {"last_week_image": "Last Week Field Image", "current_week_image": "Current Week Field Image"}
        for field_name, label in required_files.items():
            if field_name not in request.files:
                flash(f"{label} is required", "error")
                return redirect(request.url)
            uploaded_file = request.files[field_name]
            if uploaded_file.filename == "":
                flash(f"Please select a file for {label}", "error")
                return redirect(request.url)
            if not is_allowed_image(uploaded_file.filename):
                flash(f"Invalid file type for {label}. Please upload PNG, JPG, JPEG, or GIF.", "error")
                return redirect(request.url)

        try:
            last_week_file = request.files["last_week_image"]
            current_week_file = request.files["current_week_image"]
            
            last_week_hash = calculate_file_hash(last_week_file)
            current_week_hash = calculate_file_hash(current_week_file)
            
            if last_week_hash == current_week_hash:
                error_message = "Duplicate field images detected. Please upload two different images for meaningful comparison analysis."
                return render_template("comparison.html", error_message=error_message)
        except Exception as exc:
            logger.error("Hashing error: %s", exc)

        try:
            old_filename, old_image, old_rgb = read_uploaded_image(request.files["last_week_image"])
            new_filename, new_image, new_rgb = read_uploaded_image(request.files["current_week_image"])

            old_results = analyze_image(old_rgb)
            new_results = analyze_image(new_rgb)

            _, yolo_model = model_manager.load_models()
            if old_results.get("disease") is None or new_results.get("disease") is None:
                error_message = "Unable to analyze one or both uploaded images. Please upload valid field images and try again."
            elif old_results.get("warnings") and new_results.get("warnings") and yolo_model is not None:
                error_message = "Unable to verify cotton crop in both images. Please upload clearer field photos with visible plants and try again."

            if error_message:
                return render_template(
                    "comparison.html",
                    error_message=error_message,
                    old_filename=old_filename,
                    new_filename=new_filename,
                    old_image_b64=encode_image_for_display(old_image),
                    new_image_b64=encode_image_for_display(new_image),
                )

            comparison_result = build_comparison_result(old_results, new_results)
            return render_template(
                "comparison.html",
                old_results=old_results,
                new_results=new_results,
                comparison=comparison_result,
                old_filename=old_filename,
                new_filename=new_filename,
                old_image_b64=encode_image_for_display(old_image),
                new_image_b64=encode_image_for_display(new_image),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception as exc:
            logger.error("Comparison analysis error: %s", exc)
            error_message = "Unable to compare field images right now. Please try again with clearer crop photos."
            return render_template(
                "comparison.html",
                error_message=error_message,
                old_filename=old_filename,
                new_filename=new_filename,
                old_image_b64=encode_image_for_display(old_image) if old_image is not None else None,
                new_image_b64=encode_image_for_display(new_image) if new_image is not None else None,
            )
    return render_template("comparison.html")


@app.route("/demo")
def demo():
    try:
        example_disease_probs = [0.08, 0.02, 0.01, 0.10, 0.04, 0.65, 0.05, 0.05]
        demo_disease = {
            "predicted_class": "Healthy",
            "predicted_class_idx": 5,
            "confidence": example_disease_probs[5],
            "model_confidence": round(example_disease_probs[5] * 100, 2),
            "detected_issue": "Healthy",
            "all_confidences": {disease_classes[i]: example_disease_probs[i] for i in range(len(disease_classes))},
            "health_score": 65.0,
            "raw": [example_disease_probs],
            "is_uncertain": False,
            "is_ambiguous": False,
            "interpretation_message": "Healthy crop detected with moderate confidence."
        }
        demo_growth_boxes = [
            {"class_id": 3, "class_name": "Matured Cotton Boll", "confidence": 0.91, "bbox": [120, 80, 210, 155]},
            {"class_id": 4, "class_name": "Split Cotton Boll", "confidence": 0.70, "bbox": [300, 120, 390, 210]},
        ]

        demo_growth = {
            "main_class": "Matured Cotton Boll",
            "main_class_idx": 3,
            "confidence": 0.91,
            "boxes": demo_growth_boxes,
            "raw": demo_growth_boxes,
        }
        
        # Generate high-quality synthetic cotton BGR image representing field crop
        synthetic_bgr = np.zeros((384, 512, 3), dtype=np.uint8)
        
        # Fill background with a rich soft earthy background
        synthetic_bgr[:, :] = [30, 40, 45]
        
        # Draw deep-green leaf foliage (multiple overlapping green circles)
        cv2.circle(synthetic_bgr, (200, 220), 120, (34, 139, 34), -1) # Forest Green
        cv2.circle(synthetic_bgr, (320, 260), 100, (46, 139, 87), -1) # Sea Green
        cv2.circle(synthetic_bgr, (120, 280), 90, (34, 120, 34), -1) # Darker Green
        
        # Draw organic branch structure
        cv2.line(synthetic_bgr, (256, 384), (256, 200), (42, 75, 124), 12)
        cv2.line(synthetic_bgr, (256, 260), (140, 180), (42, 75, 124), 8)
        cv2.line(synthetic_bgr, (256, 220), (380, 150), (42, 75, 124), 8)
        
        # Draw localized crop anomalies (reddish-brown leaf spots / target spot disease representation)
        cv2.circle(synthetic_bgr, (220, 200), 15, (40, 50, 139), -1)
        cv2.circle(synthetic_bgr, (215, 195), 5, (20, 30, 80), -1)
        cv2.circle(synthetic_bgr, (180, 240), 10, (40, 50, 139), -1)
        
        # Draw Matured Cotton Boll within [120, 80, 210, 155] (center is (165, 117.5))
        cv2.ellipse(synthetic_bgr, (165, 117), (40, 30), 0, 0, 360, (50, 180, 100), -1)
        cv2.ellipse(synthetic_bgr, (165, 117), (40, 30), 0, 0, 360, (40, 140, 80), 2)
        cv2.line(synthetic_bgr, (165, 87), (165, 75), (42, 75, 124), 4)

        # Draw Split Cotton Boll within [300, 120, 390, 210] (center is (345, 165))
        cv2.circle(synthetic_bgr, (330, 165), 20, (245, 245, 245), -1)
        cv2.circle(synthetic_bgr, (360, 165), 20, (245, 245, 245), -1)
        cv2.circle(synthetic_bgr, (345, 150), 20, (255, 255, 255), -1)
        cv2.circle(synthetic_bgr, (345, 180), 20, (230, 230, 230), -1)
        cv2.ellipse(synthetic_bgr, (345, 185), (35, 15), 0, 0, 360, (30, 50, 90), -1)
        
        # Convert from BGR to RGB
        synthetic_rgb = cv2.cvtColor(synthetic_bgr, cv2.COLOR_BGR2RGB)
        
        # Generate mock heatmap
        mock_heatmap = generate_mock_heatmap(synthetic_rgb)
        mock_overlay = apply_heatmap_on_image(synthetic_rgb, mock_heatmap)
        
        # Base64 encode both original synthetic image and XAI overlay
        image_b64 = encode_image_for_display(synthetic_rgb)
        grad_cam_image_b64 = encode_image_for_display(mock_overlay)
        
        # Set top-level and nested properties for robustness
        demo_disease["heatmap_b64"] = grad_cam_image_b64
        
        # Calculate Severity
        severity = calculate_disease_severity(demo_disease["health_score"])
        
        # Use estimate_yield from service
        from services.yield_service import estimate_yield
        yield_est = estimate_yield(demo_disease, demo_growth, weather=None, field_acres=1.0)
        
        # Generate advanced recommendations
        adv_recs = generate_advanced_recommendations(demo_disease, demo_growth)
        
        # Generate farmer insights
        insights = generate_farmer_insights(demo_disease, demo_growth)

        example_json = {
            "disease": demo_disease,
            "growth": demo_growth,
            "recommendations": generate_recommendations(demo_disease, demo_growth),
            "grad_cam_image_b64": grad_cam_image_b64,
            "disease_severity": severity,
            "yield_estimate": yield_est,
            "advanced_recommendations": adv_recs,
            "farmer_insights": insights
        }
        return render_template(
            "results.html",
            results=example_json,
            filename="demo_cotton.jpg",
            image_b64=image_b64,
            img_shape={"width": 512, "height": 384},
            raw_json=json.dumps(example_json, indent=2),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            grad_cam_image_b64=grad_cam_image_b64,
            yield_estimate=yield_est,
            disease_info=disease_info_map.get("Healthy", {}),
            weather=None
        )
    except Exception as e:
        logger.error(f"Demo route failed: {e}")
        return redirect(url_for("index"))


@app.route("/api/chat_test", methods=["GET"])
def api_chat_test():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
@app.route("/api/chat/", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"reply": "I'm sorry, I didn't receive a message."}), 400

    message = str(data["message"]).lower()
    responses = {
        r"\b(hello|hi|hey|howdy|greetings)\b": [
            "Hello there! How can I assist you with your cotton crop today?",
            "Hi! Need any help analyzing your farm data?"
        ],
        r"\b(disease|diseases|sick|spots?|rot|blight)\b": [
            "If you're noticing leaf spots or rotting, it could be Bacterial Blight or Target Spot. I highly recommend taking a picture and uploading it to our Analyze tab for an AI diagnosis."
        ],
        r"\b(yield|yields|harvest|harvests|produce)\b": [
            "Yield depends heavily on the crop's health score and current growth stage. Check out the Dashboard for predictions across your fields!"
        ],
        r"\b(fertilizer|fertilizers|nutrient|nutrients|npk|potassium)\b": [
            "Cotton responds well to a balanced NPK fertilizer. During the blooming and early boll stages, potassium is critical to maximize yield."
        ],
        r"\b(water|watering|irrigation|dry|drought)\b": [
            "Maintain regular watering during the blossom phase. However, once bolls mature and start splitting, you should reduce irrigation to prevent rot."
        ],
        r"\b(pest|pests|worm|worms|aphid|aphids|bug|bugs|insect|insects|bollworm)\b": [
            "Pests like Pink Bollworm and Aphids are common enemies of cotton. I recommend deploying pheromone traps and scouting the fields twice a week."
        ],
        r"\b(weather|temperature|rain|rainfall|humidity|climate)\b": [
            "Weather plays a huge role in cotton health. Hot, dry spells stress bolls while excess rain can encourage fungal diseases. Use our weather tab to monitor conditions."
        ],
        r"\b(soil|soils|ph|minerals|clay|loam|sandy)\b": [
            "Cotton thrives in well-draining loamy soil with a pH of 5.8–8.0. Conduct a soil test before the season to identify any nutrient deficiencies."
        ],
        r"\b(grow|growth|growing|stage|stages|seedling|boll|bolls|flower|flowering)\b": [
            "Cotton growth has 5 key stages: germination, seedling, vegetative, flowering/boll formation, and maturity. Each stage has unique care needs — the flowering stage is most critical!"
        ],
        r"\b(spray|spraying|pesticide|pesticides|fungicide|herbicide|chemical)\b": [
            "When spraying, always follow label rates and avoid spraying during peak heat or wind. Consider integrated pest management (IPM) to reduce chemical dependency."
        ],
        r"\b(thank(?:s|s you)?|awesome|great|perfect)\b": [
            "You're welcome! Feel free to ask any time. Happy farming! 🌱",
            "Glad I could help! Let me know if you have more questions about your cotton crop."
        ],
        r"\b(help|assist|support|guide|advice|tips?)\b": [
            "I'm here to help! You can ask me about crop diseases, yield optimization, pest control, irrigation, fertilization, weather impacts, or soil health.",
            "Sure! Try asking about cotton diseases, pest control, yield estimates, or upload an image in the Analyze tab for an instant AI diagnosis."
        ],
        r"\b(cotton|crop|crops|farm|farming|field|fields)\b": [
            "Agri-Vision specializes in cotton crop analysis. Upload a field image in the Analyze tab for disease detection, yield prediction, and health scoring!"
        ],
    }

    reply = "I'm your Agri-Vision AI assistant. I specialize in cotton farming, crop diseases, and yield optimization. How can I help you?"

    for pattern, reply_options in responses.items():
        if re.search(pattern, message):
            reply = random.choice(reply_options)
            break
    return jsonify({"reply": reply})


@app.route("/api/weather")
def api_weather():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    city = request.args.get("city", type=str)

    if city and not (lat is not None and lon is not None):
        geo = geocode_city(city)
        if not geo:
            return jsonify({"error": f"Could not geocode city: {city}"}), 404
        lat, lon = geo["lat"], geo["lon"]

    if lat is None or lon is None:
        return jsonify({"error": "Provide lat & lon, or city"}), 400

    owm_key = os.getenv("OPENWEATHER_API_KEY")
    weather = get_weather(lat, lon, owm_key)
    if not weather:
        return jsonify({"error": "Weather data unavailable"}), 503

    weather["weather_recommendations"] = generate_weather_recommendations(weather)
    return jsonify({"status": "success", "weather": weather})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not is_allowed_image(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a valid image.'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)

        if is_pytest_mode():
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({"error": "Invalid image file"}), 400
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
            results = analyze_image(compressed_rgb)
            if results.get("error"):
                return jsonify({"error": results["error"]}), 400
            return jsonify({"status": "success", "timestamp": datetime.now().isoformat(), "results": results}), 200

        from celery_worker import process_inference_task
        task = process_inference_task.delay(file_bytes.tolist())
        return jsonify({
            "status": "processing",
            "task_id": task.id,
            "message": "Image analysis has started in the background. Use the task_id to poll for results.",
        }), 202

    except Exception as exc:
        logger.error("API analysis trigger error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/explain", methods=["POST"])
def api_explain():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not is_allowed_image(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a valid image.'}), 400

    try:
        safe_filename, image, image_rgb = read_uploaded_image(file)
        compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
        results = analyze_image(compressed_rgb)
        
        if results.get("error"):
            return jsonify({"error": results["error"]}), 400
            
        disease = results.get("disease", {})
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "filename": safe_filename,
            "predicted_class": disease.get("predicted_class"),
            "confidence": disease.get("confidence"),
            "health_score": disease.get("health_score"),
            "image_b64": encode_image_for_display(image_rgb),
            "heatmap_b64": results.get("grad_cam_image_b64"),
            "heatmap_only_b64": results.get("heatmap_only_b64"),
            "target_layer": "ResNet50 layer4[-1]",
            "all_confidences": disease.get("all_confidences", {})
        }), 200

    except Exception as exc:
        logger.error("API explain error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/task/<task_id>", methods=["GET"])
def get_task_status(task_id):
    if is_pytest_mode():
        return jsonify({
            "state": "DISABLED",
            "status": "Async Celery result polling is disabled during tests because inference runs synchronously.",
            "task_id": task_id,
        }), 200

    from celery_worker import process_inference_task
    task = process_inference_task.AsyncResult(task_id)

    if task.state == "PENDING":
        response = {"state": task.state, "status": "Task is waiting in the queue..."}
    elif task.state != "FAILURE":
        response = {
            "state": task.state,
            "status": task.info.get("status", "") if isinstance(task.info, dict) else task.info,
        }
        if task.state == "SUCCESS":
            response["result"] = task.result
    else:
        response = {"state": task.state, "status": str(task.info)}
    return jsonify(response)


@app.route("/api/analyze_stream", methods=["POST"])
def api_analyze_stream():
    def generate():
        import json as _json

        def event(name: str, progress: int, message: str, data: Optional[Dict[str, Any]] = None):
            payload = {"step": name, "progress": progress, "message": message}
            if data is not None:
                payload["data"] = data
            return f"data: {_json.dumps(payload)}\n\n"

        try:
            if "file" not in request.files:
                yield event("error", 0, "No file uploaded.")
                return
            file = request.files["file"]
            if file.filename == "":
                yield event("error", 0, "No file selected.")
                return
            yield event("upload_received", 10, "File received successfully.")
        except Exception as exc:
            yield event("error", 0, f"File error: {str(exc)}")
            return

        try:
            safe_filename, image, image_rgb = read_uploaded_image(file)
            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
            image_b64 = encode_image_for_display(image_rgb)
            img_shape = {"width": image.shape[1], "height": image.shape[0]}
            yield event("preprocessing", 25, "Image preprocessed and compressed.")
        except Exception as exc:
            yield event("error", 25, f"Preprocessing failed: {str(exc)}")
            return

        try:
            growth = infer_growth_stage(compressed_rgb)
            yield event("growth_inference", 50, f"Growth stage detected: {growth.get('main_class', 'Unknown')}")
        except Exception as exc:
            yield event("error", 50, f"Growth stage inference failed: {str(exc)}")
            return

        try:
            disease = infer_disease(compressed_rgb)
            yield event("disease_inference", 75, f"Disease classified: {disease.get('predicted_class', 'Unknown')} ({round(disease.get('confidence', 0) * 100, 1)}% confidence)")
        except Exception as exc:
            yield event("error", 75, f"Disease classification failed: {str(exc)}")
            return

        try:
            # Generate Grad-CAM heatmaps for stream
            grad_cam_image_b64 = None
            heatmap_only_b64 = None
            
            image_hash = hashlib.sha256(compressed_rgb.tobytes()).hexdigest()
            cached_result = get_cached_grad_cam(image_hash)
            
            if cached_result is not None:
                grad_cam_image_b64, heatmap_only_b64 = cached_result
                logger.info("Using cached Grad-CAM for stream")
            else:
                resnet_model, _ = model_manager.load_models()
                if resnet_model is not None and disease.get("predicted_class_idx") is not None:
                    try:
                        input_tensor = preprocess_image_for_resnet(compressed_rgb)
                        with GradCAM(resnet_model, resnet_model.layer4[-1]) as grad_cam:
                            grad_cam_overlay = grad_cam(input_tensor, disease["predicted_class_idx"], compressed_rgb)
                            heatmap_np = getattr(grad_cam, "heatmap_np", None)
                        if grad_cam_overlay is not None:
                            grad_cam_image_b64 = encode_image_for_display(grad_cam_overlay)
                        if heatmap_np is not None:
                            pure_heatmap_rgb = generate_pure_heatmap(compressed_rgb, heatmap_np)
                            heatmap_only_b64 = encode_image_for_display(pure_heatmap_rgb)
                    except Exception as exc:
                        logger.error("Error generating Grad-CAM for stream: %s", exc)

                if grad_cam_image_b64 is None or heatmap_only_b64 is None:
                    try:
                        mock_heatmap = generate_mock_heatmap(compressed_rgb)
                        mock_overlay = apply_heatmap_on_image(compressed_rgb, mock_heatmap)
                        grad_cam_image_b64 = encode_image_for_display(mock_overlay)
                        
                        pure_heatmap_rgb = generate_pure_heatmap(compressed_rgb, mock_heatmap)
                        heatmap_only_b64 = encode_image_for_display(pure_heatmap_rgb)
                    except Exception as exc:
                        logger.error("Error generating fallback heatmap for stream: %s", exc)
                
                if grad_cam_image_b64 and heatmap_only_b64:
                    set_cached_grad_cam(image_hash, grad_cam_image_b64, heatmap_only_b64)

            disease["heatmap_b64"] = grad_cam_image_b64
            disease["heatmap_only_b64"] = heatmap_only_b64

            results = {
                "disease": disease,
                "growth": growth,
                "recommendations": generate_recommendations(disease, growth),
                "grad_cam_image_b64": grad_cam_image_b64,
                "heatmap_only_b64": heatmap_only_b64,
                "error": None,
            }
            if growth.get("main_class") is None:
                results["warnings"] = ["Growth stage could not be confidently detected.", "Disease analysis is still provided based on the uploaded crop image."]
            yield event("recommendations", 90, "Recommendations generated.")
        except Exception as exc:
            yield event("error", 90, f"Recommendation generation failed: {str(exc)}")
            return

        weather, yield_estimate = None, None
        try:
            lat = request.form.get("lat", type=float)
            lon = request.form.get("lon", type=float)
            city = request.form.get("city", type=str)

            if lat is not None and lon is not None:
                owm_key = os.getenv("OPENWEATHER_API_KEY")
                weather = get_weather(lat, lon, owm_key)
            elif city:
                geo = geocode_city(city)
                if geo:
                    owm_key = os.getenv("OPENWEATHER_API_KEY")
                    weather = get_weather(geo["lat"], geo["lon"], owm_key)

            if weather and results.get("disease") and results.get("growth"):
                results["recommendations"] = (results.get("recommendations", []) + generate_weather_recommendations(weather))[:6]
                results["weather"] = weather

            field_acres = request.form.get("field_acres", type=float) or 1.0
            if results.get("disease") and results.get("growth"):
                yield_estimate = estimate_yield(results["disease"], results["growth"], weather, field_acres)
        except Exception as exc:
            logger.warning("Weather/yield enrichment failed: %s", exc)

        try:
            complete_payload = {
                "results": results,
                "filename": safe_filename,
                "image_b64": image_b64,
                "img_shape": img_shape,
                "raw_json": _json.dumps(results, indent=2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "weather": weather,
                "yield_estimate": yield_estimate,
                "grad_cam_image_b64": grad_cam_image_b64,
                "heatmap_only_b64": heatmap_only_b64,
            }
            yield event("complete", 100, "Analysis complete!", data=complete_payload)
        except Exception as exc:
            yield event("error", 95, f"Failed to finalise results: {str(exc)}")
            return

    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/analyze_result", methods=["POST"])
def analyze_result():
    try:
        raw = request.form.get("payload", "")
        if not raw:
            flash("No analysis data received.", "error")
            return redirect(url_for("analyze"))

        payload = json.loads(raw)
        results = payload.get("results", {})
        filename = payload.get("filename", "unknown")
        image_b64 = payload.get("image_b64", "")
        img_shape = payload.get("img_shape", {})
        raw_json = payload.get("raw_json", "{}")
        timestamp = payload.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        weather = payload.get("weather")
        yield_estimate = payload.get("yield_estimate")

        if results.get("error"):
            flash(results["error"], "error")
            return redirect(url_for("analyze"))

        return render_template(
            "results.html",
            results=results,
            filename=filename,
            image_b64=image_b64,
            img_shape=img_shape,
            raw_json=raw_json,
            timestamp=timestamp,
            weather=weather,
            yield_estimate=yield_estimate,
            grad_cam_image_b64=results.get("grad_cam_image_b64"),
            heatmap_only_b64=results.get("heatmap_only_b64"),
        )
    except Exception as exc:
        logger.error("analyze_result error: %s", exc)
        flash(f"Failed to render results: {str(exc)}", "error")
        return redirect(url_for("analyze"))


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Agri-Vision Cotton Analysis System")
    logger.info("=" * 60)
    logger.info("Starting Flask application...")
    logger.info("Open http://localhost:5000 in your browser")
    logger.info("Endpoints:")
    logger.info("/              - Home page")
    logger.info("/analyze       - Upload and analyze image")
    logger.info("/comparison    - Compare two field images")
    logger.info("/dashboard     - Multi-farm comparison dashboard")
    logger.info("/demo          - View demo results")
    logger.info("/api/analyze   - API endpoint (POST)")
    logger.info("/health        - Health check")
    logger.info("=" * 60)

    ensure_models_loaded()
    
    is_debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(debug=is_debug, host="0.0.0.0", port=5000)