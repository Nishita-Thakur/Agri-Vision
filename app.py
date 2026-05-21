"""
Agri-Vision Flask Application
Unified inference for disease classification (ResNet50) and growth stage prediction (YOLOv8)
"""

import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from flasgger import Swagger
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from services.weather_service import (
    generate_weather_recommendations,
    geocode_city,
    get_weather,
)
from torchvision import transforms
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Yahan se celery_worker ka import HATA DIYA HAI taaki circular import na ho!

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
swagger = Swagger(app)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    secret_key = "dev_secret_123"
app.secret_key = secret_key

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

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

resnet_model = None
yolo_model = None


def load_models():
    global resnet_model, yolo_model
    if resnet_model is None:
        try:
            resnet_model = torch.load(
                "models/cotton_crop_disease_classification/full_resnet50_model.pth",
                map_location=torch.device("cpu"),
            )
            logger.info("ResNet50 model loaded successfully")
        except Exception as e:
            logger.warning(f"ResNet50 model not found or failed to load: {e}")
            resnet_model = None
    if yolo_model is None:
        try:
            yolo_model = YOLO("models/cotton_crop_growth_stage_prediction/best.pt")
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.warning(f"YOLOv8 model not found or failed to load: {e}")
            yolo_model = None
    return resnet_model, yolo_model


def preprocess_image_for_resnet(image, target_size=(224, 224)):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
    )
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def infer_disease(image):
    if resnet_model:
        processed = preprocess_image_for_resnet(image)
        with torch.no_grad():
            output = resnet_model(processed)
            probs = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, 1)
        probs_np = probs.numpy()
        class_idx = int(prediction.item())
        healthy_idx = disease_classes.index("Healthy")
        health_score = float(probs_np[0][healthy_idx]) * 100
    else:
        probs_np = np.random.rand(1, len(disease_classes))
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
        class_idx = int(np.argmax(probs_np[0]))
        health_score = float(np.max(probs_np[0])) * 100

    disease_confidences = {
        disease_classes[i]: float(probs_np[0][i]) for i in range(len(disease_classes))
    }

    results = {
        "predicted_class": disease_classes[class_idx],
        "predicted_class_idx": class_idx,
        "confidence": float(probs_np[0][class_idx]),
        "all_confidences": disease_confidences,
        "health_score": health_score,
        "raw": probs_np.tolist(),
    }
    return results


def infer_growth_stage(image):
    result = {
        "main_class": None,
        "main_class_idx": None,
        "confidence": 0.0,
        "boxes": [],
        "raw": [],
    }
    if yolo_model:
        pil_image = Image.fromarray(image)
        yolo_results = yolo_model(pil_image)
        boxes = []
        for r in yolo_results:
            if hasattr(r, "boxes"):
                for b in r.boxes:
                    class_id = (
                        int(b.cls[0].item())
                        if hasattr(b.cls[0], "item")
                        else int(b.cls[0])
                    )
                    conf = (
                        float(b.conf[0].item())
                        if hasattr(b.conf[0], "item")
                        else float(b.conf[0])
                    )
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    boxes.append(
                        {
                            "class_id": class_id,
                            "class_name": growth_stage_classes[class_id]
                            if class_id < len(growth_stage_classes)
                            else str(class_id),
                            "confidence": conf,
                            "bbox": xyxy,
                        }
                    )
            else:
                continue
        if len(boxes):
            main = max(boxes, key=lambda x: x["confidence"])
            result.update(
                {
                    "main_class": main["class_name"],
                    "main_class_idx": main["class_id"],
                    "confidence": main["confidence"],
                }
            )
            result["boxes"] = boxes
        result["raw"] = boxes
    return result


def generate_recommendations(disease_result, growth_result, weather=None):
    recs = []
    dclass = disease_result["predicted_class"]
    instr_map = {
        "Aphids": [
            "Inspect leaves closely for clusters of small pests.",
            "Use recommended insecticides if infestation is severe.",
        ],
        "Army worm": [
            "Increase scouting frequency.",
            "Apply biological or suitable chemical controls early.",
        ],
        "Bacterial blight": [
            "Avoid overhead irrigation.",
            "Remove and destroy affected plant parts.",
        ],
        "Cotton Boll Rot": [
            "Improve field drainage, avoid stagnant water.",
            "Remove and destroy rotten bolls.",
        ],
        "Green Cotton Boll": [
            "Monitor bolls for signs of pests or disease.",
            "Maintain optimal nutrient regime.",
        ],
        "Healthy": [
            "Continue general crop monitoring.",
            "Maintain optimal fertilization and irrigation.",
        ],
        "Powdery mildew": [
            "Remove infected plant debris.",
            "Apply fungicide at recommended intervals.",
        ],
        "Target Spot": [
            "Monitor for spread, reduce leaf wetness.",
            "Apply suitable fungicide if required.",
        ],
    }
    recs.extend(instr_map.get(dclass, ["Practice general crop hygiene."]))

    if disease_result["health_score"] < 50:
        recs.append("Consult an agricultural expert urgently for low health score.")
    elif disease_result["health_score"] < 70:
        recs.append("Increase frequency of crop monitoring based on moderate health.")

    gmain = growth_result.get("main_class", None)
    grow_map = {
        "Cotton Blossom": [
            "Maintain regular watering during blossom phase.",
            "Scout for early flower pests.",
        ],
        "Cotton Bud": ["Ensure adequate phosphorus supply.", "Monitor for budworm."],
        "Early Boll": [
            "Start borer management as boll phase begins.",
            "Avoid excess nitrogen at this stage.",
        ],
        "Matured Cotton Boll": [
            "Reduce irrigation to harden bolls.",
            "Plan for harvest in coming weeks.",
        ],
        "Split Cotton Boll": [
            "Prepare for immediate harvest.",
            "Avoid rainfall exposure to split bolls.",
        ],
    }
    if gmain in grow_map:
        recs.extend(grow_map[gmain])

    if weather:
        weather_recs = generate_weather_recommendations(weather)
        recs.extend(weather_recs)

    return recs[:6]


def resize_image(image, max_dim=MAX_INFERENCE_DIMENSION):
    height, width = image.shape[:2]
    if max(height, width) <= max_dim:
        return image
    scale = max_dim / float(max(height, width))
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def calculate_disease_severity(health_score):
    return max(0.0, 100.0 - health_score)

def predict_yield(health_score, growth_stage, area_acres=1.0):
    base_yield = 700 
    health_factor = health_score / 100.0
    stage_factors = {
        "Cotton Blossom": 0.8,
        "Cotton Bud": 0.9,
        "Early Boll": 1.0,
        "Matured Cotton Boll": 1.1,
        "Split Cotton Boll": 1.0
    }
    g_factor = stage_factors.get(growth_stage, 0.9)
    estimated_yield = base_yield * health_factor * g_factor * area_acres
    confidence = min(95.0, 50.0 + (health_score * 0.4))
    
    return {
        "estimated_yield_kg_per_acre": round(estimated_yield, 2),
        "confidence_percentage": round(confidence, 2)
    }

def generate_farmer_insights(disease_result, growth_result):
    insights = []
    dclass = disease_result["predicted_class"]
    hscore = disease_result["health_score"]
    gmain = growth_result.get("main_class", "Unknown")
    
    if dclass != "Healthy":
        insights.append(f"Possible {dclass} risk detected. Immediate action advised.")
    else:
        if hscore > 80:
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

def generate_advanced_recommendations(disease_result, growth_result):
    gmain = growth_result.get("main_class", "Unknown")
    dclass = disease_result["predicted_class"]
    
    adv_recs = {
        "irrigation_timing": "Maintain standard schedule (every 7-10 days depending on soil moisture).",
        "fertilizer_suggestions": "Use balanced NPK (e.g., 20-20-20) as per standard guidelines.",
        "pest_prevention": "Install sticky traps and monitor for early pest signs.",
        "harvesting_window": "Monitor crop maturity daily."
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


def analyze_image(image):
    growth = infer_growth_stage(image)
    disease = infer_disease(image)
    recs = generate_recommendations(disease, growth)

    result = {
        "disease": disease,
        "growth": growth,
        "recommendations": recs,
        "disease_severity": severity,
        "yield_prediction": y_pred,
        "advanced_recommendations": adv_recs,
        "farmer_insights": insights
    }

    if growth["main_class"] is None:
        if yolo_model is None:
            fallback_reason = "Growth stage model unavailable in this deployment."
        else:
            fallback_reason = (
                "Cotton growth stage could not be detected from the uploaded image."
            )

        result["warnings"] = [
            fallback_reason,
            "Disease analysis is still provided, but comparison may be less reliable without a confirmed cotton crop detection.",
        ]

    return result


def encode_image_for_display(image):
    import base64

    display_image = resize_image(image, DISPLAY_IMAGE_MAX_DIMENSION)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), DISPLAY_JPEG_QUALITY]
    _, buffer = cv2.imencode(".jpg", display_image, encode_params)
    image_b64 = base64.b64encode(buffer).decode("utf-8")
    return image_b64


def is_allowed_image(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def read_uploaded_image(file_storage):
    safe_filename = secure_filename(file_storage.filename)
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error reading image file")
    return safe_filename, image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_comparison_result(old_results, new_results):
    if not isinstance(old_results, dict) or not isinstance(new_results, dict):
        raise ValueError("Comparison analysis did not produce valid result objects.")

    old_disease = old_results.get("disease")
    new_disease = new_results.get("disease")
    if old_disease is None or new_disease is None:
        raise ValueError(
            "Unable to compare the provided images because one or both images did not contain a valid cotton crop analysis."
        )

    old_score = float(old_disease.get("health_score", 0.0))
    new_score = float(new_disease.get("health_score", 0.0))
    change = new_score - old_score
    abs_change = abs(change)

    if change > 1:
        trend = {
            "status": "improved",
            "label": "Improved",
            "icon": "fa-arrow-trend-up",
            "direction": "up",
        }
        headline = f"Crop health improved by {abs_change:.1f}%"
        recommendation = "Continue the current treatment plan, keep irrigation steady, and scout every few days to confirm the recovery trend."
    elif change < -1:
        trend = {
            "status": "declined",
            "label": "Declined",
            "icon": "fa-arrow-trend-down",
            "direction": "down",
        }
        headline = f"Crop health declined by {abs_change:.1f}%"
        recommendation = "Increase field inspection frequency, isolate visibly affected plants, and consider expert guidance before the disease pressure spreads."
    else:
        trend = {
            "status": "stable",
            "label": "Stable",
            "icon": "fa-arrows-left-right",
            "direction": "flat",
        }
        headline = "Crop health remained stable"
        recommendation = "Maintain the current crop care routine and compare again after the next treatment or irrigation cycle."

    old_predicted = old_disease.get("predicted_class", "Unknown")
    new_predicted = new_disease.get("predicted_class", "Unknown")
    disease_reduced = old_predicted != "Healthy" and new_predicted == "Healthy"
    disease_changed = old_predicted != new_predicted

    summary = [
        headline,
        "Disease spread reduced"
        if disease_reduced
        else (
            f"Disease signal shifted from {old_predicted} to {new_predicted}"
            if disease_changed
            else f"Disease signal remains {new_predicted}"
        ),
        recommendation,
    ]

    if new_results.get("recommendations"):
        summary.append(f"Model priority: {new_results['recommendations'][0]}")
        
    # Append to farmer insights if present
    if new_results.get("farmer_insights") is not None:
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


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    lang = request.args.get("lang", "en")
    return render_template("index.html", text=LANG.get(lang, LANG["en"]), lang=lang)


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
            flash(
                "Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)",
                "error",
            )
            return redirect(request.url)
        try:
            safe_filename, image, image_rgb = read_uploaded_image(file)
            compressed_rgb = resize_image(image_rgb, MAX_INFERENCE_DIMENSION)
            image_b64 = encode_image_for_display(image)
            results = analyze_image(compressed_rgb)

            lat = request.form.get("lat", type=float)
            lon = request.form.get("lon", type=float)
            city = request.form.get("city", type=str)
            weather = None
            if lat and lon:
                owm_key = os.getenv("OPENWEATHER_API_KEY")
                weather = get_weather(lat, lon, owm_key)
            elif city:
                geo = geocode_city(city)
                if geo:
                    owm_key = os.getenv("OPENWEATHER_API_KEY")
                    weather = get_weather(geo["lat"], geo["lon"], owm_key)
            if weather and results.get("disease") and results.get("growth"):
                extra_recs = generate_weather_recommendations(weather)
                results["recommendations"] = (
                    results.get("recommendations", []) + extra_recs
                )[:6]
                results["weather"] = weather

            if results.get("error"):
                raise ValueError(results["error"])

            return render_template(
                "results.html",
                results=results,
                filename=safe_filename,
                image_b64=image_b64,
                img_shape={"width": image.shape[1], "height": image.shape[0]},
                raw_json=json.dumps(results, indent=2),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                weather=weather,
            )
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            flash(f"Error during analysis: {str(e)}", "error")
            return redirect(request.url)
    return render_template("upload.html")


@app.route("/comparison", methods=["GET", "POST"])
def comparison():
    error_message = None
    old_filename = None
    new_filename = None
    old_image = None
    new_image = None

    if request.method == "POST":
        required_files = {
            "last_week_image": "Last Week Field Image",
            "current_week_image": "Current Week Field Image",
        }

        for field_name, label in required_files.items():
            if field_name not in request.files:
                flash(f"{label} is required", "error")
                return redirect(request.url)
            uploaded_file = request.files[field_name]
            if uploaded_file.filename == "":
                flash(f"Please select a file for {label}", "error")
                return redirect(request.url)
            if not is_allowed_image(uploaded_file.filename):
                flash(
                    f"Invalid file type for {label}. Please upload PNG, JPG, JPEG, or GIF.",
                    "error",
                )
                return redirect(request.url)

        try:
            old_filename, old_image, old_rgb = read_uploaded_image(
                request.files["last_week_image"]
            )
            new_filename, new_image, new_rgb = read_uploaded_image(
                request.files["current_week_image"]
            )

            old_results = analyze_image(old_rgb)
            new_results = analyze_image(new_rgb)

            if old_results.get("disease") is None or new_results.get("disease") is None:
                error_message = "Unable to analyze one or both uploaded images. Please upload valid field images and try again."
            elif (
                old_results.get("warnings")
                and new_results.get("warnings")
                and yolo_model is not None
            ):
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
        except Exception as e:
            logger.error(f"Comparison analysis error: {e}")
            error_message = "Unable to compare field images right now. Please try again with clearer crop photos."
            return render_template(
                "comparison.html",
                error_message=error_message,
                old_filename=old_filename,
                new_filename=new_filename,
                old_image_b64=encode_image_for_display(old_image)
                if old_image is not None
                else None,
                new_image_b64=encode_image_for_display(new_image)
                if new_image is not None
                else None,
            )

    return render_template("comparison.html")


@app.route("/demo")
def demo():
    example_disease_probs = [0.08, 0.02, 0.01, 0.10, 0.04, 0.65, 0.05, 0.05]
    demo_disease = {
        "predicted_class": "Healthy",
        "predicted_class_idx": 5,
        "confidence": example_disease_probs[5],
        "all_confidences": {
            disease_classes[i]: example_disease_probs[i]
            for i in range(len(disease_classes))
        },
        "health_score": 65.0,
        "raw": [example_disease_probs],
    }
    demo_growth_boxes = [
        {
            "class_id": 3,
            "class_name": "Matured Cotton Boll",
            "confidence": 0.91,
            "bbox": [120, 80, 210, 155],
        },
        {
            "class_id": 4,
            "class_name": "Split Cotton Boll",
            "confidence": 0.70,
            "bbox": [300, 120, 390, 210],
        },
    ]
    demo_growth = {
        "main_class": "Matured Cotton Boll",
        "main_class_idx": 3,
        "confidence": 0.91,
        "boxes": demo_growth_boxes,
        "raw": demo_growth_boxes,
    }
    example_json = {
        "disease": demo_disease,
        "growth": demo_growth,
        "recommendations": generate_recommendations(demo_disease, demo_growth),
    }
    return render_template(
        "results.html",
        results=example_json,
        filename="demo_cotton.jpg",
        image_b64="",
        img_shape={"width": 512, "height": 384},
        raw_json=json.dumps(example_json, indent=2),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def is_pytest_mode():
    return "PYTEST_CURRENT_TEST" in os.environ


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Trigger analysis of a cotton crop image for disease and growth stage.

    During pytest runs, the route gracefully degrades to synchronous inference so
    CI does not need Redis/Celery. Outside pytest, it queues the work in Celery.
    ---
    tags:
      - API
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Upload the cotton crop image (PNG, JPG, JPEG, GIF) to be analyzed.
    responses:
      200:
        description: Synchronous analysis result returned during tests.
      202:
        description: Task accepted for async processing. Returns a task ID.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not is_allowed_image(file.filename):
        return jsonify(
            {"error": "Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)"}
        ), 400

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

            return jsonify(
                {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "results": results,
                }
            ), 200

        # Import Celery only when needed to avoid circular imports and to keep
        # pytest/CI from touching Redis when no result backend is available.
        from celery_worker import process_inference_task

        task = process_inference_task.delay(file_bytes.tolist())

        return jsonify(
            {
                "status": "processing",
                "task_id": task.id,
                "message": "Image analysis has started in the background. Use the task_id to poll for results.",
            }
        ), 202

    except Exception as e:
        logger.error(f"API analysis trigger error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/task/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """
    Check the status and retrieve results of an async analysis task.
    ---
    tags:
      - API
    parameters:
      - name: task_id
        in: path
        type: string
        required: true
        description: The task ID returned from /api/analyze
    responses:
      200:
        description: Task status and result (if completed)
    """
    if is_pytest_mode():
        return jsonify(
            {
                "state": "DISABLED",
                "status": "Async Celery result polling is disabled during tests because inference runs synchronously.",
                "task_id": task_id,
            }
        ), 200

    # Import Celery only when this endpoint needs the result backend.
    from celery_worker import process_inference_task

    task = process_inference_task.AsyncResult(task_id)

    if task.state == "PENDING":
        response = {"state": task.state, "status": "Task is waiting in the queue..."}
    elif task.state != "FAILURE":
        response = {
            "state": task.state,
            "status": task.info.get("status", "")
            if isinstance(task.info, dict)
            else task.info,
        }
        if task.state == "SUCCESS":
            response["result"] = task.result
    else:
        response = {"state": task.state, "status": str(task.info)}

    return jsonify(response)


import re
import random

# Enable CORS for all origins (helps with preflight OPTIONS requests)
from flask_cors import CORS
CORS(app)
@app.route("/api/chat_test", methods=["GET"])
def api_chat_test():
    return jsonify({"status": "ok"})

@app.route("/api/chat", methods=["POST"])
@app.route("/api/chat/", methods=["POST"])
def api_chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"reply": "I'm sorry, I didn't receive a message."}), 400
        
    message = data["message"].lower()
    
    # Simulated AI responses for agricultural queries
    responses = {
        r"\b(hello|hi|hey)\b": [
            "Hello there! How can I assist you with your cotton crop today?", 
            "Hi! Need any help analyzing your farm data?"
        ],
        r"\b(disease|sick|spots|rot|blight)\b": [
            "If you're noticing leaf spots or rotting, it could be Bacterial Blight or Target Spot. I highly recommend taking a picture and uploading it to our Analyze tab for an AI diagnosis.",
            "Diseases like Cotton Boll Rot can spread quickly. Upload a photo of the affected plant to get specific treatment recommendations!"
        ],
        r"\b(yield|harvest|produce)\b": [
            "Yield depends heavily on the crop's health score and current growth stage. Typically, a healthy acre yields 500-800 kg. Check out the Dashboard for predictions across your fields!",
            "For accurate yield predictions, upload a field image in the Analyze tab and I'll calculate it for you."
        ],
        r"\b(fertilizer|nutrient|npk|potassium)\b": [
            "Cotton responds well to a balanced NPK fertilizer. During the blooming and early boll stages, potassium is critical to maximize yield.",
            "Avoid excessive nitrogen late in the season, as it promotes leafy growth rather than boll development."
        ],
        r"\b(water|irrigation|dry)\b": [
            "Maintain regular watering during the blossom phase. However, once bolls mature and start splitting, you should reduce irrigation to prevent rot.",
            "Monitor soil moisture closely! Overwatering can be just as harmful as underwatering, leading to root rot."
        ],
        r"\b(pest|worm|aphid|bug)\b": [
            "Pests like Pink Bollworm and Aphids are common enemies of cotton. I recommend deploying pheromone traps and scouting the fields twice a week.",
            "If you suspect Aphids, check the underside of the leaves. Use neem oil for early control, or chemical insecticides if the infestation is severe."
        ],
    }
    
    reply = "I'm your Agri-Vision AI assistant. I specialize in cotton farming, crop diseases, and yield optimization. How can I help you?"
    
    for pattern, replies in responses.items():
        if re.search(pattern, message):
            reply = random.choice(replies)
            break
            
    return jsonify({"reply": reply})

@app.route("/health")
def health():
    """
    Check the health status of the API and models.
    ---
    tags:
      - API
    responses:
      200:
        description: Returns the health status of the application and AI models.
    """
    model_loaded = resnet_model is not None and yolo_model is not None
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model_loaded,
            "service": "Agri-Vision Cotton Analysis API",
        }
    )


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


# =========================================================
# Support Page Route
# Provides centralized help and support access
# for users interacting with the Agri-Vision platform
# =========================================================


@app.route("/support")
def support():
    return render_template("support.html")

@app.route('/stories')

@app.route("/stories")
def stories():
    return render_template("stories.html")


@app.route("/api/weather")
def api_weather():
    """
    Get current weather data for a location.
    ---
    tags:
      - API
    parameters:
      - name: lat
        in: query
        type: number
        required: false
      - name: lon
        in: query
        type: number
        required: false
      - name: city
        in: query
        type: string
        required: false
    responses:
      200:
        description: Weather data retrieved successfully
    """
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    city = request.args.get("city", type=str)

    if city and not (lat and lon):
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

    logger.info("/demo          - View demo results")
    logger.info("/api/analyze   - API endpoint (POST)")
    logger.info("/health        - Health check")
    logger.info("=" * 60)
    load_models()
    is_debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(debug=is_debug, host="0.0.0.0", port=5000)
