"""
Agri-Vision Flask Application
Unified inference for disease classification (ResNet50) and growth stage prediction (YOLOv8)
With Model Versioning and A/B Testing Framework
"""
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import cv2
import numpy as np
from datetime import datetime
import torch
import logging
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import json
from jinja2 import Environment, FileSystemLoader
from model_registry import registry

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Keep Flask's own Jinja environment so template globals like url_for and get_flashed_messages remain available
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    secret_key = "dev_secret_123"
app.secret_key = secret_key

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

LANG = {
    "en": {
        "welcome": "Welcome to Agri Vision"
    },
    "te": {
        "welcome": "అగ్రి విజన్‌కు స్వాగతం"
    }
}

# Setup directories (safe repeat)
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- Class Names ---
# --- Disease class list (from confusion matrix order) ---
disease_classes = [
    "Aphids",             # 0
    "Army worm",          # 1
    "Bacterial blight",   # 2
    "Cotton Boll Rot",    # 3
    "Green Cotton Boll",  # 4
    "Healthy",            # 5
    "Powdery mildew",     # 6
    "Target Spot",        # 7
]
# --- Growth stage class list (from data.yaml for YOLOv8) ---
growth_stage_classes = [
    "Cotton Blossom",               # 0
    "Cotton Bud",                   # 1
    "Early Boll",                   # 2
    "Matured Cotton Boll",          # 3
    "Split Cotton Boll",            # 4
]

resnet_model = None
yolo_model = None

def load_models():
    """Load models using the model registry"""
    global resnet_model, yolo_model
    try:
        resnet_model = registry.load_model("resnet")
        yolo_model = registry.load_model("yolo")
        logger.info("Models loaded successfully via registry")
    except Exception as e:
        logger.warning(f"Models not found or failed to load: {e}")
        resnet_model = None
        yolo_model = None
    return resnet_model, yolo_model

def preprocess_image_for_resnet(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def infer_disease(image, model_version=None):
    # Returns all disease outputs, including confidences for each class
    import time
    start_time = time.time()
    
    # Load model from registry (supports version selection)
    try:
        model = registry.load_model("resnet", model_version)
        active_version = registry.get_active_model("resnet").version if model_version is None else model_version
    except:
        model = resnet_model
        active_version = "v1.0" if resnet_model else "fallback"
    
    if model:
        try:
            processed = preprocess_image_for_resnet(image)
            with torch.no_grad():
                output = model(processed)
                probs = F.softmax(output, dim=1)
                confidence, prediction = torch.max(probs, 1)
            probs_np = probs.numpy()  # shape: (1, 8)
            class_idx = int(prediction.item())
            healthy_idx = disease_classes.index("Healthy")  
            health_score = float(probs_np[0][healthy_idx]) * 100
            
            inference_time = time.time() - start_time
            
            # Track metrics in registry
            registry.update_metrics(
                "resnet", active_version, 
                float(probs_np[0][class_idx]), 
                inference_time, 
                success=True
            )
            
        except Exception as e:
            logger.error(f"Disease inference error: {e}")
            # Demo fallback
            probs_np = np.random.rand(1, len(disease_classes))
            probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
            class_idx = int(np.argmax(probs_np[0]))
            health_score = float(np.max(probs_np[0]))*100
            inference_time = time.time() - start_time
            
            # Track error in metrics
            if active_version != "fallback":
                registry.update_metrics("resnet", active_version, 0.0, inference_time, success=False)

    else:
        # Demo fallback
        probs_np = np.random.rand(1, len(disease_classes))
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
        class_idx = int(np.argmax(probs_np[0]))
        health_score = float(np.max(probs_np[0]))*100
        inference_time = time.time() - start_time

    # Format probabilities per class
    disease_confidences = {disease_classes[i]: float(probs_np[0][i]) for i in range(len(disease_classes))}

    results = {
        "predicted_class": disease_classes[class_idx],
        "predicted_class_idx": class_idx,
        "confidence": float(probs_np[0][class_idx]),
        "all_confidences": disease_confidences,
        "health_score": health_score,  # 0-100
        "raw": probs_np.tolist(),
        "model_version": active_version,
        "inference_time": inference_time
    }
    return results

def infer_growth_stage(image, model_version=None):
    import time
    start_time = time.time()
    
    result = {
        "main_class": None,
        "main_class_idx": None,
        "confidence": 0.0,
        "boxes": [],
        "raw": [],
    }
    
    # Load model from registry (supports version selection)
    try:
        model = registry.load_model("yolo", model_version)
        active_version = registry.get_active_model("yolo").version if model_version is None else model_version
    except:
        model = yolo_model
        active_version = "v1.0" if yolo_model else "fallback"
    
    if model:
        try:
            pil_image = Image.fromarray(image)
            yolo_results = model(pil_image)
            boxes = []
            for r in yolo_results:
                if hasattr(r, 'boxes'):
                    for b in r.boxes:
                        class_id = int(b.cls[0].item()) if hasattr(b.cls[0], 'item') else int(b.cls[0])
                        conf = float(b.conf[0].item()) if hasattr(b.conf[0], 'item') else float(b.conf[0])
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        boxes.append({
                            "class_id": class_id,
                            "class_name": growth_stage_classes[class_id] if class_id < len(growth_stage_classes) else str(class_id),
                            "confidence": conf,
                            "bbox": xyxy,  # [x1, y1, x2, y2]
                        })
                else:
                    continue
            # Most confident box as main prediction
            if len(boxes):
                main = max(boxes, key=lambda x: x['confidence'])
                result.update({
                    "main_class": main["class_name"],
                    "main_class_idx": main["class_id"],
                    "confidence": main["confidence"],
                })
                result["boxes"] = boxes
            result["raw"] = boxes
            
            inference_time = time.time() - start_time
            
            # Track metrics in registry
            avg_confidence = sum([b['confidence'] for b in boxes]) / len(boxes) if boxes else 0.0
            registry.update_metrics(
                "yolo", active_version,
                avg_confidence,
                inference_time,
                success=len(boxes) > 0
            )
            
        except Exception as e:
            logger.error(f"Growth stage inference error: {e}")
            inference_time = time.time() - start_time
            if active_version != "fallback":
                registry.update_metrics("yolo", active_version, 0.0, inference_time, success=False)
    
    result["model_version"] = active_version
    result["inference_time"] = inference_time if 'inference_time' in locals() else 0.0
    return result

def generate_recommendations(disease_result, growth_result):
    recs = []
    # Disease-based recommendations
    dclass, dscore = disease_result["predicted_class"], disease_result["confidence"]
    # Preset disease recommendations (feel free to expand for each class)
    instr_map = {
        "Aphids": [
            "Inspect leaves closely for clusters of small pests.",
            "Use recommended insecticides if infestation is severe."
        ],
        "Army worm": [
            "Increase scouting frequency.",
            "Apply biological or suitable chemical controls early."
        ],
        "Bacterial blight": [
            "Avoid overhead irrigation.",
            "Remove and destroy affected plant parts."
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
            "Maintain optimal fertilization and irrigation."
        ],
        "Powdery mildew": [
            "Remove infected plant debris.",
            "Apply fungicide at recommended intervals.",
        ],
        "Target Spot": [
            "Monitor for spread, reduce leaf wetness.",
            "Apply suitable fungicide if required.",
        ]
    }
    recs.extend(instr_map.get(dclass, ["Practice general crop hygiene."]))
    # Score-based adjustment
    if disease_result["health_score"] < 50:
        recs.append("Consult an agricultural expert urgently for low health score.")
    elif disease_result["health_score"] < 70:
        recs.append("Increase frequency of crop monitoring based on moderate health.")

    # Growth stage based recommendations
    gmain = growth_result.get("main_class", None)
    grow_map = {
        "Cotton Blossom": [
            "Maintain regular watering during blossom phase.",
            "Scout for early flower pests."
        ],
        "Cotton Bud": [
            "Ensure adequate phosphorus supply.",
            "Monitor for budworm."
        ],
        "Early Boll": [
            "Start borer management as boll phase begins.",
            "Avoid excess nitrogen at this stage."
        ],
        "Matured Cotton Boll": [
            "Reduce irrigation to harden bolls.",
            "Plan for harvest in coming weeks."
        ],
        "Split Cotton Boll": [
            "Prepare for immediate harvest.",
            "Avoid rainfall exposure to split bolls."
        ]
    }
    if gmain in grow_map:
        recs.extend(grow_map[gmain])
    # Recommend only top 5 relevant
    return recs[:5]
def analyze_image(image):

    # First detect cotton growth stage
    growth = infer_growth_stage(image)

    # Stop analysis if no cotton growth stage is detected
    if growth["main_class"] is None:
        return {
            "error": "No cotton plant detected",
            "disease": None,
            "growth": growth,
            "recommendations": [
                "Please upload a valid cotton crop image."
            ]
        }

    # Continue disease analysis only for cotton crops
    disease = infer_disease(image)

    # Generate recommendations
    recs = generate_recommendations(disease, growth)

    return {
        "disease": disease,
        "growth": growth,
        "recommendations": recs,
    }
# UTILITY: For image bounding box rendering in the frontend, also supply dimensions
def encode_image_for_display(image):
    import base64
    _, buffer = cv2.imencode('.jpg', image)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    return image_b64

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/")
def index():
    lang = request.args.get("lang", "en")
    return render_template(
        "index.html",
        text=LANG.get(lang, LANG["en"]),
        lang=lang
    )

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not '.' in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)', 'error')
            return redirect(request.url)
        try:
            safe_filename = secure_filename(file.filename)
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                flash('Error reading image file', 'error')
                return redirect(request.url)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_b64 = encode_image_for_display(image)
            results = analyze_image(image_rgb)
            # Render UI, pass bounding boxes for JS drawing, raw json, etc
            return render_template(
                "results.html",
                results=results,
                filename=safe_filename,
                image_b64=image_b64,
                img_shape={"width": image.shape[1], "height": image.shape[0]},
                raw_json=json.dumps(results, indent=2),
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            flash(f'Error during analysis: {str(e)}', 'error')
            return redirect(request.url)
    return render_template("upload.html")

@app.route("/demo")
def demo():
    # Generate demo outputs covering all class types
    example_disease_probs = [0.08, 0.02, 0.01, 0.10, 0.04, 0.65, 0.05, 0.05]
    demo_disease = {
        "predicted_class": "Healthy",
        "predicted_class_idx": 5,
        "confidence": example_disease_probs[5],
        "all_confidences": {disease_classes[i]: example_disease_probs[i] for i in range(len(disease_classes))},
        "health_score": 65.0,
        "raw": [example_disease_probs]
    }
    demo_growth_boxes = [
        {
            "class_id": 3,
            "class_name": "Matured Cotton Boll",
            "confidence": 0.91,
            "bbox": [120, 80, 210, 155]
        },
        {
            "class_id": 4,
            "class_name": "Split Cotton Boll",
            "confidence": 0.70,
            "bbox": [300, 120, 390, 210]
        }
    ]
    demo_growth = {
        "main_class": "Matured Cotton Boll",
        "main_class_idx": 3,
        "confidence": 0.91,
        "boxes": demo_growth_boxes,
        "raw": demo_growth_boxes
    }
    example_json = {
        "disease": demo_disease,
        "growth": demo_growth,
        "recommendations": generate_recommendations(demo_disease, demo_growth)
    }
    return render_template(
        "results.html",
        results=example_json,
        filename="demo_cotton.jpg",
        image_b64="",
        img_shape={"width": 512, "height": 384},
        raw_json=json.dumps(example_json, indent=2),
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

@app.route("/health")
def health():
    model_loaded = resnet_model is not None and yolo_model is not None
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'service': 'Agri-Vision Cotton Analysis API'
    })

@app.route("/set-language/<lang>")
def set_language(lang):
    return redirect(url_for("index", lang=lang))

@app.template_filter('datetimeformat')
def datetimeformat_filter(value):
    if value == "now":
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return value
@app.route('/tutorials')
def tutorials():
    return render_template('tutorials.html')

@app.route('/stories')
def stories():
    return render_template("stories.html")

@app.route('/model-admin')
def admin_dashboard():
    return render_template("admin.html")

# --- Model Management Admin Endpoints ---

@app.route('/admin/models', methods=['GET'])
def list_models():
    """List all registered models with their metadata"""
    model_type = request.args.get('type')
    try:
        models = registry.list_models(model_type)
        return jsonify({
            "status": "success",
            "models": models,
            "ab_test_enabled": registry.ab_test_enabled,
            "rollback_threshold": registry.rollback_threshold
        })
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/active', methods=['GET'])
def get_active_models():
    """Get currently active models"""
    try:
        active_resnet = registry.get_active_model("resnet")
        active_yolo = registry.get_active_model("yolo")
        return jsonify({
            "status": "success",
            "active_models": {
                "resnet": active_resnet.to_dict() if active_resnet else None,
                "yolo": active_yolo.to_dict() if active_yolo else None
            }
        })
    except Exception as e:
        logger.error(f"Error getting active models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/register', methods=['POST'])
def register_model():
    """Register a new model version"""
    try:
        data = request.get_json()
        required_fields = ['model_type', 'version', 'path']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        metadata = registry.register_model(
            model_type=data['model_type'],
            version=data['version'],
            path=data['path'],
            accuracy=data.get('accuracy', 0.0),
            dataset_version=data.get('dataset_version', 'unknown'),
            parameters=data.get('parameters', 0),
            is_active=data.get('is_active', False),
            ab_test_ratio=data.get('ab_test_ratio', 0.0)
        )
        return jsonify({
            "status": "success",
            "message": f"Model {data['model_type']} version {data['version']} registered successfully",
            "metadata": metadata.to_dict()
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/activate', methods=['POST'])
def activate_model():
    """Set a model version as active"""
    try:
        data = request.get_json()
        required_fields = ['model_type', 'version']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        registry.set_active_model(data['model_type'], data['version'])
        return jsonify({
            "status": "success",
            "message": f"Model {data['model_type']} version {data['version']} activated successfully"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error activating model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/delete', methods=['DELETE'])
def delete_model():
    """Delete a model version"""
    try:
        data = request.get_json()
        required_fields = ['model_type', 'version']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        registry.delete_model(data['model_type'], data['version'])
        return jsonify({
            "status": "success",
            "message": f"Model {data['model_type']} version {data['version']} deleted successfully"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/ab-testing', methods=['POST'])
def toggle_ab_testing():
    """Enable or disable A/B testing"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        registry.enable_ab_testing(enabled)
        return jsonify({
            "status": "success",
            "message": f"A/B testing {'enabled' if enabled else 'disabled'}",
            "ab_test_enabled": registry.ab_test_enabled
        })
    except Exception as e:
        logger.error(f"Error toggling A/B testing: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/ab-ratio', methods=['POST'])
def set_ab_ratio():
    """Set A/B testing ratio for a model version"""
    try:
        data = request.get_json()
        required_fields = ['model_type', 'version', 'ratio']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        registry.set_ab_test_ratio(data['model_type'], data['version'], data['ratio'])
        return jsonify({
            "status": "success",
            "message": f"A/B test ratio for {data['model_type']} version {data['version']} set to {data['ratio']}"
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error setting A/B ratio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/metrics', methods=['GET'])
def get_model_metrics():
    """Get performance metrics for all models"""
    try:
        models = registry.list_models()
        return jsonify({
            "status": "success",
            "metrics": models
        })
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/models/rollback-threshold', methods=['POST'])
def set_rollback_threshold():
    """Set automatic rollback threshold"""
    try:
        data = request.get_json()
        threshold = data.get('threshold')
        if threshold is None:
            return jsonify({"error": "Missing required field: threshold"}), 400
        
        if not 0.0 <= threshold <= 1.0:
            return jsonify({"error": "Threshold must be between 0.0 and 1.0"}), 400
        
        registry.rollback_threshold = threshold
        registry.save_config()
        return jsonify({
            "status": "success",
            "message": f"Rollback threshold set to {threshold}",
            "rollback_threshold": registry.rollback_threshold
        })
    except Exception as e:
        logger.error(f"Error setting rollback threshold: {e}")
        return jsonify({"error": str(e)}), 500

# --- Enhanced API Endpoints with Model Version Support ---

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get optional model version parameters
    resnet_version = request.form.get('resnet_version')
    yolo_version = request.form.get('yolo_version')
    request_id = request.form.get('request_id')  # For consistent A/B testing
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use A/B testing if enabled and request_id provided
        if registry.ab_test_enabled and request_id:
            resnet_model_version = None
            yolo_model_version = None
            # Registry will handle A/B routing internally
        else:
            resnet_model_version = resnet_version
            yolo_model_version = yolo_version
        
        # First detect cotton growth stage
        growth = infer_growth_stage(image_rgb, yolo_model_version)
        
        # Stop analysis if no cotton growth stage is detected
        if growth["main_class"] is None:
            return jsonify({
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "results": {
                    "error": "No cotton plant detected",
                    "disease": None,
                    "growth": growth,
                    "recommendations": ["Please upload a valid cotton crop image."]
                }
            })
        
        # Continue disease analysis only for cotton crops
        disease = infer_disease(image_rgb, resnet_model_version)
        
        # Generate recommendations
        recs = generate_recommendations(disease, growth)
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "results": {
                "disease": disease,
                "growth": growth,
                "recommendations": recs,
            }
        })
    except Exception as e:
        logger.error(f"API analysis error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Agri-Vision Cotton Analysis System")
    logger.info("=" * 60)
    logger.info("Starting Flask application...")
    logger.info("Open http://localhost:5000 in your browser")
    logger.info("Endpoints:")
    logger.info("/              - Home page")
    logger.info("/analyze       - Upload and analyze image")
    logger.info("/demo          - View demo results")
    logger.info("/api/analyze   - API endpoint (POST)")
    logger.info("/health        - Health check")
    logger.info("=" * 60)
    load_models()
    is_debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(debug=is_debug, host='0.0.0.0', port=5000)