import os

import cv2
import numpy as np
from celery import Celery

celery = Celery(
    "agri_vision_tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)


@celery.task(bind=True)
def process_inference_task(self, image_bytes_list, lat=None, lon=None, city=None):
    # Import inside task to fix circular import
    from app import analyze_image, enrich_results_with_weather

    file_bytes = np.array(image_bytes_list, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Invalid image"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = analyze_image(image_rgb)
    enrich_results_with_weather(results, lat=lat, lon=lon, city=city)
    return results
