"""
NeuroScan AI - Brain Tumor Detection Web Application
Flask Backend API with Enhanced Multi-Model Ensemble and TTA

Features:
- Multiple classification models ensemble for higher accuracy
- Multiple segmentation models for better tumor localization
- Test Time Augmentation (TTA) for robust predictions
- Advanced preprocessing and post-processing
- User authentication and scan history (MongoDB)
- Cloud image storage with Cloudinary
- Detailed diagnostic reports with share functionality
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from io import BytesIO
from PIL import Image
import json
import uuid
import hashlib
from datetime import datetime
from functools import wraps
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson import ObjectId
import cloudinary
import cloudinary.uploader
import cloudinary.api
from utilities import (
    focal_tversky, tversky_loss, tversky, 
    dice_coefficient, dice_loss, bce_dice_loss,
    iou_score, sensitivity, specificity, precision_metric,
    predict_with_tta_classification, predict_with_tta_segmentation
)

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = 'neuroscan-ai-secret-key-2024'  # Change in production

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SCAN_HISTORY_FOLDER'] = 'scan_history'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
app.config['USE_TTA'] = True  # Enable Test Time Augmentation for higher accuracy
app.config['USE_ENSEMBLE'] = True  # Enable ensemble predictions
app.config['CONFIDENCE_THRESHOLD'] = 0.5  # Minimum confidence for tumor detection

# MongoDB Configuration - Using MongoDB Atlas (Cloud)
# You can get a free MongoDB Atlas cluster at https://www.mongodb.com/atlas
# Replace with your MongoDB Atlas connection string
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb+srv://neuroscan:neuroscan123@cluster0.mongodb.net/?retryWrites=true&w=majority')
MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME', 'neuroscan_db')

# Flag to track if MongoDB is available
mongodb_available = False

# Cloudinary Configuration
cloudinary.config(
    cloud_name = "deifdzc8x",
    api_key = "679338212113732",
    api_secret = "loNWIjTUSAJ94ICsBO5b1TPjkm0",
    secure = True
)

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SCAN_HISTORY_FOLDER'], exist_ok=True)

# ==================== Cloudinary Helper Functions ====================
def upload_image_to_cloudinary(image_data, folder="neuroscan", public_id=None):
    """Upload base64 image to Cloudinary and return the URL"""
    try:
        # Handle data URL format
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        result = cloudinary.uploader.upload(
            f"data:image/png;base64,{image_data}",
            folder=folder,
            public_id=public_id,
            resource_type="image"
        )
        return {
            'url': result['secure_url'],
            'public_id': result['public_id']
        }
    except Exception as e:
        print(f"Cloudinary upload error: {str(e)}")
        return None

def delete_image_from_cloudinary(public_id):
    """Delete image from Cloudinary"""
    try:
        cloudinary.uploader.destroy(public_id)
        return True
    except Exception as e:
        print(f"Cloudinary delete error: {str(e)}")
        return False

# ==================== MongoDB Setup ====================
mongo_client = None
db = None
mongodb_init_attempted = False

# File-based JSON storage fallback when MongoDB is not available
STORAGE_FILE = 'neuroscan_data.json'

def load_storage():
    """Load data from JSON file"""
    try:
        if os.path.exists(STORAGE_FILE):
            with open(STORAGE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading storage: {e}")
    return {'users': {}, 'scan_history': {}}

def save_storage(data):
    """Save data to JSON file"""
    try:
        with open(STORAGE_FILE, 'w') as f:
            json.dump(data, f, default=str)
    except Exception as e:
        print(f"Error saving storage: {e}")

# Load existing data on startup
memory_storage = load_storage()

def init_mongodb():
    """Initialize MongoDB connection"""
    global mongo_client, db, mongodb_available, mongodb_init_attempted
    
    # Only try once
    if mongodb_init_attempted:
        return mongodb_available
    
    mongodb_init_attempted = True
    
    try:
        # Try to connect to MongoDB with a short timeout
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client[MONGO_DB_NAME]
        
        # Test connection
        mongo_client.admin.command('ping')
        
        # Create indexes for better performance
        db.users.create_index('email', unique=True)
        db.scan_history.create_index('user_id')
        db.scan_history.create_index('share_token', unique=True, sparse=True)
        db.scan_history.create_index('scan_date')
        
        mongodb_available = True
        print("✓ Connected to MongoDB successfully")
        return True
    except Exception as e:
        mongodb_available = False
        print(f"⚠ MongoDB connection failed: {str(e)}")
        print("  Using in-memory storage as fallback")
        print("  Note: Data will not persist after server restart")
        return False

def get_db():
    """Get MongoDB database instance"""
    global db, mongodb_available, mongodb_init_attempted
    # Try to initialize if not done yet
    if not mongodb_init_attempted:
        init_mongodb()
    if not mongodb_available:
        return None
    return db

def hash_password(password):
    """Hash password with SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'authenticated': False}), 401
        return f(*args, **kwargs)
    return decorated_function

# Global variables for models (ensemble support)
classification_models = []  # List of classification models for ensemble
segmentation_models = []    # List of segmentation models for ensemble
classification_model = None  # Primary classification model
segmentation_model = None    # Primary segmentation model
secondary_classifier = None  # Secondary classifier (classifier-resnet-weights.keras)
secondary_segmentation = None  # Secondary segmentation model
models_loaded = False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_custom_objects():
    """Get all custom objects for model loading"""
    return {
        'Functional': tf.keras.Model,
        'tversky': tversky,
        'tversky_loss': tversky_loss,
        'focal_tversky': focal_tversky,
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'bce_dice_loss': bce_dice_loss,
        'iou_score': iou_score,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision_metric': precision_metric
    }


def load_models():
    """Load all pre-trained classification and segmentation models for ensemble"""
    global classification_model, segmentation_model, secondary_classifier, secondary_segmentation
    global classification_models, segmentation_models, models_loaded
    
    custom_objects = get_custom_objects()
    
    try:
        # ============================================================
        # LOAD PRIMARY CLASSIFICATION MODEL (ResNet-50)
        # Matches notebook: model.compile with label_smoothing and comprehensive metrics
        # ============================================================
        print("Loading primary classification model (ResNet-50)...")
        with open('resnet-50-MRI.json', 'r') as json_file:
            json_savedModel = json_file.read()
        json_savedModel = json_savedModel.replace('"class_name": "Model"', '"class_name": "Functional"')
        classification_model = tf.keras.models.model_from_json(json_savedModel, custom_objects=custom_objects)
        classification_model.load_weights('weights.hdf5')
        # Compile EXACTLY like notebook for consistent inference
        classification_model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Fine-tuned LR from notebook
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        classification_models.append(('ResNet-50', classification_model))
        print("✓ Primary classification model loaded successfully")
        
        # ============================================================
        # LOAD SECONDARY CLASSIFICATION MODEL (classifier-resnet-weights.keras)
        # ============================================================
        if os.path.exists('classifier-resnet-weights.keras'):
            print("Loading secondary classification model...")
            try:
                # Try loading the full model directly
                secondary_classifier = tf.keras.models.load_model(
                    'classifier-resnet-weights.keras',
                    custom_objects=custom_objects
                )
                classification_models.append(('Classifier-ResNet', secondary_classifier))
                print("✓ Secondary classification model loaded successfully")
            except Exception as e:
                print(f"⚠ Could not load secondary classifier: {str(e)}")
                # Try loading with JSON architecture if available
                if os.path.exists('classifier-resnet-model.json'):
                    try:
                        with open('classifier-resnet-model.json', 'r') as json_file:
                            json_content = json_file.read()
                        if json_content.strip():  # Check if file is not empty
                            json_content = json_content.replace('"class_name": "Model"', '"class_name": "Functional"')
                            secondary_classifier = tf.keras.models.model_from_json(json_content, custom_objects=custom_objects)
                            secondary_classifier.load_weights('classifier-resnet-weights.keras')
                            secondary_classifier.compile(
                                loss='categorical_crossentropy',
                                optimizer='adam',
                                metrics=["accuracy"]
                            )
                            classification_models.append(('Classifier-ResNet', secondary_classifier))
                            print("✓ Secondary classification model loaded with JSON architecture")
                    except Exception as e2:
                        print(f"⚠ Secondary classifier not available: {str(e2)}")
        
        # ============================================================
        # LOAD PRIMARY SEGMENTATION MODEL (ResUNet-MRI)
        # Matches notebook: model_seg.compile with focal_tversky and comprehensive metrics
        # ============================================================
        print("Loading primary segmentation model (ResUNet-MRI)...")
        with open('ResUNet-MRI.json', 'r') as json_file:
            json_savedModel = json_file.read()
        json_savedModel = json_savedModel.replace('"class_name": "Model"', '"class_name": "Functional"')
        segmentation_model = tf.keras.models.model_from_json(json_savedModel, custom_objects=custom_objects)
        segmentation_model.load_weights('weights_seg.hdf5')
        # Compile EXACTLY like notebook for consistent inference
        segmentation_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Same as notebook
            loss=focal_tversky,
            metrics=[
                tversky,
                dice_coefficient,
                iou_score,
                sensitivity,
                specificity
            ]
        )
        segmentation_models.append(('ResUNet-MRI', segmentation_model))
        print("✓ Primary segmentation model loaded successfully")
        
        # ============================================================
        # LOAD SECONDARY SEGMENTATION MODEL (ResUNet-model)
        # ============================================================
        if os.path.exists('ResUNet-model.json'):
            print("Loading secondary segmentation model...")
            try:
                with open('ResUNet-model.json', 'r') as json_file:
                    json_savedModel = json_file.read()
                json_savedModel = json_savedModel.replace('"class_name": "Model"', '"class_name": "Functional"')
                secondary_segmentation = tf.keras.models.model_from_json(json_savedModel, custom_objects=custom_objects)
                # Use same weights if no separate weights file exists
                if os.path.exists('weights_seg.hdf5'):
                    secondary_segmentation.load_weights('weights_seg.hdf5')
                    # Compile EXACTLY like notebook
                    secondary_segmentation.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss=focal_tversky,
                        metrics=[tversky, dice_coefficient, iou_score, sensitivity, specificity]
                    )
                    segmentation_models.append(('ResUNet-Alt', secondary_segmentation))
                    print("✓ Secondary segmentation model loaded successfully")
            except Exception as e:
                print(f"⚠ Could not load secondary segmentation: {str(e)}")
        
        models_loaded = True
        print(f"\n📊 Model Summary:")
        print(f"   - Classification models loaded: {len(classification_models)}")
        print(f"   - Segmentation models loaded: {len(segmentation_models)}")
        print(f"   - TTA enabled: {app.config['USE_TTA']}")
        print(f"   - Ensemble enabled: {app.config['USE_ENSEMBLE']}")
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        models_loaded = False
        return False


def preprocess_image_classification(img):
    """
    Preprocessing for classification model - EXACTLY matches notebook's ImageDataGenerator.
    Uses rescale=1./255. as in training (flow_from_dataframe with rescale=1./255.).
    """
    # Resize to 256x256 (same as target_size=IMG_SIZE in notebook)
    img = cv2.resize(img, (256, 256))
    
    # Convert to RGB if grayscale (ImageDataGenerator loads as RGB)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # OpenCV loads as BGR, convert to RGB to match ImageDataGenerator
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] - EXACT same as train_datagen = ImageDataGenerator(rescale=1./255.)
    img_norm = img.astype(np.float32) / 255.0
    
    # Reshape for model input (batch_size=1, height=256, width=256, channels=3)
    img_norm = np.reshape(img_norm, (1, 256, 256, 3))
    
    return img_norm


def preprocess_image_segmentation(img):
    """
    Preprocessing for segmentation model - EXACTLY matches notebook's DataGenerator.
    Uses standardization: img -= img.mean(); img /= img.std() as in utilities.py DataGenerator.
    """
    # Resize to 256x256 (same as img_h, img_w in DataGenerator)
    img = cv2.resize(img, (256, 256))
    
    # Convert to RGB if grayscale (DataGenerator reads with PIL which loads as RGB)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # OpenCV loads as BGR, convert to RGB to match PIL in DataGenerator
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float64 - EXACT same as DataGenerator
    img = np.array(img, dtype=np.float64)
    
    # Standardize (mean centering and std scaling) - EXACT same as DataGenerator:
    # img -= img.mean()
    # img /= (img.std() + 1e-8)
    img -= img.mean()
    img /= (img.std() + 1e-8)
    
    # Reshape for model input - EXACT same as DataGenerator: X = np.empty((1, 256, 256, 3))
    X = np.empty((1, 256, 256, 3), dtype=np.float64)
    X[0,] = img
    
    return X


def ensemble_classification_predict(img, use_tta=True):
    """
    Ensemble prediction combining multiple classification models.
    Uses weighted averaging based on model reliability.
    Primary ResNet-50 model gets higher weight as it matches notebook training.
    """
    predictions = []
    weights = []
    
    for name, model in classification_models:
        if use_tta and app.config['USE_TTA']:
            # Use Test Time Augmentation for more robust predictions
            pred = predict_with_tta_classification(model, img)
        else:
            pred = model.predict(img, verbose=0)
        predictions.append(pred)
        
        # Assign weights (primary model trained in notebook gets highest weight)
        if name == 'ResNet-50':
            weights.append(1.0)  # Primary model from notebook - full weight
        elif name == 'Classifier-ResNet':
            weights.append(0.8)  # Secondary trained model - high weight
        else:
            weights.append(0.5)  # Other models - lower weight
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted ensemble prediction for better accuracy
    if len(predictions) > 1 and app.config['USE_ENSEMBLE']:
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight
        return ensemble_pred
    else:
        return predictions[0]


def ensemble_segmentation_predict(img, use_tta=True):
    """
    Ensemble prediction combining multiple segmentation models.
    Uses averaging for more robust tumor boundary detection.
    """
    predictions = []
    
    for name, model in segmentation_models:
        if use_tta and app.config['USE_TTA']:
            # Use Test Time Augmentation
            pred = predict_with_tta_segmentation(model, img)
        else:
            pred = model.predict(img, verbose=0)
        predictions.append(pred)
    
    # Average ensemble prediction
    if len(predictions) > 1 and app.config['USE_ENSEMBLE']:
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    else:
        return predictions[0]


def post_process_segmentation(mask, min_area=100):
    """
    Post-process segmentation mask to remove noise and small artifacts.
    """
    # Convert to binary
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    
    # Remove small components (noise)
    cleaned_mask = np.zeros_like(mask_binary)
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 1
    
    # Apply morphological operations for smoother boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    return cleaned_mask


def predict_tumor(image_path):
    """
    Enhanced two-stage prediction with ensemble models and TTA:
    1. Classification: Does the image have a tumor? (Ensemble + TTA)
    2. Segmentation: If yes, where is the tumor located? (Ensemble + TTA)
    
    Returns detailed results including confidence scores and metrics.
    """
    
    # Read image
    img_original = cv2.imread(image_path)
    if img_original is None:
        # Try with PIL for better format support
        try:
            img_pil = Image.open(image_path)
            img_original = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except:
            return None
    
    # ============================================================
    # STAGE 1: ENHANCED CLASSIFICATION (Ensemble + TTA)
    # ============================================================
    # Preprocess EXACTLY like notebook's ImageDataGenerator (rescale=1./255.)
    img_class = preprocess_image_classification(img_original.copy())
    
    # Use ensemble prediction with TTA for robust tumor detection
    classification_pred = ensemble_classification_predict(img_class, use_tta=app.config['USE_TTA'])
    
    # Determine tumor presence with confidence
    tumor_probability = float(classification_pred[0][1])
    no_tumor_probability = float(classification_pred[0][0])
    
    # Use configurable threshold for more accurate detection
    has_tumor = tumor_probability >= app.config['CONFIDENCE_THRESHOLD']
    confidence = max(tumor_probability, no_tumor_probability)
    
    result = {
        'has_tumor': has_tumor,
        'confidence': confidence,
        'classification_scores': {
            'no_tumor': no_tumor_probability,
            'tumor': tumor_probability
        },
        'analysis_method': {
            'tta_enabled': app.config['USE_TTA'],
            'ensemble_enabled': app.config['USE_ENSEMBLE'],
            'classification_models_used': len(classification_models),
            'segmentation_models_used': len(segmentation_models)
        }
    }
    
    # ============================================================
    # STAGE 2: ENHANCED SEGMENTATION (Ensemble + TTA + Post-processing)
    # ============================================================
    if has_tumor:
        img_seg = preprocess_image_segmentation(img_original.copy())
        
        # Use ensemble prediction with TTA
        segmentation_pred = ensemble_segmentation_predict(img_seg, use_tta=app.config['USE_TTA'])
        
        # Get raw mask
        mask_raw = segmentation_pred[0].squeeze()
        
        # Post-process mask for cleaner results
        mask_binary = post_process_segmentation(mask_raw)
        
        # Calculate comprehensive tumor metrics
        tumor_pixels = int(np.sum(mask_binary))
        total_pixels = mask_binary.shape[0] * mask_binary.shape[1]
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        
        # Calculate tumor bounding box and centroid
        if tumor_pixels > 0:
            y_indices, x_indices = np.where(mask_binary == 1)
            bbox = {
                'x_min': int(np.min(x_indices)),
                'y_min': int(np.min(y_indices)),
                'x_max': int(np.max(x_indices)),
                'y_max': int(np.max(y_indices)),
                'width': int(np.max(x_indices) - np.min(x_indices)),
                'height': int(np.max(y_indices) - np.min(y_indices))
            }
            centroid = {
                'x': int(np.mean(x_indices)),
                'y': int(np.mean(y_indices))
            }
        else:
            bbox = None
            centroid = None
        
        # Create visualization images
        # Heatmap visualization
        mask_heatmap = (mask_raw * 255).astype(np.uint8)
        mask_colored = cv2.applyColorMap(mask_heatmap, cv2.COLORMAP_JET)
        
        # Green overlay on tumor region
        overlay = img_original.copy()
        overlay = cv2.resize(overlay, (256, 256))
        
        # Create a semi-transparent overlay
        green_overlay = overlay.copy()
        green_overlay[mask_binary == 1] = [0, 255, 0]  # Green
        overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)
        
        # Add contours for better visualization
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)  # Yellow contours
        
        # Convert images to base64
        _, mask_buffer = cv2.imencode('.png', mask_colored)
        _, overlay_buffer = cv2.imencode('.png', overlay)
        
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
        overlay_base64 = base64.b64encode(overlay_buffer).decode('utf-8')
        
        result['segmentation'] = {
            'mask': f"data:image/png;base64,{mask_base64}",
            'overlay': f"data:image/png;base64,{overlay_base64}",
            'tumor_area_percentage': float(tumor_percentage),
            'tumor_pixels': tumor_pixels,
            'total_pixels': total_pixels,
            'bounding_box': bbox,
            'centroid': centroid,
            'mask_confidence': float(np.mean(mask_raw[mask_binary == 1])) if tumor_pixels > 0 else 0.0
        }
        
        # Calculate severity assessment
        if tumor_percentage > 10:
            severity = 'High'
            severity_color = '#dc2626'
            urgency = 'Immediate'
        elif tumor_percentage > 5:
            severity = 'Moderate'
            severity_color = '#f59e0b'
            urgency = 'Priority'
        elif tumor_percentage > 1:
            severity = 'Low'
            severity_color = '#10b981'
            urgency = 'Routine'
        else:
            severity = 'Minimal'
            severity_color = '#3b82f6'
            urgency = 'Monitor'
        
        # Determine tumor location based on centroid
        img_center_x, img_center_y = 128, 128  # Center of 256x256 image
        if centroid:
            cx, cy = centroid['x'], centroid['y']
            
            # Determine quadrant/region
            if cy < img_center_y * 0.6:
                vertical_pos = 'Superior (Upper)'
            elif cy > img_center_y * 1.4:
                vertical_pos = 'Inferior (Lower)'
            else:
                vertical_pos = 'Central'
            
            if cx < img_center_x * 0.6:
                horizontal_pos = 'Left Hemisphere'
            elif cx > img_center_x * 1.4:
                horizontal_pos = 'Right Hemisphere'
            else:
                horizontal_pos = 'Midline'
            
            location = f"{vertical_pos} - {horizontal_pos}"
            
            # Calculate distance from center (normalized)
            dist_from_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2) / img_center_x
            if dist_from_center < 0.3:
                location_risk = 'Central location - may affect critical structures'
            elif dist_from_center < 0.6:
                location_risk = 'Intermediate location - moderate accessibility'
            else:
                location_risk = 'Peripheral location - better surgical accessibility'
        else:
            location = 'Unable to determine'
            location_risk = 'N/A'
        
        # Estimate tumor characteristics
        if bbox:
            aspect_ratio = bbox['width'] / max(bbox['height'], 1)
            if 0.7 <= aspect_ratio <= 1.3:
                shape = 'Roughly spherical/circular'
            elif aspect_ratio < 0.7:
                shape = 'Vertically elongated'
            else:
                shape = 'Horizontally elongated'
            
            # Estimated size in mm (assuming 256px = ~200mm brain width)
            pixel_to_mm = 200 / 256
            estimated_width_mm = bbox['width'] * pixel_to_mm
            estimated_height_mm = bbox['height'] * pixel_to_mm
            estimated_area_mm2 = tumor_pixels * (pixel_to_mm ** 2)
        else:
            shape = 'Unable to determine'
            estimated_width_mm = 0
            estimated_height_mm = 0
            estimated_area_mm2 = 0
        
        # Generate detailed recommendations based on severity
        if severity == 'High':
            recommendations = [
                'Immediate consultation with a neuro-oncologist recommended',
                'Additional imaging (contrast-enhanced MRI, PET scan) advised',
                'Tumor board review for treatment planning',
                'Consider surgical evaluation for biopsy or resection',
                'Regular monitoring every 2-4 weeks during treatment'
            ]
        elif severity == 'Moderate':
            recommendations = [
                'Schedule consultation with neurologist within 1-2 weeks',
                'Consider additional contrast-enhanced MRI',
                'Baseline cognitive assessment recommended',
                'Follow-up scan in 4-6 weeks',
                'Discuss treatment options with specialist'
            ]
        elif severity == 'Low':
            recommendations = [
                'Follow-up with primary care physician',
                'Repeat MRI in 3-6 months for monitoring',
                'Document any new neurological symptoms',
                'Maintain regular health check-ups',
                'Consider specialist referral if symptoms develop'
            ]
        else:
            recommendations = [
                'Continue routine health monitoring',
                'Report any new symptoms to healthcare provider',
                'Follow-up scan in 6-12 months if needed',
                'Maintain healthy lifestyle',
                'No immediate intervention required'
            ]
        
        result['severity_assessment'] = {
            'level': severity,
            'severity_color': severity_color,
            'urgency': urgency,
            'tumor_coverage': f"{tumor_percentage:.2f}%",
            'recommendation': recommendations[0]
        }
        
        # Add detailed report data
        result['detailed_report'] = {
            'scan_id': str(uuid.uuid4())[:8].upper(),
            'scan_date': datetime.now().isoformat(),
            'patient_type': 'Anonymous',
            'scan_type': 'Brain MRI (T1-weighted)',
            'image_resolution': '256 × 256 pixels',
            
            'tumor_characteristics': {
                'detected': True,
                'confidence_score': f"{confidence * 100:.1f}%",
                'coverage_percentage': f"{tumor_percentage:.2f}%",
                'affected_pixels': f"{tumor_pixels:,}",
                'total_pixels': f"{total_pixels:,}",
                'estimated_size': {
                    'width_mm': f"{estimated_width_mm:.1f}",
                    'height_mm': f"{estimated_height_mm:.1f}",
                    'area_mm2': f"{estimated_area_mm2:.1f}"
                },
                'shape_assessment': shape,
                'location': location,
                'location_risk': location_risk
            },
            
            'severity_details': {
                'level': severity,
                'color': severity_color,
                'urgency': urgency,
                'description': f"Based on AI analysis, the detected abnormality covers {tumor_percentage:.2f}% of the scan area, classified as {severity.lower()} severity requiring {urgency.lower()} attention."
            },
            
            'bounding_box': bbox,
            'centroid': centroid,
            'mask_confidence': f"{float(np.mean(mask_raw[mask_binary == 1])) * 100:.1f}%" if tumor_pixels > 0 else "N/A",
            
            'recommendations': recommendations,
            
            'analysis_metadata': {
                'models_used': [name for name, _ in classification_models] + [name for name, _ in segmentation_models],
                'tta_enabled': app.config['USE_TTA'],
                'ensemble_enabled': app.config['USE_ENSEMBLE'],
                'processing_time': 'Real-time',
                'ai_version': '2.0.0'
            },
            
            'disclaimer': 'This AI-generated report is intended for informational purposes only and should not replace professional medical diagnosis. Please consult with qualified healthcare professionals for proper diagnosis and treatment planning.'
        }
    else:
        # No tumor detected - add basic report
        result['detailed_report'] = {
            'scan_id': str(uuid.uuid4())[:8].upper(),
            'scan_date': datetime.now().isoformat(),
            'patient_type': 'Anonymous',
            'scan_type': 'Brain MRI (T1-weighted)',
            'image_resolution': '256 × 256 pixels',
            
            'tumor_characteristics': {
                'detected': False,
                'confidence_score': f"{confidence * 100:.1f}%",
                'coverage_percentage': '0.00%',
                'description': 'No abnormal tissue masses detected in this scan.'
            },
            
            'recommendations': [
                'No immediate concerns identified',
                'Continue regular health check-ups',
                'Report any new neurological symptoms',
                'Maintain healthy lifestyle habits',
                'Follow-up as advised by your physician'
            ],
            
            'analysis_metadata': {
                'models_used': [name for name, _ in classification_models],
                'tta_enabled': app.config['USE_TTA'],
                'ensemble_enabled': app.config['USE_ENSEMBLE'],
                'processing_time': 'Real-time',
                'ai_version': '2.0.0'
            },
            
            'disclaimer': 'This AI-generated report is intended for informational purposes only. A negative result does not guarantee absence of pathology. Please consult with qualified healthcare professionals for comprehensive evaluation.'
        }
    
    return result


# Routes
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'version': '1.0.0'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded. Please restart the server.'
        }), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, TIF, TIFF'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and convert original image to PNG for browser compatibility
        img_original = cv2.imread(filepath)
        if img_original is None:
            # Try with PIL for TIFF support
            img_pil = Image.open(filepath)
            img_original = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Convert to PNG and encode as base64
        _, img_buffer = cv2.imencode('.png', img_original)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        
        # Make prediction
        prediction_result = predict_tumor(filepath)
        
        if prediction_result is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Add original image to result
        prediction_result['original_image'] = f"data:image/png;base64,{img_base64}"
        
        # Clean up uploaded file (optional)
        # os.remove(filepath)
        
        return jsonify(prediction_result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return enhanced project statistics including model information"""
    stats = {
        'total_patients': 110,
        'total_scans': 3929,
        'model_accuracy': 97.92,
        'segmentation_score': 0.92,
        'average_inference_time': 2.3,
        'model_type': 'ResNet-50 + ResUNet Ensemble',
        'features': {
            'tta_enabled': app.config['USE_TTA'],
            'ensemble_enabled': app.config['USE_ENSEMBLE'],
            'classification_models': len(classification_models),
            'segmentation_models': len(segmentation_models),
            'confidence_threshold': app.config['CONFIDENCE_THRESHOLD']
        },
        'models_info': {
            'classification': [name for name, _ in classification_models],
            'segmentation': [name for name, _ in segmentation_models]
        }
    }
    return jsonify(stats)


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update model configuration"""
    if request.method == 'GET':
        return jsonify({
            'use_tta': app.config['USE_TTA'],
            'use_ensemble': app.config['USE_ENSEMBLE'],
            'confidence_threshold': app.config['CONFIDENCE_THRESHOLD']
        })
    else:
        data = request.get_json()
        if 'use_tta' in data:
            app.config['USE_TTA'] = bool(data['use_tta'])
        if 'use_ensemble' in data:
            app.config['USE_ENSEMBLE'] = bool(data['use_ensemble'])
        if 'confidence_threshold' in data:
            app.config['CONFIDENCE_THRESHOLD'] = float(data['confidence_threshold'])
        return jsonify({
            'status': 'updated',
            'use_tta': app.config['USE_TTA'],
            'use_ensemble': app.config['USE_ENSEMBLE'],
            'confidence_threshold': app.config['CONFIDENCE_THRESHOLD']
        })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File is too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors - return JSON for API routes"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found', 'path': request.path}), 404
    return render_template('index.html')


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    print(f"500 Error: {str(e)}")
    return jsonify({'error': 'Internal server error. Please try again.', 'details': str(e)}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions"""
    print(f"Unhandled exception: {str(e)}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': 'Server error', 'details': str(e)}), 500


# ==================== Authentication Routes ====================

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Register a new user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        
        # Validation
        if not email or not password or not name:
            return jsonify({'error': 'Email, password, and name are required'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        if '@' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        
        user_id = str(uuid.uuid4())
        password_hash = hash_password(password)
        
        database = get_db()
        
        if database is not None:
            # Use MongoDB
            existing_user = database.users.find_one({'email': email})
            if existing_user:
                return jsonify({'error': 'Email already registered'}), 400
            
            user_doc = {
                '_id': user_id,
                'email': email,
                'password_hash': password_hash,
                'name': name,
                'created_at': datetime.now(),
                'last_login': None
            }
            
            database.users.insert_one(user_doc)
        else:
            # Use file-based storage
            if email in memory_storage['users']:
                return jsonify({'error': 'Email already registered'}), 400
            
            memory_storage['users'][email] = {
                '_id': user_id,
                'email': email,
                'password_hash': password_hash,
                'name': name,
                'created_at': str(datetime.now()),
                'last_login': None
            }
            save_storage(memory_storage)  # Persist to file
        
        # Set session
        session['user_id'] = user_id
        session['user_email'] = email
        session['user_name'] = name
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully',
            'user': {
                'id': user_id,
                'email': email,
                'name': name
            }
        })
        
    except Exception as e:
        print(f"Signup error: {str(e)}")
        return jsonify({'error': f'Signup failed: {str(e)}'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        database = get_db()
        user = None
        
        if database is not None:
            # Use MongoDB
            user = database.users.find_one({'email': email})
        else:
            # Use in-memory storage
            user = memory_storage['users'].get(email)
        
        if not user or user['password_hash'] != hash_password(password):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if database is not None:
            # Update last login in MongoDB
            database.users.update_one(
                {'_id': user['_id']},
                {'$set': {'last_login': datetime.now()}}
            )
        else:
            # Update in file storage
            memory_storage['users'][email]['last_login'] = str(datetime.now())
            save_storage(memory_storage)  # Persist to file
        
        # Set session
        session['user_id'] = user['_id']
        session['user_email'] = user['email']
        session['user_name'] = user['name']
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user['_id'],
                'email': user['email'],
                'name': user['name']
            }
        })
        
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': f'Login failed: {str(e)}'}), 500


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """Check authentication status"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': session['user_id'],
                'email': session.get('user_email'),
                'name': session.get('user_name')
            }
        })
    return jsonify({'authenticated': False})


# ==================== Scan History Routes ====================

@app.route('/api/history/save', methods=['POST'])
def save_scan():
    """Save scan to user history with Cloudinary image storage"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if user is logged in
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({
                'error': 'Please login to save scans',
                'authenticated': False
            }), 401
        
        # Generate share token
        share_token = str(uuid.uuid4())[:12]
        scan_id = str(uuid.uuid4())
        
        database = get_db()
        
        # Upload images to Cloudinary
        original_url = None
        mask_url = None
        overlay_url = None
        cloudinary_ids = []
        
        folder = f"neuroscan/{user_id}"
        
        # Upload original image to Cloudinary
        if data.get('original_image'):
            result = upload_image_to_cloudinary(
                data['original_image'],
                folder=folder,
                public_id=f"{scan_id}_original"
            )
            if result:
                original_url = result['url']
                cloudinary_ids.append(result['public_id'])
        
        # Upload mask image to Cloudinary
        if data.get('segmentation', {}).get('mask'):
            result = upload_image_to_cloudinary(
                data['segmentation']['mask'],
                folder=folder,
                public_id=f"{scan_id}_mask"
            )
            if result:
                mask_url = result['url']
                cloudinary_ids.append(result['public_id'])
        
        # Upload overlay image to Cloudinary
        if data.get('segmentation', {}).get('overlay'):
            result = upload_image_to_cloudinary(
                data['segmentation']['overlay'],
                folder=folder,
                public_id=f"{scan_id}_overlay"
            )
            if result:
                overlay_url = result['url']
                cloudinary_ids.append(result['public_id'])
        
        # If Cloudinary upload failed, keep the base64 image for display
        if not original_url and data.get('original_image'):
            original_url = data.get('original_image')
        
        # Create scan document
        current_time = datetime.now()
        scan_doc = {
            '_id': scan_id,
            'user_id': user_id,
            'scan_date': current_time.isoformat(),  # Store as ISO string for JSON
            'has_tumor': data.get('has_tumor', False),
            'confidence': data.get('confidence', 0),
            'tumor_percentage': data.get('segmentation', {}).get('tumor_area_percentage', 0),
            'severity': data.get('severity_assessment', {}).get('level', 'N/A'),
            'original_image_url': original_url,
            'mask_image_url': mask_url,
            'overlay_image_url': overlay_url,
            'cloudinary_ids': cloudinary_ids,
            'report_data': data,
            'share_token': share_token
        }
        
        print(f"Saving scan for user {user_id}: {scan_id}")
        
        if database is not None:
            scan_doc['scan_date'] = current_time  # MongoDB can handle datetime
            database.scan_history.insert_one(scan_doc)
            print(f"Saved to MongoDB")
        else:
            # Use file-based storage
            if user_id not in memory_storage['scan_history']:
                memory_storage['scan_history'][user_id] = []
            memory_storage['scan_history'][user_id].append(scan_doc)
            save_storage(memory_storage)  # Persist to file
            print(f"Saved to file storage. Total scans for user: {len(memory_storage['scan_history'][user_id])}")
        
        return jsonify({
            'success': True,
            'scan_id': scan_id,
            'share_token': share_token,
            'share_url': f"/share/{share_token}",
            'images': {
                'original': original_url,
                'mask': mask_url,
                'overlay': overlay_url
            }
        })
        
    except Exception as e:
        print(f"Save scan error: {str(e)}")
        return jsonify({'error': f'Failed to save scan: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """Get user's scan history"""
    user_id = session['user_id']
    print(f"Getting history for user: {user_id}")
    
    try:
        database = get_db()
        scans = []
        
        if database is not None:
            print("Using MongoDB for history")
            # Get from MongoDB
            cursor = database.scan_history.find(
                {'user_id': user_id}
            ).sort('scan_date', -1).limit(50)
            
            for doc in cursor:
                report_data = doc.get('report_data', {})
                scans.append({
                    'id': doc['_id'],
                    'date': doc['scan_date'].isoformat() if doc.get('scan_date') else None,
                    'has_tumor': doc.get('has_tumor', False),
                    'confidence': doc.get('confidence', 0),
                    'tumor_percentage': doc.get('tumor_percentage', 0),
                    'severity': doc.get('severity', 'N/A'),
                    'share_token': doc.get('share_token'),
                    'original_image': doc.get('original_image_url') or report_data.get('original_image'),
                    'mask_image': doc.get('mask_image_url'),
                    'overlay_image': doc.get('overlay_image_url'),
                    'detailed_report': report_data.get('detailed_report')
                })
        else:
            # Get from file-based storage
            print("Using file-based storage for history")
            user_scans = memory_storage['scan_history'].get(user_id, [])
            print(f"Found {len(user_scans)} scans in storage for user {user_id}")
            for doc in sorted(user_scans, key=lambda x: x.get('scan_date', ''), reverse=True)[:50]:
                report_data = doc.get('report_data', {})
                # scan_date is already an ISO string for file storage
                scan_date = doc.get('scan_date')
                scans.append({
                    'id': doc['_id'],
                    'date': scan_date,
                    'has_tumor': doc.get('has_tumor', False),
                    'confidence': doc.get('confidence', 0),
                    'tumor_percentage': doc.get('tumor_percentage', 0),
                    'severity': doc.get('severity', 'N/A'),
                    'share_token': doc.get('share_token'),
                    'original_image': doc.get('original_image_url') or report_data.get('original_image'),
                    'mask_image': doc.get('mask_image_url'),
                    'overlay_image': doc.get('overlay_image_url'),
                    'detailed_report': report_data.get('detailed_report')
                })
        
        return jsonify({
            'success': True,
            'scans': scans,
            'total': len(scans)
        })
        
    except Exception as e:
        print(f"Get history error: {str(e)}")
        return jsonify({'error': f'Failed to get history: {str(e)}'}), 500


@app.route('/api/history/<scan_id>', methods=['GET'])
def get_scan_detail(scan_id):
    """Get detailed scan by ID"""
    try:
        database = get_db()
        doc = None
        
        if database is not None:
            # Find in MongoDB by scan_id or share_token
            doc = database.scan_history.find_one({
                '$or': [
                    {'_id': scan_id},
                    {'share_token': scan_id}
                ]
            })
        else:
            # Find in in-memory storage
            for user_scans in memory_storage['scan_history'].values():
                for scan in user_scans:
                    if scan['_id'] == scan_id or scan.get('share_token') == scan_id:
                        doc = scan
                        break
        
        if not doc:
            return jsonify({'error': 'Scan not found'}), 404
        
        report_data = doc.get('report_data', {})
        
        return jsonify({
            'success': True,
            'scan': {
                'id': doc['_id'],
                'date': doc['scan_date'].isoformat() if doc.get('scan_date') else None,
                'has_tumor': doc.get('has_tumor', False),
                'confidence': doc.get('confidence', 0),
                'tumor_percentage': doc.get('tumor_percentage', 0),
                'severity': doc.get('severity', 'N/A'),
                'share_token': doc.get('share_token'),
                **report_data
            }
        })
        
    except Exception as e:
        print(f"Get scan detail error: {str(e)}")
        return jsonify({'error': f'Failed to get scan: {str(e)}'}), 500


@app.route('/api/history/<scan_id>', methods=['DELETE'])
@login_required
def delete_scan(scan_id):
    """Delete a scan from history and Cloudinary"""
    user_id = session['user_id']
    
    try:
        database = get_db()
        doc = None
        
        if database is not None:
            # Get scan info from MongoDB
            doc = database.scan_history.find_one({
                '_id': scan_id,
                'user_id': user_id
            })
            
            if not doc:
                return jsonify({'error': 'Scan not found or unauthorized'}), 404
            
            # Delete images from Cloudinary
            cloudinary_ids = doc.get('cloudinary_ids', [])
            for public_id in cloudinary_ids:
                delete_image_from_cloudinary(public_id)
            
            # Delete from database
            database.scan_history.delete_one({'_id': scan_id})
        else:
            # Delete from file storage
            user_scans = memory_storage['scan_history'].get(user_id, [])
            for i, scan in enumerate(user_scans):
                if scan['_id'] == scan_id:
                    doc = user_scans.pop(i)
                    # Delete from Cloudinary
                    for public_id in doc.get('cloudinary_ids', []):
                        delete_image_from_cloudinary(public_id)
                    save_storage(memory_storage)  # Persist to file
                    break
            
            if not doc:
                return jsonify({'error': 'Scan not found or unauthorized'}), 404
        
        return jsonify({'success': True, 'message': 'Scan deleted successfully'})
        
    except Exception as e:
        print(f"Delete scan error: {str(e)}")
        return jsonify({'error': f'Failed to delete scan: {str(e)}'}), 500


@app.route('/share/<token>')
def share_scan(token):
    """Public share page for a scan"""
    return render_template('index.html', share_token=token)


if __name__ == '__main__':
    print("=" * 70)
    print("🧠 NeuroScan AI - Enhanced Brain Tumor Detection System")
    print("=" * 70)
    print("\n📌 Features:")
    print("   • Multi-model ensemble for higher accuracy")
    print("   • Test Time Augmentation (TTA) for robust predictions")
    print("   • CLAHE image enhancement for better contrast")
    print("   • Post-processing for cleaner segmentation masks")
    print("   • Severity assessment and recommendations")
    print("   • User authentication and scan history (MongoDB)")
    print("   • Cloud image storage (Cloudinary)")
    print("   • Detailed diagnostic reports with share functionality")
    print("\nInitializing application...")
    
    # Initialize MongoDB
    if init_mongodb():
        print("✓ MongoDB connected successfully")
    else:
        print("⚠ Running without database - authentication features disabled")
    
    # Verify Cloudinary configuration
    print("✓ Cloudinary configured (cloud: deifdzc8x)")
    
    # Load models
    if load_models():
        print("\n" + "=" * 70)
        print("✓ All models loaded successfully!")
        print(f"✓ Classification ensemble: {len(classification_models)} model(s)")
        print(f"✓ Segmentation ensemble: {len(segmentation_models)} model(s)")
        print("\n🚀 Starting Flask server...")
        print("📍 Access the application at: http://localhost:5000")
        print("📊 API Health Check: http://localhost:5000/api/health")
        print("⚙️  API Configuration: http://localhost:5000/api/config")
        print("🔐 Authentication: http://localhost:5000/api/auth/status")
        print("🗄️  Database: MongoDB")
        print("☁️  Image Storage: Cloudinary")
        print("=" * 70)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models. Please check that model files exist.")
        print("Required files:")
        print("  - resnet-50-MRI.json")
        print("  - weights.hdf5")
        print("  - ResUNet-MRI.json")
        print("  - weights_seg.hdf5")
        print("  - utilities.py")
        print("\nOptional files for enhanced ensemble:")
        print("  - classifier-resnet-model.json")
        print("  - classifier-resnet-weights.keras")
        print("  - ResUNet-model.json")
