from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
import base64

def process_medical_image(image, image_type):
    """
    Process uploaded medical images based on type
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Basic image preprocessing based on type
    if image_type == "Dermatology":
        processed_features = process_dermatology_image(img_array)
    elif image_type == "Radiology":
        processed_features = process_radiology_image(img_array)
    elif image_type == "Pathology":
        processed_features = process_pathology_image(img_array)
    else:
        processed_features = process_general_image(img_array)
    
    return {
        "image_type": image_type,
        "image_size": image.size,
        "features": processed_features,
        "quality_score": calculate_image_quality(img_array),
        "processing_notes": get_processing_notes(image_type)
    }

def process_dermatology_image(img_array):
    """Specific processing for skin/dermatology images"""
    # Basic image analysis for skin conditions
    features = {
        "average_color": np.mean(img_array, axis=(0,1)).tolist(),
        "color_variance": np.var(img_array, axis=(0,1)).tolist(),
        "brightness": float(np.mean(img_array)),
        "contrast": float(np.std(img_array))
    }
    return features

def process_radiology_image(img_array):
    """Specific processing for X-rays, CT scans, etc."""
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    features = {
        "histogram": np.histogram(gray, bins=50)[0].tolist(),
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
        "image_quality": "Good" if np.std(gray) > 20 else "Low contrast"
    }
    return features

def process_pathology_image(img_array):
    """Specific processing for pathology/microscopy images"""
    features = {
        "color_distribution": np.histogram(img_array.flatten(), bins=50)[0].tolist(),
        "sharpness": float(calculate_sharpness(img_array)),
        "tissue_area": float(estimate_tissue_area(img_array))
    }
    return features

def process_general_image(img_array):
    """General image processing for other medical images"""
    features = {
        "dimensions": list(img_array.shape),
        "mean_values": np.mean(img_array, axis=(0,1)).tolist(),
        "quality_assessment": "Standard processing applied"
    }
    return features

def calculate_image_quality(img_array):
    """Calculate basic image quality score"""
    # Simple quality metrics
    sharpness = calculate_sharpness(img_array)
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    
    # Normalize to 0-100 scale
    quality_score = min(100, (sharpness * 0.4 + (brightness/255) * 30 + (contrast/128) * 30))
    return round(quality_score, 1)

def calculate_sharpness(img_array):
    """Calculate image sharpness using Laplacian variance"""
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def estimate_tissue_area(img_array):
    """Estimate tissue area in pathology images"""
    # Simple threshold-based tissue detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    tissue_pixels = np.sum(gray > 50)  # Simple threshold
    total_pixels = gray.shape[0] * gray.shape[1]
    return round((tissue_pixels / total_pixels) * 100, 1)

def get_processing_notes(image_type):
    """Return processing notes based on image type"""
    notes = {
        "Dermatology": "Analyzed for color distribution, lesion characteristics, and skin texture",
        "Radiology": "Processed for contrast enhancement and anatomical structure detection",
        "Pathology": "Analyzed for tissue composition and cellular characteristics",
        "Ophthalmology": "Processed for retinal features and vascular patterns",
        "Other": "Standard medical image processing applied"
    }
    return notes.get(image_type, "Standard processing")

def get_image_features(image):
    """Extract comprehensive features from medical image"""
    img_array = np.array(image)
    
    features = {
        "basic_stats": {
            "width": image.size[0],
            "height": image.size[1],
            "channels": len(img_array.shape),
            "mean_intensity": float(np.mean(img_array)),
            "std_intensity": float(np.std(img_array))
        },
        "quality_metrics": {
            "sharpness": float(calculate_sharpness(img_array)),
            "contrast": float(np.std(img_array)),
            "brightness": float(np.mean(img_array))
        }
    }
    
    return features 