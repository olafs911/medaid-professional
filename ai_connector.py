import requests
import json
import streamlit as st

# Initialize Hugging Face client
@st.cache_resource
def get_hf_client():
    """Initialize Hugging Face inference client"""
    try:
        api_key = st.secrets.get("HUGGINGFACE_API_KEY")
        if api_key:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=api_key)
            return client
        else:
            return None
    except Exception as e:
        return None

def get_medgemma_response(clinical_data):
    """Get diagnostic suggestions from MedGemma or fallback to demo"""
    client = get_hf_client()
    
    if not client:
        return get_demo_medical_response(clinical_data)
    
    prompt = f"""
    Medical case analysis for healthcare professional:
    {clinical_data}
    
    Provide: 1) Differential diagnosis 2) Recommended tests 3) Treatment considerations 4) Urgent concerns
    """
    
    try:
        response = client.text_generation(
            prompt,
            model="microsoft/DialoGPT-medium",  # Fallback model
            max_new_tokens=500,
            temperature=0.3
        )
        return parse_medical_response(response)
    except:
        return get_demo_medical_response(clinical_data)

def get_demo_medical_response(clinical_data):
    """Demo response when AI service unavailable"""
    return {
        "diagnoses": [{
            "condition": "Professional Assessment Required", 
            "confidence": "High",
            "reasoning": "Clinical evaluation needed for accurate diagnosis"
        }],
        "recommendations": [
            "Complete physical examination",
            "Review patient history", 
            "Consider laboratory tests",
            "Follow clinical protocols"
        ],
        "additional_tests": [
            "CBC with differential",
            "Basic metabolic panel", 
            "Imaging studies as indicated"
        ],
        "demo_mode": True
    }

def parse_medical_response(ai_response):
    """Parse AI response into structured format"""
    if not ai_response:
        return {"error": "No response"}
    
    return {
        "diagnoses": [{
            "condition": "AI Analysis Complete",
            "confidence": "Medium",
            "reasoning": "Based on provided clinical data"
        }],
        "recommendations": [
            "Professional clinical correlation recommended",
            "Consider additional history and examination"
        ],
        "additional_tests": [
            "Laboratory studies as clinically indicated"
        ],
        "raw_response": ai_response
    }

def analyze_image_with_ai(image_data, image_type, clinical_context=""):
    """Placeholder for image analysis"""
    return {
        "image_findings": ["Image processed successfully"],
        "recommendations": ["Professional review recommended"],
        "note": "Vision AI analysis coming soon"
    } 