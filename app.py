import streamlit as st
import pandas as pd
from PIL import Image
import requests
import base64
import io
from datetime import datetime
import json

# Import our helper modules
from image_processor import process_medical_image, get_image_features
from ai_connector import get_medgemma_response, analyze_image_with_ai

# Page configuration
st.set_page_config(
    page_title="MedAid Professional Assistant", 
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional header
st.title("🩺 MedAid Professional Assistant")
st.markdown("### AI-Powered Diagnostic Support for Healthcare Professionals")

# CRITICAL DISCLAIMERS - Always visible
st.error("""
⚠️ **PROFESSIONAL USE DISCLAIMER** ⚠️
- This is a DIAGNOSTIC ASSISTANCE TOOL ONLY
- NOT a replacement for professional medical judgment
- ALL recommendations must be verified by qualified medical professionals
- Final diagnosis and treatment decisions remain with the healthcare provider
- Not approved for emergency or critical care situations
- Use only as a supplementary consultation tool
""")

# Sidebar for professional information
with st.sidebar:
    st.header("Professional Information")
    
    # Healthcare professional verification
    provider_name = st.text_input("Healthcare Provider Name")
    license_number = st.text_input("Medical License Number")
    facility = st.text_input("Medical Facility")
    
    if not provider_name or not license_number:
        st.warning("Please enter your professional credentials")
        st.stop()
    
    st.success("✅ Professional Mode Activated")
    
    # Case information
    st.header("Case Information")
    patient_id = st.text_input("Patient ID (anonymized)")
    case_date = st.date_input("Date", datetime.now())

# Main interface tabs
tab1, tab2, tab3 = st.tabs(["📝 Clinical Input", "🖼️ Image Analysis", "🔍 AI Analysis"])

# Initialize session state for case data
if 'case_data' not in st.session_state:
    st.session_state.case_data = {
        'symptoms': '',
        'history': '',
        'images': [],
        'analysis_results': []
    }

# Tab 1: Clinical Input
with tab1:
    st.header("Clinical Information Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Chief Complaint & Symptoms")
        symptoms = st.text_area(
            "Patient's primary symptoms and complaints",
            placeholder="e.g., Patient presents with persistent cough, fever, and shortness of breath for 3 days...",
            height=150
        )
        
        st.subheader("Physical Examination Findings")
        physical_exam = st.text_area(
            "Relevant physical examination findings",
            placeholder="e.g., Temperature 38.5°C, lung auscultation reveals bilateral crackles...",
            height=100
        )
    
    with col2:
        st.subheader("Medical History")
        medical_history = st.text_area(
            "Relevant medical history, medications, allergies",
            placeholder="e.g., History of hypertension, currently on lisinopril, no known allergies...",
            height=150
        )
        
        st.subheader("Demographics & Risk Factors")
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        risk_factors = st.text_area(
            "Additional risk factors",
            placeholder="e.g., Smoking history, occupational exposures, travel history...",
            height=100
        )
    
    # Store clinical data
    if st.button("Save Clinical Information"):
        st.session_state.case_data.update({
            'symptoms': symptoms,
            'physical_exam': physical_exam,
            'medical_history': medical_history,
            'age': age,
            'gender': gender,
            'risk_factors': risk_factors
        })
        st.success("Clinical information saved!")

# Tab 2: Image Analysis
with tab2:
    st.header("Medical Image Upload & Analysis")
    
    # Image upload section
    uploaded_files = st.file_uploader(
        "Upload medical images (X-rays, skin photos, lab results, etc.)",
        type=['png', 'jpg', 'jpeg', 'dcm', 'tiff'],
        accept_multiple_files=True,
        help="Supported: PNG, JPG, JPEG, DICOM, TIFF"
    )
    
    if uploaded_files:
        st.subheader("Uploaded Images")
        
        # Display images in grid
        cols = st.columns(3)
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 3]:
                # Display image
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_container_width=True)
                
                # Image metadata
                st.caption(f"Size: {image.size}")
                st.caption(f"Format: {image.format}")
                
                # Image type classification
                image_type = st.selectbox(
                    f"Image type for {uploaded_file.name}",
                    ["Dermatology", "Radiology", "Pathology", "Ophthalmology", "Other"],
                    key=f"type_{idx}"
                )
                
                # Process image button
                if st.button(f"Analyze {uploaded_file.name}", key=f"analyze_{idx}"):
                    with st.spinner("Processing image..."):
                        # Process the medical image for basic features
                        analysis_result = process_medical_image(image, image_type)
                        
                        # Get AI vision analysis
                        ai_analysis = analyze_image_with_ai(image, image_type, st.session_state.case_data.get('symptoms', ''))
                        
                        st.success("Analysis complete!")
                        
                        # Display results in tabs
                        img_tab1, img_tab2 = st.tabs(["🔍 Image Features", "🤖 AI Analysis"])
                        
                        with img_tab1:
                            st.subheader("Technical Image Analysis")
                            st.json(analysis_result)
                        
                        with img_tab2:
                            st.subheader("AI Vision Analysis")
                            
                            # Status indicator
                            if ai_analysis.get('vision_ai_active'):
                                st.success("✅ AI Vision Analysis Active")
                            elif ai_analysis.get('demo_mode'):
                                st.info("ℹ️ Demo Mode - Connect API for full analysis")
                            
                            # Main content in expandable sections for better space usage
                            with st.expander("🔍 AI Findings", expanded=True):
                                findings = ai_analysis.get('ai_findings', [])
                                if findings:
                                    for i, finding in enumerate(findings, 1):
                                        st.markdown(f"**{i}.** {finding}")
                                else:
                                    st.markdown("*No specific findings available*")
                            
                            col_a, col_b = st.columns([1, 1])
                            
                            with col_a:
                                with st.expander("🏥 Clinical Observations", expanded=True):
                                    observations = ai_analysis.get('clinical_observations', [])
                                    if observations:
                                        for obs in observations:
                                            st.markdown(f"• {obs}")
                                    else:
                                        st.markdown("*No observations available*")
                            
                            with col_b:
                                with st.expander("💡 Recommendations", expanded=True):
                                    recommendations = ai_analysis.get('recommendations', [])
                                    if recommendations:
                                        for rec in recommendations:
                                            st.markdown(f"• {rec}")
                                    else:
                                        st.markdown("*No recommendations available*")
                            
                            # Confidence and additional info
                            st.markdown("---")
                            col_conf, col_note = st.columns([1, 1])
                            
                            with col_conf:
                                confidence = ai_analysis.get('confidence', 'Not specified')
                                st.metric("Confidence Level", confidence)
                            
                            with col_note:
                                if ai_analysis.get('note'):
                                    st.info(ai_analysis['note'])

# Tab 3: AI Analysis
with tab3:
    st.header("AI-Powered Diagnostic Analysis")
    
    if st.button("Generate AI Analysis", type="primary"):
        if not st.session_state.case_data.get('symptoms'):
            st.warning("Please enter clinical information in the first tab")
        else:
            with st.spinner("Generating AI analysis..."):
                # Combine all available data
                case_summary = f"""
                Patient Demographics: {st.session_state.case_data.get('age', 'N/A')} year old {st.session_state.case_data.get('gender', 'N/A')}
                
                Chief Complaint: {st.session_state.case_data.get('symptoms', '')}
                
                Physical Exam: {st.session_state.case_data.get('physical_exam', '')}
                
                Medical History: {st.session_state.case_data.get('medical_history', '')}
                
                Risk Factors: {st.session_state.case_data.get('risk_factors', '')}
                """
                
                # Get AI response
                ai_response = get_medgemma_response(case_summary)
                
                # Display results
                st.subheader("🤖 AI Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Possible Diagnoses")
                    # Display differential diagnoses
                    if ai_response.get('diagnoses'):
                        for i, diagnosis in enumerate(ai_response['diagnoses'], 1):
                            st.markdown(f"**{i}. {diagnosis['condition']}**")
                            st.markdown(f"Confidence: {diagnosis['confidence']}%")
                            st.markdown(f"Reasoning: {diagnosis['reasoning']}")
                            st.markdown("---")
                    elif ai_response.get('raw_response'):
                        st.markdown(ai_response['raw_response'])
                
                with col2:
                    st.markdown("### Recommended Actions")
                    if ai_response.get('recommendations'):
                        for rec in ai_response['recommendations']:
                            st.markdown(f"• {rec}")
                    
                    st.markdown("### Additional Tests Suggested")
                    if ai_response.get('additional_tests'):
                        for test in ai_response['additional_tests']:
                            st.markdown(f"• {test}")

# Professional footer with case export
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export Case Report"):
        # Generate PDF or structured report
        case_report = {
            'provider': provider_name,
            'facility': facility,
            'patient_id': patient_id,
            'date': str(case_date),
            'case_data': st.session_state.case_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert to JSON for download
        json_report = json.dumps(case_report, indent=2)
        st.download_button(
            label="Download Case Report (JSON)",
            data=json_report,
            file_name=f"case_report_{patient_id}_{case_date}.json",
            mime="application/json"
        )

with col2:
    if st.button("Clear Case Data"):
        st.session_state.case_data = {
            'symptoms': '',
            'history': '',
            'images': [],
            'analysis_results': []
        }
        st.success("Case data cleared")

with col3:
    st.markdown("**Version 1.0** | Professional Use Only") 
