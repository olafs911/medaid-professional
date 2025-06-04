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
from ai_connector import get_medgemma_response, analyze_image_with_ai, analyze_image

# Page configuration
st.set_page_config(
    page_title="MedAid Professional Assistant", 
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional header with better spacing
st.title("🩺 MedAid Professional Assistant")
st.markdown("### AI-Powered Diagnostic Support for Healthcare Professionals")
st.markdown("---")  # Add separator

# CRITICAL DISCLAIMERS - Always visible with better formatting
st.error("""
⚠️ **PROFESSIONAL USE DISCLAIMER** ⚠️
- This is a DIAGNOSTIC ASSISTANCE TOOL ONLY
- NOT a replacement for professional medical judgment
- ALL recommendations must be verified by qualified medical professionals
- Final diagnosis and treatment decisions remain with the healthcare provider
- Not approved for emergency or critical care situations
- Use only as a supplementary consultation tool
""")

st.markdown("---")  # Add separator after disclaimer

# Sidebar for professional information
with st.sidebar:
    st.header("👨‍⚕️ Professional Information")
    st.markdown("---")
    
    # Healthcare professional verification
    provider_name = st.text_input("Healthcare Provider Name", placeholder="Dr. John Smith")
    license_number = st.text_input("Medical License Number", placeholder="MD123456")
    facility = st.text_input("Medical Facility", placeholder="General Hospital")
    
    if not provider_name or not license_number:
        st.warning("⚠️ Please enter your professional credentials")
        st.stop()
    
    st.success("✅ Professional Mode Activated")
    st.markdown("---")
    
    # Case information with better organization
    st.header("📋 Case Information")
    patient_id = st.text_input("Patient ID (anonymized)", placeholder="CASE001")
    case_date = st.date_input("Date", datetime.now())
    
    # Add some helpful info in sidebar
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info("""
    **For Best Results:**
    • Be specific in symptom descriptions
    • Include relevant medical history
    • Upload clear, relevant images
    • Always correlate with clinical judgment
    """)

# Main interface tabs with better spacing
st.markdown("## 📊 Clinical Analysis Dashboard")
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
    st.header("📝 Clinical Information Input")
    st.markdown("Please provide detailed clinical information for accurate analysis.")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("🩺 Chief Complaint & Symptoms")
        symptoms = st.text_area(
            "Patient's primary symptoms and complaints",
            placeholder="e.g., Patient presents with persistent cough, fever, and shortness of breath for 3 days...",
            height=150,
            help="Be specific about onset, duration, severity, and associated symptoms"
        )
        
        st.subheader("🔍 Physical Examination Findings")
        physical_exam = st.text_area(
            "Relevant physical examination findings",
            placeholder="e.g., Temperature 38.5°C, lung auscultation reveals bilateral crackles...",
            height=120,
            help="Include vital signs and pertinent physical findings"
        )
    
    with col2:
        st.subheader("📋 Medical History")
        medical_history = st.text_area(
            "Relevant medical history, medications, allergies",
            placeholder="e.g., History of hypertension, currently on lisinopril, no known allergies...",
            height=150,
            help="Include chronic conditions, current medications, and allergies"
        )
        
        st.subheader("👤 Demographics & Risk Factors")
        
        # Better organized demographics
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
        with demo_col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        
        risk_factors = st.text_area(
            "Additional risk factors",
            placeholder="e.g., Smoking history, occupational exposures, travel history...",
            height=80,
            help="Include social history and relevant risk factors"
        )
    
    # Store clinical data with better feedback
    st.markdown("---")
    col_save, col_status = st.columns([1, 2])
    with col_save:
        if st.button("💾 Save Clinical Information", type="primary"):
            st.session_state.case_data.update({
                'symptoms': symptoms,
                'physical_exam': physical_exam,
                'medical_history': medical_history,
                'age': age,
                'gender': gender,
                'risk_factors': risk_factors
            })
            st.success("✅ Clinical information saved!")
    
    with col_status:
        if st.session_state.case_data.get('symptoms'):
            st.info("📊 Clinical data ready for AI analysis")
        else:
            st.warning("⏳ Enter clinical information above")

# Tab 2: Image Analysis
with tab2:
    st.header("🖼️ Medical Image Upload & Analysis")
    st.markdown("Upload medical images for AI-powered analysis.")
    
    # Medical AI Information Panel
    with st.expander("🔬 **Image Analysis Options & Capabilities**", expanded=False):
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown("### 🏆 **Vertex AI MedGemma 4B (BEST!):**")
            st.success("""
            **Google's Official Medical AI:**
            • ✅ REAL medical image analysis
            • ✅ Trained on radiology, pathology, dermatology
            • ✅ Production-grade Google infrastructure  
            • ✅ No local hardware needed
            • 💰 ~$0.002-0.02 per analysis
            
            **Setup:** Google Cloud account + Vertex AI API
            """)
        
        with col_info2:
            st.markdown("### 🆓 **Free Computer Vision:**")
            st.info("""
            **Hugging Face API (FREE):**
            • ✅ Real image analysis
            • ✅ Object detection + classification
            • ✅ Medical context correlation
            • ⚠️ General AI + medical knowledge
            
            **Setup:** Free Hugging Face API key
            """)
        
        with col_info3:
            st.markdown("### 📚 **Demo Mode:**")
            st.warning("""
            **Your Current Setup:**
            • ✅ Clinical text analysis
            • ✅ Medical knowledge responses
            • ❌ No image pixel analysis
            • 🎯 Educational/demo purposes
            
            **Upgrade:** Add API keys for real analysis
            """)
        
        st.markdown("---")
        st.markdown("### 🔑 **Quick Setup Guide:**")
        
        setup_option = st.radio(
            "What would you like to set up?",
            ["🏥 Vertex AI MedGemma 4B (Recommended)", "🆓 Free Computer Vision", "📖 Learn More"],
            horizontal=True
        )
        
        if setup_option == "🏥 Vertex AI MedGemma 4B (Recommended)":
            st.markdown("""
            **Step 1:** Create [Google Cloud account](https://cloud.google.com/) (free tier available)
            **Step 2:** Enable Vertex AI API in [Google Cloud Console](https://console.cloud.google.com/)
            **Step 3:** Add `GOOGLE_CLOUD_PROJECT_ID` to your Streamlit secrets
            **Step 4:** Install: `pip install google-cloud-aiplatform`
            
            **Benefits:** 
            - Official Google medical AI 
            - Highest accuracy for medical images
            - Professional-grade infrastructure
            - Pay-per-use pricing (~$0.002-0.02 per image)
            """)
        
        elif setup_option == "🆓 Free Computer Vision":
            st.markdown("""
            **Step 1:** Get free [Hugging Face API key](https://huggingface.co/settings/tokens)
            **Step 2:** Add `HUGGINGFACE_API_KEY` to Streamlit secrets  
            **Step 3:** Enjoy free computer vision analysis!
            
            **Benefits:**
            - Completely free
            - Real image analysis 
            - Good for general medical context
            - No usage limits
            """)
        
        elif setup_option == "📖 Learn More":
            st.markdown("""
            **Vertex AI vs Free Options:**
            
            | Feature | Vertex AI MedGemma | Free Computer Vision | Demo Mode |
            |---------|-------------------|---------------------|-----------|
            | Medical Training | ✅ Specialized | ⚠️ General | ❌ Text only |
            | Image Analysis | ✅ Medical AI | ✅ Basic vision | ❌ None |
            | Accuracy | 🏆 Highest | 📊 Good | 📚 Educational |
            | Cost | 💰 Pay per use | 🆓 Free | 🆓 Free |
            | Setup | ⚙️ Moderate | 🔧 Easy | ✅ Ready |
            """)
    
    st.markdown("---")
    
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
                        
                        # Get AI vision analysis - try real medical AI first
                        ai_analysis = analyze_image(image, image_type, st.session_state.case_data.get('symptoms', ''))
                        
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
                
                # Enhanced status indicators
                if ai_response.get('ai_analysis_active'):
                    st.success("✅ **AI Model Analysis Active** - Using " + ai_response.get('ai_model_used', 'medical AI'))
                elif ai_response.get('demo_mode'):
                    st.info("🧠 **Intelligent Medical Analysis** - Analyzing your clinical input with medical knowledge base")
                else:
                    st.warning("⚠️ **Basic Analysis Mode**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Possible Diagnoses")
                    # Display differential diagnoses
                    if ai_response.get('diagnoses'):
                        for i, diagnosis in enumerate(ai_response['diagnoses'], 1):
                            confidence_color = "🔴" if "high" in diagnosis.get('confidence', '').lower() else "🟡" if "medium" in diagnosis.get('confidence', '').lower() else "🟢"
                            st.markdown(f"**{i}. {diagnosis['condition']}**")
                            st.markdown(f"{confidence_color} Confidence: {diagnosis['confidence']}")
                            st.markdown(f"📋 Reasoning: {diagnosis['reasoning']}")
                            st.markdown("---")
                    elif ai_response.get('raw_response'):
                        st.markdown(ai_response['raw_response'])
                
                with col2:
                    st.markdown("### Recommended Actions")
                    if ai_response.get('recommendations'):
                        for i, rec in enumerate(ai_response['recommendations'], 1):
                            priority = "🚨" if any(word in rec.lower() for word in ['urgent', 'immediate', 'stat', 'emergency']) else "⚡"
                            st.markdown(f"{priority} **{i}.** {rec}")
                    
                    st.markdown("### Additional Tests Suggested")
                    if ai_response.get('additional_tests'):
                        for i, test in enumerate(ai_response['additional_tests'], 1):
                            st.markdown(f"🔬 **{i}.** {test}")
                
                # Show urgent concerns prominently if present
                if ai_response.get('urgent_concerns'):
                    st.markdown("---")
                    st.error("⚠️ **URGENT CLINICAL CONCERNS**")
                    for concern in ai_response['urgent_concerns']:
                        st.markdown(f"🚨 {concern}")
                
                # Add medical disclaimer for intelligent analysis
                if ai_response.get('demo_mode'):
                    st.markdown("---")
                    st.info("💡 **Note:** This analysis is based on clinical pattern recognition. Always correlate with physical examination and use clinical judgment.")

# Professional footer with case export
st.markdown("---")
st.markdown("## 📤 Case Management")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("💾 Export Report")
    if st.button("📄 Download Case Report", type="secondary"):
        # Generate PDF or structured report
        case_report = {
            'provider': provider_name,
            'facility': facility,
            'patient_id': patient_id,
            'date': str(case_date),
            'case_data': st.session_state.case_data,
            'timestamp': datetime.now().isoformat(),
            'app_version': '1.0'
        }
        
        # Convert to JSON for download
        json_report = json.dumps(case_report, indent=2)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_report,
            file_name=f"medaid_case_{patient_id}_{case_date}.json",
            mime="application/json"
        )

with col2:
    st.subheader("🗑️ Reset Case")
    if st.button("🔄 Clear All Data", type="secondary"):
        st.session_state.case_data = {
            'symptoms': '',
            'history': '',
            'images': [],
            'analysis_results': []
        }
        st.success("✅ Case data cleared")

with col3:
    st.subheader("ℹ️ App Info")
    st.markdown("**Version 1.0**")
    st.markdown("*Professional Use Only*")
    st.markdown("🏥 Healthcare Assistant")
    
    # Add helpful links
    with st.expander("📚 Resources"):
        st.markdown("""
        - [Medical Guidelines](https://example.com)
        - [Support Documentation](https://example.com) 
        - [Feature Requests](https://example.com)
        """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 12px;'>"
    "MedAid Professional Assistant • Built for Healthcare Professionals • "
    "Always verify with clinical judgment"
    "</div>", 
    unsafe_allow_html=True
) 
