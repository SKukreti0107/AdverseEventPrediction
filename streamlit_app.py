import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import from other modules
sys.path.append(str(Path(".").resolve()))

# Import the AdverseEventPredictor class
from src.model.predict import AdverseEventPredictor

# Set page configuration
st.set_page_config(
    page_title="Adverse Drug Event Detection System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .severity-critical {
        color: #D32F2F;
        font-weight: bold;
    }
    .severity-near-critical {
        color: #FF9800;
        font-weight: bold;
    }
    .severity-needs-attention {
        color: #2196F3;
        font-weight: bold;
    }
    .severity-unknown {
        color: #757575;
    }
    .confidence-high {
        background-color: rgba(76, 175, 80, 0.2);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    .confidence-medium {
        background-color: rgba(255, 152, 0, 0.2);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    .confidence-low {
        background-color: rgba(244, 67, 54, 0.2);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    .stProgress .st-bo {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'predictor' not in st.session_state:
    st.session_state.predictor = None

if 'results' not in st.session_state:
    st.session_state.results = None

if 'conversation_text' not in st.session_state:
    st.session_state.conversation_text = ""

# Header
st.markdown('<h1 class="main-header">Adverse Drug Event Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect potential adverse drug events from medical conversations</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "This application detects potential adverse drug events from medical conversations "
        "by extracting medicine names and symptoms, then matching them with the FDA Adverse "
        "Event Reporting System (FAERS) dataset."
    )
    
    st.markdown("### How it works")
    st.markdown(
        "1. Enter a medical conversation in the text area\n"
        "2. The system extracts medicines and symptoms\n"
        "3. Extracted entities are matched with FAERS data\n"
        "4. Severity predictions are made for potential adverse events"
    )
    
    st.markdown("### Severity Levels")
    st.markdown(
        "- <span class='severity-critical'>Critical</span>: Death, Life-Threatening, Hospitalization\n"
        "- <span class='severity-near-critical'>Near-Critical</span>: Disability, Congenital Anomaly, Required Intervention\n"
        "- <span class='severity-needs-attention'>Needs Attention</span>: Other Serious Events",
        unsafe_allow_html=True
    )

# Main content area
tabs = st.tabs(["Input Conversation", "Analysis Results", "Visualizations"])

# Input tab
with tabs[0]:
    st.markdown("### Enter Medical Conversation")
    
    # Example button
    if st.button("Load Example Conversation"):
        example_conversation = """Patient: I've been taking Amlodipine 5mg for my blood pressure for a few weeks, but I've started getting really swollen ankles.
Doctor: I see. How long has the swelling been happening? Is it constant or does it come and go?
Patient: It's been getting worse over the last few days, especially after I've been standing for a while. It's mostly around my ankles and calves.
Doctor: Thank you for the details. Amlodipine can sometimes cause swelling in the legs or ankles. Are you experiencing any other symptoms, like dizziness or shortness of breath?
Patient: I've been feeling a little lightheaded when I stand up quickly, but I haven't had trouble breathing.
Doctor: It sounds like the swelling could be a side effect of Amlodipine, and the dizziness may be from the drop in blood pressure. I think we might want to try switching to a different medication, like ACE inhibitors or ARBs, which can help manage your blood pressure without the same swelling.
Patient: That sounds good. I'm also taking some aspirin for my heart condition and occasionally some ibuprofen for muscle pain.
Doctor: Got it. Aspirin is fine, and we'll make sure there are no issues with your new medication. Ibuprofen can sometimes affect kidney function, so it's something to be mindful of while we adjust your treatment plan. We'll keep monitoring your blood pressure and make sure we're on the right track. How's your heart doing otherwise?"""
        st.session_state.conversation_text = example_conversation
    
    # Text area for conversation input
    conversation_text = st.text_area(
        "Conversation Text", 
        value=st.session_state.conversation_text,
        height=300,
        help="Enter the medical conversation between patient and healthcare provider"
    )
    
    # Update session state
    st.session_state.conversation_text = conversation_text
    
    # Analyze button
    if st.button("Analyze Conversation"):
        if not conversation_text.strip():
            st.error("Please enter a conversation to analyze.")
        else:
            try:
                # Initialize progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Initialize predictor
                status_text.text("Initializing predictor...")
                if st.session_state.predictor is None:
                    st.session_state.predictor = AdverseEventPredictor()
                progress_bar.progress(25)
                
                # Step 2: Process conversation
                status_text.text("Processing conversation...")
                progress_bar.progress(50)
                
                # Step 3: Analyze conversation
                status_text.text("Analyzing for adverse events...")
                results = st.session_state.predictor.analyze_conversation(conversation_text)
                st.session_state.results = results
                progress_bar.progress(100)
                
                # Complete
                status_text.text("Analysis complete!")
                st.success("Conversation analyzed successfully! View results in the Analysis Results tab.")
                
                # Automatically switch to results tab
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
            except Exception as e:
                st.error(f"Error analyzing conversation: {str(e)}")

# Results tab
with tabs[1]:
    if st.session_state.results is not None:
        results = st.session_state.results
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Medicines Detected", results['summary']['medicine_count'])
        with col2:
            st.metric("Symptoms Detected", results['summary']['symptom_count'])
        with col3:
            st.metric("Potential Adverse Events", results['summary']['adverse_event_count'])
        
        # Extracted entities
        st.markdown("### Extracted Entities")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Medicines")
            if results['extracted_medicines']:
                for medicine in results['extracted_medicines']:
                    st.markdown(f"- {medicine}")
            else:
                st.info("No medicines detected in the conversation.")
        
        with col2:
            st.markdown("#### Symptoms")
            if results['extracted_symptoms']:
                for symptom in results['extracted_symptoms']:
                    st.markdown(f"- {symptom}")
            else:
                st.info("No symptoms detected in the conversation.")
        
        # Adverse events
        st.markdown("### Potential Adverse Events")
        if results['adverse_events']:
            for i, event in enumerate(results['adverse_events'], 1):
                with st.expander(f"Adverse Event #{i}: {event['medicine']} → {len(event['matched_symptoms'])} symptom(s)"):
                    # Medicine info
                    st.markdown(f"**Medicine:** {event['medicine']}")
                    st.markdown(f"**Matched to FAERS drug:** {event['matched_drug']} (confidence: {event['drug_match_confidence']:.2f})")
                    
                    # Severity
                    severity_class = f"severity-{event['severity'].lower().replace(' ', '-')}" if event['severity'] else "severity-unknown"
                    st.markdown(f"**FAERS Severity:** <span class='{severity_class}'>{event['severity']}</span>", unsafe_allow_html=True)
                    
                    # Matched symptoms
                    st.markdown("**Matched Symptoms:**")
                    for match in event['matched_symptoms']:
                        # Determine confidence class
                        confidence = match.get('prediction_confidence', 0)
                        if confidence >= 0.7:
                            confidence_class = "confidence-high"
                        elif confidence >= 0.4:
                            confidence_class = "confidence-medium"
                        else:
                            confidence_class = "confidence-low"
                        
                        # Determine severity class
                        severity = match.get('predicted_severity', 'Unknown')
                        severity_class = f"severity-{severity.lower().replace(' ', '-')}" if severity else "severity-unknown"
                        
                        # Display symptom match
                        st.markdown(
                            f"- **{match['symptom']}** (matched to *{match['matched_reaction']}*)\n"
                            f"  - Predicted Severity: <span class='{severity_class}'>{severity}</span> "
                            f"<span class='{confidence_class}'>{confidence:.2f}</span>",
                            unsafe_allow_html=True
                        )
        else:
            st.info("No potential adverse events detected.")
    else:
        st.info("Please analyze a conversation first to see results.")

# Visualizations tab
with tabs[2]:
    if st.session_state.results is not None:
        results = st.session_state.results
        
        if results['adverse_events']:
            st.markdown("### Visualizations")
            
            # Prepare data for visualizations
            medicines = []
            severities = []
            confidence_scores = []
            symptom_counts = []
            
            for event in results['adverse_events']:
                medicines.append(event['medicine'])
                severities.append(event['severity'])
                confidence_scores.append(event['drug_match_confidence'])
                symptom_counts.append(len(event['matched_symptoms']))
            
            # Create a DataFrame for easier plotting
            df = pd.DataFrame({
                'Medicine': medicines,
                'Severity': severities,
                'Confidence': confidence_scores,
                'Symptom Count': symptom_counts
            })
            
            # Visualization 1: Severity distribution
            st.markdown("#### Severity Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            severity_counts = df['Severity'].value_counts()
            colors = ['#D32F2F', '#FF9800', '#2196F3', '#757575']
            severity_counts.plot(kind='bar', ax=ax, color=colors[:len(severity_counts)])
            plt.title('Distribution of Adverse Event Severities')
            plt.xlabel('Severity Level')
            plt.ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Visualization 2: Confidence vs Symptom Count
            st.markdown("#### Confidence vs Symptom Count")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                df['Confidence'], 
                df['Symptom Count'],
                c=[0 if s == 'Critical' else 1 if s == 'Near-Critical' else 2 if s == 'Needs Attention' else 3 for s in df['Severity']],
                cmap='RdYlBu_r',
                s=100,
                alpha=0.7
            )
            
            # Add medicine labels
            for i, txt in enumerate(df['Medicine']):
                ax.annotate(txt, (df['Confidence'].iloc[i], df['Symptom Count'].iloc[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            plt.title('Confidence Score vs Number of Matched Symptoms')
            plt.xlabel('Confidence Score')
            plt.ylabel('Number of Matched Symptoms')
            plt.colorbar(scatter, label='Severity (0=Critical, 3=Unknown)')
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.info("No adverse events detected to visualize.")
    else:
        st.info("Please analyze a conversation first to see visualizations.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Adverse Drug Event Detection System | Powered by FAERS data"
    "</div>",
    unsafe_allow_html=True
)