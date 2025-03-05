"""Predict severity of adverse drug events.

This script uses the trained model to predict the severity of adverse drug events
based on extracted medicine names and symptoms from conversations.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import from other modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from extraction.medicine_extractor import MedicineExtractor
from extraction.symptom_extractor import SymptomExtractor
from matching.faers_matcher import FAERSMatcher

# Define paths
MODEL_DIR = Path("src/model")
PROCESSED_DATA_DIR = Path("data/processed")

class AdverseEventPredictor:
    """Class for predicting adverse event severity from conversations."""
    
    def __init__(self, model_path=None):
        """Initialize the predictor with the trained model.
        
        Args:
            model_path: Path to the trained model file
                       Default is None, which will use the default path
        """
        if model_path is None:
            model_path = MODEL_DIR / "severity_model.pkl"
        
        print(f"Initializing AdverseEventPredictor with model: {model_path}")
        try:
            # Load the trained model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
            
            # Initialize extractors and matcher
            self.medicine_extractor = MedicineExtractor()
            self.symptom_extractor = SymptomExtractor()
            self.faers_matcher = FAERSMatcher()
            
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            raise
    
    def process_conversation(self, conversation_text):
        """Process a conversation to extract medicines and symptoms.
        
        Args:
            conversation_text: The conversation transcript text
            
        Returns:
            Tuple of (medicines, symptoms)
        """
        print("Processing conversation...")
        
        # Extract medicines and symptoms from the conversation
        medicines = self.medicine_extractor.extract_medicines_from_conversation(conversation_text)
        symptoms = self.symptom_extractor.extract_symptoms_from_conversation(conversation_text)
        
        print(f"Extracted {len(medicines)} medicines and {len(symptoms)} symptoms")
        return medicines, symptoms
    
    def match_with_faers(self, medicines, symptoms):
        """Match extracted medicines and symptoms with FAERS data.
        
        Args:
            medicines: List of extracted medicine names
            symptoms: List of extracted symptoms
            
        Returns:
            List of detected adverse events
        """
        print("Matching with FAERS data...")
        
        # Detect adverse events using the FAERS matcher
        adverse_events = self.faers_matcher.detect_adverse_events(medicines, symptoms)
        
        return adverse_events
    
    def predict_severity(self, medicine, symptom):
        """Predict the severity of an adverse event.
        
        Args:
            medicine: Medicine name
            symptom: Symptom text
            
        Returns:
            Predicted severity category
        """
        # Combine medicine and symptom as features
        feature = f"{medicine} {symptom}"
        
        # Make prediction
        try:
            severity = self.model.predict([feature])[0]
            probability = np.max(self.model.predict_proba([feature]))
            
            return {
                'severity': severity,
                'confidence': float(probability)
            }
        except Exception as e:
            print(f"Error predicting severity: {e}")
            return {
                'severity': 'Unknown',
                'confidence': 0.0
            }
    
    def analyze_conversation(self, conversation_text):
        """Analyze a conversation for adverse drug events.
        
        Args:
            conversation_text: The conversation transcript text
            
        Returns:
            Dictionary with analysis results
        """
        print("Analyzing conversation for adverse drug events...")
        
        # Process the conversation
        medicines, symptoms = self.process_conversation(conversation_text)
        
        # Match with FAERS data
        adverse_events = self.match_with_faers(medicines, symptoms)
        
        # Enhance with model predictions
        for event in adverse_events:
            # For each matched symptom, predict severity
            for symptom_match in event['matched_symptoms']:
                prediction = self.predict_severity(
                    event['medicine'], 
                    symptom_match['symptom']
                )
                symptom_match['predicted_severity'] = prediction['severity']
                symptom_match['prediction_confidence'] = prediction['confidence']
        
        # Prepare results
        results = {
            'extracted_medicines': medicines,
            'extracted_symptoms': symptoms,
            'adverse_events': adverse_events,
            'summary': {
                'medicine_count': len(medicines),
                'symptom_count': len(symptoms),
                'adverse_event_count': len(adverse_events)
            }
        }
        
        return results

# Example usage
def main():
    """Example usage of the AdverseEventPredictor class."""
    # Sample conversation text
    conversation = """
     Patient: I've been taking Amlodipine 5mg for my blood pressure for a few weeks, but I’ve started getting really swollen ankles.
    Doctor: I see. How long has the swelling been happening? Is it constant or does it come and go?
    Patient: It’s been getting worse over the last few days, especially after I’ve been standing for a while. It’s mostly around my ankles and calves.
    Doctor: Thank you for the details. Amlodipine can sometimes cause swelling in the legs or ankles. Are you experiencing any other symptoms, like dizziness or shortness of breath?
    Patient: I’ve been feeling a little lightheaded when I stand up quickly, but I haven’t had trouble breathing.
    Doctor: It sounds like the swelling could be a side effect of Amlodipine, and the dizziness may be from the drop in blood pressure. I think we might want to try switching to a different medication, like ACE inhibitors or ARBs, which can help manage your blood pressure without the same swelling.
    Patient: That sounds good. I’m also taking some aspirin for my heart condition and occasionally some ibuprofen for muscle pain.
    Doctor: Got it. Aspirin is fine, and we’ll make sure there are no issues with your new medication. Ibuprofen can sometimes affect kidney function, so it’s something to be mindful of while we adjust your treatment plan. We’ll keep monitoring your blood pressure and make sure we’re on the right track. How’s your heart doing otherwise?
    """
    
    try:
        # Initialize the predictor
        predictor = AdverseEventPredictor()
        
        # Analyze the conversation
        results = predictor.analyze_conversation(conversation)
        
        # Print the results
        print("\nAnalysis Results:")
        print(f"Extracted Medicines: {results['extracted_medicines']}")
        print(f"Extracted Symptoms: {results['extracted_symptoms']}")
        
        print("\nDetected Adverse Events:")
        for i, event in enumerate(results['adverse_events'], 1):
            print(f"\nAdverse Event #{i}:")
            print(f"Medicine: {event['medicine']} (matched to {event['matched_drug']})")
            print(f"FAERS Severity: {event['severity']}")
            print("Matched Symptoms:")
            for match in event['matched_symptoms']:
                print(f"  - {match['symptom']} (matched to {match['matched_reaction']})")
                print(f"    Predicted Severity: {match['predicted_severity']} (confidence: {match['prediction_confidence']:.2f})")
    
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()