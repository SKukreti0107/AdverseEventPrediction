"""Symptom Extractor Module.

This module extracts symptoms from text using enhanced biomedical NER.
It processes conversation text and identifies symptom mentions with high accuracy.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from .biomedical_ner import BiomedicalNER

class SymptomExtractor:
    """Class for extracting symptoms from text using enhanced biomedical NER."""
    
    def __init__(self, model_name="alvaroalon2/biobert_genetic_ner"):
        """Initialize the symptom extractor with a specialized biomedical NER model.
        
        Args:
            model_name: The name of the pre-trained model to use
                       Default is BioBERT which is fine-tuned for biomedical NER
        """
        print(f"Initializing SymptomExtractor with enhanced biomedical NER")
        try:
            # Initialize the biomedical NER component
            self.ner = BiomedicalNER(model_name=model_name)
            print("Enhanced biomedical NER initialized successfully")
            
            # Legacy model support (for backward compatibility)
            self.device = self.ner.device
            
            # Keep symptom keywords for additional context
            self.symptom_keywords = [
                'pain', 'ache', 'discomfort', 'nausea', 'vomiting', 'dizziness',
                'headache', 'fever', 'cough', 'rash', 'swelling', 'fatigue',
                'tired', 'exhaustion', 'weakness', 'numbness', 'tingling',
                'itching', 'burning', 'cramping', 'stiffness', 'sore',
                'difficulty', 'problem', 'issue', 'trouble', 'distress',
                'discomfort', 'feeling', 'sensation'
            ]
            
        except Exception as e:
            print(f"Error initializing enhanced biomedical NER: {e}")
            raise
    
    def preprocess_text(self, text):
        """Preprocess the input text for better extraction.
        
        Args:
            text: The input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_symptoms(self, text, confidence_threshold=0.7):
        """Extract symptoms from the given text using enhanced biomedical NER.
        
        Args:
            text: The input text to extract symptoms from
            confidence_threshold: Minimum confidence score to include an entity
            
        Returns:
            List of extracted symptoms
        """
        try:
            # Use the biomedical NER to extract symptoms and diseases
            # (diseases are often mentioned as symptoms in patient conversations)
            symptoms = self.ner.extract_symptoms(text, confidence_threshold=confidence_threshold)
            
            print(f"Extracted {len(symptoms)} symptoms from text using enhanced biomedical NER")
            return symptoms
        
        except Exception as e:
            print(f"Error extracting symptoms: {e}")
            return []
    
    def extract_symptoms_from_conversation(self, conversation_text, confidence_threshold=0.7):
        """Extract symptoms from a conversation transcript using enhanced biomedical NER.
        
        Args:
            conversation_text: The conversation transcript text
            confidence_threshold: Minimum confidence score to include an entity
            
        Returns:
            List of extracted symptoms
        """
        try:
            # Process the entire conversation with the biomedical NER
            # This is more effective than sentence-by-sentence as it captures context
            symptom_entities = self.ner.extract_entities_from_conversation(conversation_text, entity_type="SYMPTOM")
            disease_entities = self.ner.extract_entities_from_conversation(conversation_text, entity_type="DISEASE")
            
            # Filter by confidence threshold and extract just the text
            symptoms = [entity['text'] for entity in symptom_entities if entity['score'] >= confidence_threshold]
            diseases = [entity['text'] for entity in disease_entities if entity['score'] >= confidence_threshold]
            
            # Add rule-based extraction for common symptoms that might be missed
            # This helps catch symptoms that the model might not recognize
            for symptom in self.ner.common_symptoms:
                # Check if the symptom appears in the conversation (case-insensitive)
                if re.search(r'\b' + re.escape(symptom) + r'\b', conversation_text.lower()):
                    if symptom not in symptoms and symptom not in diseases:
                        symptoms.append(symptom)
            
            # Combine symptoms and diseases, remove duplicates and sort
            all_symptoms = symptoms + diseases
            unique_symptoms = sorted(list(set(all_symptoms)))
            
            print(f"Extracted {len(unique_symptoms)} unique symptoms from conversation using enhanced biomedical NER")
            return unique_symptoms
        
        except Exception as e:
            print(f"Error extracting symptoms from conversation: {e}")
            return []

# Example usage
def main():
    """Example usage of the SymptomExtractor class."""
    # Sample conversation text
    conversation = """
    Patient: I've been having a terrible headache for the past three days, and I'm also experiencing some dizziness.
    Doctor: I see. Are you having any other symptoms like nausea or sensitivity to light?
    Patient: Yes, I do feel nauseated, especially in the morning. And bright lights make the headache worse.
    Doctor: Have you noticed any fever or neck stiffness?
    Patient: No fever, but my neck does feel a bit stiff and sore.
    """
    
    # Initialize the extractor
    try:
        extractor = SymptomExtractor()
        
        # Extract symptoms from the conversation
        symptoms = extractor.extract_symptoms_from_conversation(conversation)
        
        # Print the results
        print("\nExtracted Symptoms:")
        for symptom in symptoms:
            print(f"- {symptom}")
    
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()