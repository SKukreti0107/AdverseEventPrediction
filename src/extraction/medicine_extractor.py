"""Medicine Name Extractor Module.

This module extracts medicine names from text using enhanced biomedical NER.
It processes conversation text and identifies medication mentions with high accuracy.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from .biomedical_ner import BiomedicalNER

class MedicineExtractor:
    """Class for extracting medicine names from text using enhanced biomedical NER."""
    
    def __init__(self, model_name="alvaroalon2/biobert_genetic_ner"):
        """Initialize the medicine extractor with a specialized biomedical NER model.
        
        Args:
            model_name: The name of the pre-trained model to use
                       Default is BioBERT which is fine-tuned for biomedical NER
        """
        print(f"Initializing MedicineExtractor with enhanced biomedical NER")
        try:
            # Initialize the biomedical NER component
            self.ner = BiomedicalNER(model_name=model_name)
            print("Enhanced biomedical NER initialized successfully")
            
            # Legacy model support (for backward compatibility)
            self.device = self.ner.device
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
        
        # Replace common abbreviations
        replacements = {
            'mg': ' milligrams ',
            'ml': ' milliliters ',
            'g': ' grams ',
            'mcg': ' micrograms ',
            'tabs': ' tablets ',
            'tab': ' tablet ',
            'caps': ' capsules ',
            'cap': ' capsule '
        }
        
        for abbr, full in replacements.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text)
        
        return text
    
    def extract_medicines(self, text, confidence_threshold=0.7):
        """Extract medicine names from the given text using enhanced biomedical NER.
        
        Args:
            text: The input text to extract medicine names from
            confidence_threshold: Minimum confidence score to include an entity
            
        Returns:
            List of extracted medicine names
        """
        try:
            # Use the biomedical NER to extract drugs
            medicines = self.ner.extract_drugs(text, confidence_threshold=confidence_threshold)
            
            print(f"Extracted {len(medicines)} medicine names from text using enhanced biomedical NER")
            return medicines
        
        except Exception as e:
            print(f"Error extracting medicines: {e}")
            return []
    
    def extract_medicines_from_conversation(self, conversation_text, confidence_threshold=0.7):
        """Extract medicine names from a conversation transcript using enhanced biomedical NER.
        
        Args:
            conversation_text: The conversation transcript text
            confidence_threshold: Minimum confidence score to include an entity
            
        Returns:
            List of extracted medicine names
        """
        try:
            # Process the entire conversation with the biomedical NER
            # This is more effective than sentence-by-sentence as it captures context
            drug_entities = self.ner.extract_entities_from_conversation(conversation_text, entity_type="DRUG")
            
            # Filter by confidence threshold and extract just the text
            medicines = [entity['text'] for entity in drug_entities if entity['score'] >= confidence_threshold]
            
            # Add rule-based extraction for common drug names that might be missed
            # This helps catch medicines that the model might not recognize
            for drug in self.ner.common_drugs:
                # Check if the drug name appears in the conversation (case-insensitive)
                if re.search(r'\b' + re.escape(drug) + r'\b', conversation_text.lower()):
                    if drug not in medicines:
                        medicines.append(drug)
            
            # Remove duplicates and sort
            unique_medicines = sorted(list(set(medicines)))
            
            print(f"Extracted {len(unique_medicines)} unique medicine names from conversation using enhanced biomedical NER")
            return unique_medicines
        
        except Exception as e:
            print(f"Error extracting medicines from conversation: {e}")
            return []

# Example usage
def main():
    """Example usage of the MedicineExtractor class."""
    # Sample conversation text
    conversation = """
    Patient: I've been taking Lisinopril 10mg for my blood pressure, but I think it's causing a dry cough.
    Doctor: How long have you been experiencing this cough?
    Patient: About two weeks. I'm also on Metformin 500mg twice daily for diabetes and occasionally take Tylenol for headaches.
    Doctor: I see. Let's consider switching your blood pressure medication. Perhaps we could try Losartan instead.
    """
    
    # Initialize the extractor
    try:
        extractor = MedicineExtractor()
        
        # Extract medicines from the conversation
        medicines = extractor.extract_medicines_from_conversation(conversation)
        
        # Print the results
        print("\nExtracted Medicines:")
        for medicine in medicines:
            print(f"- {medicine}")
    
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()