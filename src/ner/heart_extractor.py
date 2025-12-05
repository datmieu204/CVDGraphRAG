"""
Heart Failure Entity Extractor
Author: CVDGraphRAG Team
"""

import torch
from transformers import BertForTokenClassification, AutoTokenizer

class HeartExtractor:
    def __init__(self):
        self.model_name = "MilosKosRad/BioNER"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BertForTokenClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.eval()
        self.all_classes = [
            "Specific Disease",            
            "Disease Class",
            "Cardiovascular Symptom",      
            "Disease Or Phenotypic Feature", 
            "Anatomical Structure",         
            "Gene Or Gene Product",       
            "Diagnostic Procedure",         
            "Lab Test",                    
            "Measurement",                
            "Clinical Classification",      
            "Drug",              
            "Medical Device",              
            "Therapeutic Procedure",        
            "Dosage", "Route", "Frequency", "Risk Factor", 
            "Clinical Outcome", "Chemical", "Frequency", "Route", "ADE",
            "Therapeutic Procedure", "Composite Mention", "Modifier",
            "Sequence Variant", "Gene Or Gene Product", "Disease Or Phenotypic Feature",
            "Organism Taxon", "Cell Line", "Cell Type", "Protein", "DNA", "RNA",
            "Chemical Entity", "Chemical", "Chemical Family", "Drug",
            "Frequency", "Strength", "Dosage", "Form", "Reason", "Route", 
            "ADE", "Duration"
        ]

    
    def extract_entities(self, text) -> dict:
        """
        Extract heart failure related entities from the input text.
        Args:
            text (str): Input text from which to extract entities.
        Returns:
            dict: A dictionary with entity classes as keys and lists of extracted phrases as values.
        """
        result = {}

        inputs = self.tokenizer(
            self.all_classes,
            [text] * len(self.all_classes),
            padding=True,
            truncation=True, 
            return_tensors="pt", 
            max_length=512,
            return_offsets_mapping=True,
            add_special_tokens=True
        )

        with torch.no_grad():
            model_inputs = {k: v for k, v in inputs.items() if k != "offset_mapping"}
            outputs = self.model(**model_inputs)
        
        batch_predictions = torch.argmax(outputs.logits, dim=2)

        for i, entity_class in enumerate(self.all_classes):
            predictions = batch_predictions[i].tolist()
            offset_mapping = inputs['offset_mapping'][i].tolist()
            sequence_ids = inputs.sequence_ids(i)

            found_phrases = []
            current_start = None
            current_end = None

            for idx, (pred, offset, seq_id) in enumerate(zip(predictions, offset_mapping, sequence_ids)):
                if seq_id != 1 or offset[0] == 0:
                    continue

                if pred == 1:
                    if current_start is None:
                        current_start = offset[0]
                    current_end = offset[1]
                else:
                    if current_start is not None:
                        phrase = text[current_start:current_end]
                        found_phrases.append(phrase)
                        current_start = None
                        current_end = None
            
            if current_start is not None:
                phrase = text[current_start:current_end]
                found_phrases.append(phrase)
            
            if found_phrases:
                result[entity_class] = list(set(found_phrases))

        return result
    

if __name__ == "__main__":
    extractor = HeartExtractor()
    clinical_text = """
    Patient is a 65-year-old male admitted with acute decompensated heart failure.
    He reports worsening dyspnea on exertion and orthopnea. 
    Physical exam reveals bilateral pitting edema and jugular venous distention.
    Echocardiogram shows a reduced Ejection Fraction of 35% and left ventricular hypertrophy.
    Labs are significant for elevated NT-proBNP (4000 pg/mL).
    Current medications include Furosemide 40mg IV, Lisinopril, and Carvedilol.
    """
    entities = extractor.extract_entities(clinical_text)
    for entity_class, phrases in entities.items():
        print(f"{entity_class}: {phrases}")