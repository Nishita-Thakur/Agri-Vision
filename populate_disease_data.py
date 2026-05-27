"""
Populate database with cotton disease data
"""
from app import app, db
from models import Disease, Treatment, Symptom, DiseaseSymptom

def populate_diseases():
    """Populate database with cotton diseases, treatments, and symptoms"""
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Clear existing data
        DiseaseSymptom.query.delete()
        Symptom.query.delete()
        Treatment.query.delete()
        Disease.query.delete()
        db.session.commit()
        
        # Create symptoms
        symptoms_data = [
            # Leaf symptoms
            {"name": "Yellow spots on leaves", "description": "Small yellow spots appearing on leaf surface", "category": "leaf", "severity_indicator": "mild"},
            {"name": "Brown lesions on leaves", "description": "Brown or black lesions on leaves", "category": "leaf", "severity_indicator": "moderate"},
            {"name": "Leaf curling", "description": "Leaves curling upward or downward", "category": "leaf", "severity_indicator": "moderate"},
            {"name": "Leaf wilting", "description": "Leaves losing turgor and drooping", "category": "leaf", "severity_indicator": "severe"},
            {"name": "Reddening of leaves", "description": "Leaves turning red or purple", "category": "leaf", "severity_indicator": "moderate"},
            {"name": "White powdery coating", "description": "White powdery substance on leaf surface", "category": "leaf", "severity_indicator": "moderate"},
            {"name": "Leaf spots with concentric rings", "description": "Circular spots with target-like appearance", "category": "leaf", "severity_indicator": "moderate"},
            {"name": "Chlorosis between veins", "description": "Yellowing between leaf veins", "category": "leaf", "severity_indicator": "mild"},
            
            # Stem symptoms
            {"name": "Stem cankers", "description": "Sunken lesions on stems", "category": "stem", "severity_indicator": "severe"},
            {"name": "Stem discoloration", "description": "Darkening or reddening of stem tissue", "category": "stem", "severity_indicator": "moderate"},
            {"name": "Stem splitting", "description": "Cracks or splits in stem bark", "category": "stem", "severity_indicator": "severe"},
            {"name": "Stem galls", "description": "Abnormal growths on stems", "category": "stem", "severity_indicator": "moderate"},
            
            # Boll symptoms
            {"name": "Boll rot", "description": "Soft rotting of cotton bolls", "category": "boll", "severity_indicator": "severe"},
            {"name": "Boll shedding", "description": "Premature dropping of bolls", "category": "boll", "severity_indicator": "severe"},
            {"name": "Boll discoloration", "description": "Unusual color changes in bolls", "category": "boll", "severity_indicator": "moderate"},
            {"name": "Boll deformation", "description": "Abnormal boll shape or size", "category": "boll", "severity_indicator": "moderate"},
            {"name": "Lint discoloration", "description": "Cotton fibers turning yellow or brown", "category": "boll", "severity_indicator": "severe"},
            
            # Root symptoms
            {"name": "Root rot", "description": "Decay and discoloration of roots", "category": "root", "severity_indicator": "severe"},
            {"name": "Root galls", "description": "Swollen growths on roots", "category": "root", "severity_indicator": "severe"},
            {"name": "Stunted root growth", "description": "Poor root development", "category": "root", "severity_indicator": "moderate"},
            
            # General symptoms
            {"name": "Stunted growth", "description": "Overall reduced plant size", "category": "general", "severity_indicator": "moderate"},
            {"name": "Premature defoliation", "description": "Early leaf drop", "category": "general", "severity_indicator": "moderate"},
            {"name": "Plant death", "description": "Complete plant mortality", "category": "general", "severity_indicator": "severe"},
            {"name": "Reduced yield", "description": "Significant decrease in cotton production", "category": "general", "severity_indicator": "severe"},
        ]
        
        symptoms = {}
        for s_data in symptoms_data:
            symptom = Symptom(**s_data)
            db.session.add(symptom)
            db.session.flush()
            symptoms[s_data["name"]] = symptom
        
        # Create diseases
        diseases_data = [
            {
                "name": "Bacterial Blight",
                "scientific_name": "Xanthomonas axonopodis pv. malvacearum",
                "description": "Bacterial blight is a serious disease affecting cotton plants, caused by the bacterium Xanthomonas axonopodis. It can cause significant yield losses if not managed properly.",
                "causes": "Caused by the bacterium Xanthomonas axonopodis pv. malvacearum. The bacteria survive in crop debris and are spread by rain, irrigation water, and contaminated equipment.",
                "symptoms": "Water-soaked lesions on leaves that turn brown and angular. Cankers on stems. Bolls may become infected and rot.",
                "severity": "high",
                "spread_rate": "fast",
                "affected_parts": "leaves, stems, bolls",
                "favorable_conditions": "Warm, wet weather with high humidity. Temperatures between 25-30°C favor disease development.",
                "prevention": "Use disease-free seeds. Practice crop rotation. Avoid overhead irrigation. Remove infected plant debris. Use resistant varieties when available.",
                "treatments": [
                    {
                        "name": "Copper-based bactericides",
                        "type": "chemical",
                        "description": "Copper sprays can help suppress bacterial populations when applied early in the infection cycle.",
                        "application_method": "Foliar spray",
                        "dosage": "Follow manufacturer instructions",
                        "timing": "At first sign of symptoms",
                        "effectiveness": "moderate",
                        "cost": "moderate",
                        "precautions": "Avoid applying during flowering to prevent phytotoxicity. Use protective equipment.",
                        "resistance_management": "Rotate with other control methods to prevent resistance buildup."
                    },
                    {
                        "name": "Streptomycin sprays",
                        "type": "chemical",
                        "description": "Antibiotic treatment for bacterial infections. Effective against Xanthomonas species.",
                        "application_method": "Foliar spray",
                        "dosage": "As per label instructions",
                        "timing": "Early infection stage",
                        "effectiveness": "high",
                        "cost": "high",
                        "precautions": "Use sparingly to prevent antibiotic resistance. Follow all safety guidelines.",
                        "resistance_management": "Use only when necessary and rotate with other treatments."
                    }
                ],
                "symptom_names": ["Water-soaked lesions", "Angular leaf spots", "Stem cankers", "Boll rot"]
            },
            {
                "name": "Fusarium Wilt",
                "scientific_name": "Fusarium oxysporum f. sp. vasinfectum",
                "description": "Fusarium wilt is a soil-borne fungal disease that attacks the vascular system of cotton plants, causing wilting and plant death.",
                "causes": "Caused by the soil-borne fungus Fusarium oxysporum f. sp. vasinfectum. The fungus enters through roots and colonizes the vascular system.",
                "symptoms": "Yellowing and wilting of lower leaves. Vascular discoloration (brown streaks) in stem. Stunted growth. Plant death in severe cases.",
                "severity": "severe",
                "spread_rate": "moderate",
                "affected_parts": "roots, stems, leaves",
                "favorable_conditions": "Warm soil temperatures (25-30°C). Stress conditions like drought or nutrient deficiency increase susceptibility.",
                "prevention": "Use resistant varieties. Practice long crop rotation (3-4 years). Avoid soil compaction. Maintain optimal soil moisture and nutrition.",
                "treatments": [
                    {
                        "name": "Soil fumigation",
                        "type": "chemical",
                        "description": "Chemical treatment of soil to reduce fungal populations before planting.",
                        "application_method": "Soil treatment",
                        "dosage": "As per product label",
                        "timing": "Pre-planting",
                        "effectiveness": "high",
                        "cost": "high",
                        "precautions": "Requires professional application. Follow all safety regulations.",
                        "resistance_management": "Rotate with non-host crops to reduce pathogen load."
                    },
                    {
                        "name": "Biological control agents",
                        "type": "biological",
                        "description": "Beneficial microorganisms that suppress Fusarium growth in soil.",
                        "application_method": "Soil drench or seed treatment",
                        "dosage": "As per manufacturer",
                        "timing": "At planting",
                        "effectiveness": "moderate",
                        "cost": "moderate",
                        "precautions": "Store properly. Use before expiration date.",
                        "resistance_management": "Combine with other cultural practices for best results."
                    }
                ],
                "symptom_names": ["Leaf wilting", "Vascular discoloration", "Stunted growth", "Yellowing leaves"]
            },
            {
                "name": "Verticillium Wilt",
                "scientific_name": "Verticillium dahliae",
                "description": "Verticillium wilt is a fungal disease that causes wilting, yellowing, and death of cotton plants. It persists in soil for many years.",
                "causes": "Caused by the soil-borne fungus Verticillium dahliae. Microsclerotia survive in soil for extended periods.",
                "symptoms": "Yellowing and wilting of leaves, often on one side of plant. Vascular browning. Premature defoliation. Stunted growth.",
                "severity": "severe",
                "spread_rate": "slow",
                "affected_parts": "roots, stems, leaves",
                "favorable_conditions": "Cool to moderate temperatures (20-25°C). High soil moisture. Stress conditions increase susceptibility.",
                "prevention": "Use resistant varieties. Long crop rotation with non-host crops. Avoid excessive nitrogen fertilization. Maintain soil health.",
                "treatments": [
                    {
                        "name": "Solarization",
                        "type": "cultural",
                        "description": "Soil solarization using clear plastic to heat soil and reduce pathogen populations.",
                        "application_method": "Soil treatment",
                        "dosage": "4-6 weeks during hot season",
                        "timing": "Pre-planting in summer",
                        "effectiveness": "moderate",
                        "cost": "low",
                        "precautions": "Requires hot climate. Not effective in cool regions.",
                        "resistance_management": "Combine with crop rotation for long-term control."
                    },
                    {
                        "name": "Resistant varieties",
                        "type": "cultural",
                        "description": "Planting cotton varieties with genetic resistance to Verticillium wilt.",
                        "application_method": "Seed selection",
                        "dosage": "N/A",
                        "timing": "At planting",
                        "effectiveness": "high",
                        "cost": "moderate",
                        "precautions": "Verify resistance level for local pathogen strains.",
                        "resistance_management": "Rotate resistance genes if available."
                    }
                ],
                "symptom_names": ["Leaf wilting", "Vascular discoloration", "Premature defoliation", "Yellowing leaves"]
            },
            {
                "name": "Powdery Mildew",
                "scientific_name": "Leveillula taurica",
                "description": "Powdery mildew is a fungal disease characterized by white powdery growth on leaves, reducing photosynthesis and yield.",
                "causes": "Caused by the fungus Leveillula taurica. Fungal spores spread by wind and thrive in moderate temperatures.",
                "symptoms": "White powdery coating on leaf surfaces. Yellowing of affected areas. Leaf curling and distortion. Premature defoliation in severe cases.",
                "severity": "moderate",
                "spread_rate": "fast",
                "affected_parts": "leaves",
                "favorable_conditions": "Moderate temperatures (20-25°C). High humidity but not free water. Dense plant canopy.",
                "prevention": "Plant resistant varieties. Ensure good air circulation. Avoid excessive nitrogen. Practice proper spacing.",
                "treatments": [
                    {
                        "name": "Sulfur fungicides",
                        "type": "chemical",
                        "description": "Sulfur-based fungicides effective against powdery mildew.",
                        "application_method": "Foliar spray",
                        "dosage": "As per label",
                        "timing": "At first symptoms, repeat every 7-14 days",
                        "effectiveness": "high",
                        "cost": "low",
                        "precautions": "Avoid application in hot weather (>32°C) to prevent phytotoxicity.",
                        "resistance_management": "Rotate with different mode of action fungicides."
                    },
                    {
                        "name": "Neem oil",
                        "type": "biological",
                        "description": "Natural oil with antifungal properties against powdery mildew.",
                        "application_method": "Foliar spray",
                        "dosage": "1-2% solution",
                        "timing": "At first sign of disease",
                        "effectiveness": "moderate",
                        "cost": "low",
                        "precautions": "Test on small area first. Avoid during flowering.",
                        "resistance_management": "Combine with other control methods."
                    }
                ],
                "symptom_names": ["White powdery coating", "Leaf curling", "Yellowing leaves", "Leaf spots"]
            },
            {
                "name": "Cotton Root Rot",
                "scientific_name": "Phymatotrichopsis omnivora",
                "description": "Cotton root rot is a devastating soil-borne disease that causes rapid wilting and death of cotton plants.",
                "causes": "Caused by the soil-borne fungus Phymatotrichopsis omnivora. The fungus forms strand-like structures in soil.",
                "symptoms": "Sudden wilting of plants. Leaves remain attached but turn brown. Root systems decay and have string-like fungal strands. Plant death.",
                "severity": "severe",
                "spread_rate": "moderate",
                "affected_parts": "roots",
                "favorable_conditions": "Warm soil temperatures (28-35°C). Alkaline soils. Moist conditions followed by dry periods.",
                "prevention": "Avoid planting in infested fields. Use deep plowing to expose fungal strands. Plant early to avoid peak disease period.",
                "treatments": [
                    {
                        "name": "Deep plowing",
                        "type": "cultural",
                        "description": "Deep tillage to expose fungal strands to sunlight and drying conditions.",
                        "application_method": "Soil cultivation",
                        "dosage": "Plow to 30-40 cm depth",
                        "timing": "Pre-planting or post-harvest",
                        "effectiveness": "moderate",
                        "cost": "moderate",
                        "precautions": "May increase soil erosion. Use conservation practices.",
                        "resistance_management": "Combine with crop rotation."
                    },
                    {
                        "name": "Early planting",
                        "type": "cultural",
                        "description": "Plant cotton early to avoid peak disease period when soil temperatures are highest.",
                        "application_method": "Cultural practice",
                        "dosage": "N/A",
                        "timing": "Early season planting",
                        "effectiveness": "moderate",
                        "cost": "low",
                        "precautions": "Ensure adequate soil moisture for early planting.",
                        "resistance_management": "Combine with resistant varieties."
                    }
                ],
                "symptom_names": ["Root rot", "Sudden wilting", "Plant death", "Stunted growth"]
            },
            {
                "name": "Alternaria Leaf Spot",
                "scientific_name": "Alternaria macrospora",
                "description": "Alternaria leaf spot causes circular spots on cotton leaves with concentric rings, leading to defoliation and yield loss.",
                "causes": "Caused by the fungus Alternaria macrospora. Spores spread by wind and rain splash.",
                "symptoms": "Circular brown spots with concentric rings on leaves. Yellow halos around spots. Premature defoliation in severe infections.",
                "severity": "moderate",
                "spread_rate": "moderate",
                "affected_parts": "leaves",
                "favorable_conditions": "Warm, humid weather. Leaf wetness from rain or dew. Poor plant nutrition.",
                "prevention": "Use disease-free seeds. Avoid overhead irrigation. Remove crop debris. Ensure proper plant spacing.",
                "treatments": [
                    {
                        "name": "Chlorothalonil",
                        "type": "chemical",
                        "description": "Broad-spectrum fungicide effective against Alternaria and other leaf spot diseases.",
                        "application_method": "Foliar spray",
                        "dosage": "As per label",
                        "timing": "At first symptoms, repeat every 7-10 days",
                        "effectiveness": "high",
                        "cost": "moderate",
                        "precautions": "Follow safety guidelines. Avoid application during bloom.",
                        "resistance_management": "Rotate with different fungicide classes."
                    },
                    {
                        "name": "Mancozeb",
                        "type": "chemical",
                        "description": "Protective fungicide for Alternaria leaf spot control.",
                        "application_method": "Foliar spray",
                        "dosage": "As per label",
                        "timing": "Preventative applications every 7-14 days",
                        "effectiveness": "high",
                        "cost": "low",
                        "precautions": "Use protective equipment. Follow pre-harvest intervals.",
                        "resistance_management": "Alternate with other fungicides."
                    }
                ],
                "symptom_names": ["Leaf spots with concentric rings", "Yellow spots on leaves", "Premature defoliation", "Brown lesions on leaves"]
            },
            {
                "name": "Cotton Boll Rot",
                "scientific_name": "Various pathogens",
                "description": "Boll rot is a complex disease caused by multiple fungi and bacteria that infect cotton bolls, causing lint damage and yield loss.",
                "causes": "Caused by various fungi (Diplodia, Fusarium) and bacteria. Often associated with insect damage or boll opening issues.",
                "symptoms": "Soft, watery rot of bolls. Discoloration of lint. Boll shedding. Fungal growth on boll surface.",
                "severity": "severe",
                "spread_rate": "fast",
                "affected_parts": "bolls",
                "favorable_conditions": "Warm, humid conditions. Rain during boll opening. Insect damage to bolls.",
                "prevention": "Control boll-feeding insects. Ensure proper plant spacing for air circulation. Avoid late-season irrigation.",
                "treatments": [
                    {
                        "name": "Insect control",
                        "type": "integrated",
                        "description": "Control boll weevils and other insects that create entry points for pathogens.",
                        "application_method": "IPM program",
                        "dosage": "As per insecticide labels",
                        "timing": "Throughout boll development",
                        "effectiveness": "high",
                        "cost": "moderate",
                        "precautions": "Follow insecticide resistance management guidelines.",
                        "resistance_management": "Rotate insecticide modes of action."
                    },
                    {
                        "name": "Fungicide sprays",
                        "type": "chemical",
                        "description": "Protective fungicide applications during boll development.",
                        "application_method": "Foliar spray",
                        "dosage": "As per label",
                        "timing": "During boll development, especially before rain",
                        "effectiveness": "moderate",
                        "cost": "moderate",
                        "precautions": "Consider pre-harvest intervals. Protect pollinators.",
                        "resistance_management": "Rotate fungicide classes."
                    }
                ],
                "symptom_names": ["Boll rot", "Boll shedding", "Lint discoloration", "Boll deformation"]
            },
            {
                "name": "Red Leaf Spot",
                "scientific_name": "Cercospora gossypina",
                "description": "Red leaf spot causes reddish-purple spots on cotton leaves, leading to reduced photosynthesis and premature defoliation.",
                "causes": "Caused by the fungus Cercospora gossypina. Spores spread by wind and rain.",
                "symptoms": "Red to purple circular spots on leaves. Spots may have gray centers. Yellowing around spots. Premature defoliation.",
                "severity": "moderate",
                "spread_rate": "moderate",
                "affected_parts": "leaves",
                "favorable_conditions": "Warm, humid weather. Leaf wetness. Dense plant canopy.",
                "prevention": "Use resistant varieties. Ensure good air circulation. Remove infected leaves. Avoid overhead irrigation.",
                "treatments": [
                    {
                        "name": "Propiconazole",
                        "type": "chemical",
                        "description": "Systemic fungicide effective against Cercospora leaf spots.",
                        "application_method": "Foliar spray",
                        "dosage": "As per label",
                        "timing": "At first symptoms, repeat every 14-21 days",
                        "effectiveness": "high",
                        "cost": "moderate",
                        "precautions": "Follow label restrictions. Use protective equipment.",
                        "resistance_management": "Rotate with non-DMI fungicides."
                    },
                    {
                        "name": "Azoxystrobin",
                        "type": "chemical",
                        "description": "Broad-spectrum fungicide with preventive and curative activity.",
                        "application_method": "Foliar spray",
                        "dosage": "As per label",
                        "timing": "Preventative or early curative",
                        "effectiveness": "high",
                        "cost": "high",
                        "precautions": "Follow resistance management guidelines. Avoid consecutive applications.",
                        "resistance_management": "Rotate with different fungicide classes."
                    }
                ],
                "symptom_names": ["Reddening of leaves", "Leaf spots with concentric rings", "Yellow spots on leaves", "Premature defoliation"]
            }
        ]
        
        for d_data in diseases_data:
            # Extract treatments
            treatments_data = d_data.pop("treatments", [])
            symptom_names = d_data.pop("symptom_names", [])
            
            # Create disease
            disease = Disease(**d_data)
            db.session.add(disease)
            db.session.flush()
            
            # Create treatments
            for t_data in treatments_data:
                treatment = Treatment(disease_id=disease.id, **t_data)
                db.session.add(treatment)
            
            # Link symptoms
            for symptom_name in symptom_names:
                # Find matching symptom (partial match)
                matching_symptom = None
                for symptom in symptoms.values():
                    if symptom_name.lower() in symptom.name.lower() or symptom.name.lower() in symptom_name.lower():
                        matching_symptom = symptom
                        break
                
                if matching_symptom:
                    association = DiseaseSymptom(
                        disease_id=disease.id,
                        symptom_id=matching_symptom.id,
                        confidence=0.7
                    )
                    db.session.add(association)
        
        db.session.commit()
        print("✓ Disease database populated successfully!")
        print(f"  - {len(diseases_data)} diseases added")
        print(f"  - {len(symptoms_data)} symptoms added")
        total_treatments = Treatment.query.count()
        print(f"  - Total treatments: {total_treatments}")

if __name__ == "__main__":
    populate_diseases()
