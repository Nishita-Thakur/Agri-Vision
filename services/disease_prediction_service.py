"""
Disease Prediction Service
Uses weather data and historical patterns to predict disease outbreaks
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """ML-based disease prediction using weather data"""
    
    def __init__(self):
        # Disease-specific weather thresholds (based on agricultural research)
        self.disease_thresholds = {
            'bacterial_blight': {
                'temp_min': 25,
                'temp_max': 35,
                'humidity_min': 70,
                'rainfall_min': 5,
                'temp_weight': 0.3,
                'humidity_weight': 0.4,
                'rainfall_weight': 0.3
            },
            'fusarium_wilt': {
                'temp_min': 25,
                'temp_max': 30,
                'humidity_min': 60,
                'rainfall_min': 0,
                'temp_weight': 0.4,
                'humidity_weight': 0.3,
                'rainfall_weight': 0.3
            },
            'verticillium_wilt': {
                'temp_min': 20,
                'temp_max': 25,
                'humidity_min': 60,
                'rainfall_min': 0,
                'temp_weight': 0.4,
                'humidity_weight': 0.3,
                'rainfall_weight': 0.3
            },
            'powdery_mildew': {
                'temp_min': 20,
                'temp_max': 28,
                'humidity_min': 60,
                'rainfall_min': 0,
                'temp_weight': 0.3,
                'humidity_weight': 0.5,
                'rainfall_weight': 0.2
            },
            'cotton_root_rot': {
                'temp_min': 28,
                'temp_max': 35,
                'humidity_min': 50,
                'rainfall_min': 10,
                'temp_weight': 0.3,
                'humidity_weight': 0.3,
                'rainfall_weight': 0.4
            },
            'alternaria_leaf_spot': {
                'temp_min': 20,
                'temp_max': 30,
                'humidity_min': 70,
                'rainfall_min': 5,
                'temp_weight': 0.3,
                'humidity_weight': 0.4,
                'rainfall_weight': 0.3
            },
            'cotton_boll_rot': {
                'temp_min': 25,
                'temp_max': 32,
                'humidity_min': 80,
                'rainfall_min': 10,
                'temp_weight': 0.3,
                'humidity_weight': 0.4,
                'rainfall_weight': 0.3
            },
            'red_leaf_spot': {
                'temp_min': 22,
                'temp_max': 30,
                'humidity_min': 65,
                'rainfall_min': 5,
                'temp_weight': 0.3,
                'humidity_weight': 0.4,
                'rainfall_weight': 0.3
            }
        }
    
    def calculate_risk_score(self, weather_data: Dict, disease_name: str) -> float:
        """
        Calculate disease risk score (0-100) based on weather conditions
        """
        if disease_name not in self.disease_thresholds:
            return 0
        
        thresholds = self.disease_thresholds[disease_name]
        
        temp = weather_data.get('temperature_avg', weather_data.get('temperature', 0))
        humidity = weather_data.get('humidity', 0)
        rainfall = weather_data.get('rainfall', 0)
        
        # Calculate individual factor scores
        temp_score = self._calculate_factor_score(
            temp, 
            thresholds['temp_min'], 
            thresholds['temp_max']
        )
        
        humidity_score = self._calculate_factor_score(
            humidity,
            thresholds['humidity_min'],
            100  # Max humidity
        )
        
        rainfall_score = self._calculate_factor_score(
            rainfall,
            thresholds['rainfall_min'],
            50  # High rainfall threshold
        )
        
        # Weighted average
        risk_score = (
            temp_score * thresholds['temp_weight'] +
            humidity_score * thresholds['humidity_weight'] +
            rainfall_score * thresholds['rainfall_weight']
        ) * 100
        
        return min(max(risk_score, 0), 100)
    
    def _calculate_factor_score(self, value: float, min_threshold: float, max_threshold: float) -> float:
        """
        Calculate normalized score for a single factor (0-1)
        """
        if value < min_threshold:
            # Below minimum - linear decrease
            return max(0, value / min_threshold * 0.5)
        elif value > max_threshold:
            # Above maximum - cap at 1
            return 1.0
        else:
            # Within optimal range
            return 0.8 + (value - min_threshold) / (max_threshold - min_threshold) * 0.2
    
    def get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score < 25:
            return 'low'
        elif risk_score < 50:
            return 'moderate'
        elif risk_score < 75:
            return 'high'
        else:
            return 'severe'
    
    def predict_disease_risk(self, weather_forecast: List[Dict], disease_name: str) -> List[Dict]:
        """
        Predict disease risk for multiple days based on weather forecast
        """
        predictions = []
        
        for day_data in weather_forecast:
            risk_score = self.calculate_risk_score(day_data, disease_name)
            risk_level = self.get_risk_level(risk_score)
            
            predictions.append({
                'date': day_data.get('date'),
                'risk_score': round(risk_score, 1),
                'risk_level': risk_level,
                'weather': {
                    'temperature_avg': day_data.get('temperature_avg'),
                    'humidity': day_data.get('humidity'),
                    'rainfall': day_data.get('rainfall')
                }
            })
        
        return predictions
    
    def get_all_disease_predictions(self, weather_forecast: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Get predictions for all diseases
        """
        predictions = {}
        
        for disease_name in self.disease_thresholds.keys():
            predictions[disease_name] = self.predict_disease_risk(weather_forecast, disease_name)
        
        return predictions
    
    def get_high_risk_days(self, predictions: List[Dict], threshold: float = 60) -> List[Dict]:
        """
        Filter predictions to show only high-risk days
        """
        return [p for p in predictions if p['risk_score'] >= threshold]
    
    def generate_recommendations(self, disease_name: str, risk_level: str) -> List[str]:
        """
        Generate preventive recommendations based on disease and risk level
        """
        recommendations = []
        
        if risk_level in ['high', 'severe']:
            recommendations.append(f"Immediate action required for {disease_name.replace('_', ' ').title()}.")
            recommendations.append("Apply preventive fungicides if conditions persist.")
            recommendations.append("Monitor fields daily for early symptoms.")
            recommendations.append("Consider drainage improvements if rainfall is high.")
        elif risk_level == 'moderate':
            recommendations.append(f"Monitor for {disease_name.replace('_', ' ').title()} development.")
            recommendations.append("Ensure proper field ventilation and spacing.")
            recommendations.append("Avoid overhead irrigation during high humidity.")
            recommendations.append("Have treatment options ready if conditions worsen.")
        else:
            recommendations.append(f"Low risk for {disease_name.replace('_', ' ').title()}.")
            recommendations.append("Continue regular monitoring.")
            recommendations.append("Maintain good cultural practices.")
        
        return recommendations


class HistoricalPatternAnalyzer:
    """Analyze historical disease patterns for ML learning"""
    
    def __init__(self):
        pass
    
    def analyze_seasonal_patterns(self, occurrences: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze seasonal patterns in disease occurrences
        Returns disease-specific seasonal risk by month
        """
        seasonal_data = {}
        
        for occurrence in occurrences:
            disease_name = occurrence.get('disease_name', 'unknown')
            date_str = occurrence.get('occurrence_date')
            
            if date_str:
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    month = date.month
                    
                    if disease_name not in seasonal_data:
                        seasonal_data[disease_name] = {i: 0 for i in range(1, 13)}
                    
                    seasonal_data[disease_name][month] += 1
                except ValueError:
                    continue
        
        # Normalize to percentages
        for disease in seasonal_data:
            total = sum(seasonal_data[disease].values())
            if total > 0:
                for month in seasonal_data[disease]:
                    seasonal_data[disease][month] = (seasonal_data[disease][month] / total) * 100
        
        return seasonal_data
    
    def get_regional_patterns(self, occurrences: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Analyze disease patterns by region
        """
        regional_data = {}
        
        for occurrence in occurrences:
            location = occurrence.get('location_name', 'unknown')
            disease_name = occurrence.get('disease_name', 'unknown')
            
            if location not in regional_data:
                regional_data[location] = {}
            
            if disease_name not in regional_data[location]:
                regional_data[location][disease_name] = 0
            
            regional_data[location][disease_name] += 1
        
        return regional_data
    
    def predict_from_history(self, historical_data: List[Dict], current_location: str, 
                            current_month: int) -> Dict[str, float]:
        """
        Predict disease risk based on historical patterns
        """
        seasonal_patterns = self.analyze_seasonal_patterns(historical_data)
        
        predictions = {}
        
        for disease_name, monthly_data in seasonal_patterns.items():
            risk_score = monthly_data.get(current_month, 0)
            predictions[disease_name] = risk_score
        
        return predictions
    
    def get_peak_season(self, disease_name: str, seasonal_data: Dict) -> Optional[Dict]:
        """
        Get peak season for a specific disease
        """
        if disease_name not in seasonal_data:
            return None
        
        monthly_data = seasonal_data[disease_name]
        peak_month = max(monthly_data, key=monthly_data.get)
        
        return {
            'month': peak_month,
            'risk_percentage': monthly_data[peak_month]
        }
