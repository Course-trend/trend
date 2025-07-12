# student_input_handler.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings
from datetime import datetime, date
warnings.filterwarnings('ignore')

class StudentInputHandler:
    """
    Handles input data from current students and prepares it for prediction
    """
    
    def __init__(self, feature_engineer, trained_model):
        """
        Initialize with trained feature engineer and model
        
        Args:
            feature_engineer: Trained FeatureEngineer instance
            trained_model: Trained CareerPredictionModel instance
        """
        self.feature_engineer = feature_engineer
        self.trained_model = trained_model
        self.required_fields = self._define_required_fields()
        self.optional_fields = self._define_optional_fields()
        
    def _define_required_fields(self) -> Dict[str, Dict[str, Any]]:
        """Define required fields for student input"""
        return {
            'student_id': {
                'type': str,
                'description': 'Unique student identifier'
            },
            'gender': {
                'type': str,
                'options': ['Male', 'Female'],
                'description': 'Student gender'
            },
            'age_at_enrollment': {
                'type': int,
                'min': 17,
                'max': 30,
                'description': 'Age when enrolled at university'
            },
            'province': {
                'type': str,
                'options': ['Western', 'Central', 'Southern', 'Northern', 'Eastern', 
                           'North Western', 'North Central', 'Uva', 'Sabaragamuwa'],
                'description': 'Province of origin'
            },
            'district': {
                'type': str,
                'description': 'District of origin'
            },
            'z_score_AL': {
                'type': float,
                'min': 1.0,
                'max': 2.5,
                'description': 'A/L Z-score (1.0 to 2.5)'
            },
            'pathway': {
                'type': str,
                'options': ['Artificial Intelligence', 'Data Science', 'Cyber Security', 
                           'Scientific Computing', 'Standard'],
                'description': 'Academic pathway/specialization'
            },
            'intake_year': {
                'type': int,
                'min': 2018,
                'max': 2025,
                'description': 'Year of university intake'
            },
            'current_semester': {
                'type': int,
                'min': 1,
                'max': 8,
                'description': 'Current semester (1-8)'
            }
        }
    
    def _define_optional_fields(self) -> Dict[str, Dict[str, Any]]:
        """Define optional fields that enhance prediction accuracy"""
        return {
            'current_gpa': {
                'type': float,
                'min': 0.0,
                'max': 4.0,
                'description': 'Current cumulative GPA (if available)'
            },
            'completed_internships': {
                'type': int,
                'min': 0,
                'max': 10,
                'description': 'Number of completed internships'
            },
            'internship_ratings': {
                'type': list,
                'description': 'List of internship performance ratings (1-5)'
            },
            'total_internship_months': {
                'type': int,
                'min': 0,
                'max': 24,
                'description': 'Total months of internship experience'
            },
            'completed_projects': {
                'type': int,
                'min': 0,
                'max': 20,
                'description': 'Number of completed projects'
            },
            'project_technologies': {
                'type': list,
                'description': 'List of technologies used in projects'
            },
            'certifications_earned': {
                'type': int,
                'min': 0,
                'max': 15,
                'description': 'Number of professional certifications'
            },
            'capstone_domain': {
                'type': str,
                'options': ['AI/ML', 'Web Development', 'Mobile Development', 
                           'Data Analytics', 'Cybersecurity', 'IoT', 'Other'],
                'description': 'Capstone project domain (if completed)'
            },
            'leadership_roles': {
                'type': int,
                'min': 0,
                'max': 5,
                'description': 'Number of leadership positions held'
            },
            'extracurricular_activities': {
                'type': int,
                'min': 0,
                'max': 10,
                'description': 'Number of extracurricular activities'
            }
        }
    
    def get_input_form_schema(self) -> Dict[str, Any]:
        """
        Generate a schema for creating input forms
        
        Returns:
            Dictionary containing field definitions for UI generation
        """
        return {
            'required_fields': self.required_fields,
            'optional_fields': self.optional_fields,
            'form_title': 'Student Career Prediction Input',
            'form_description': 'Provide your current academic and experience information for career outcome prediction'
        }
    
    def validate_input(self, student_data: Dict[str, Any]) -> Tuple[bool, list[str]]:
        """
        Validate student input data
        
        Args:
            student_data: Dictionary containing student information
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        for field_name, field_config in self.required_fields.items():
            if field_name not in student_data:
                errors.append(f"Required field '{field_name}' is missing")
                continue
                
            value = student_data[field_name]
            
            # Type validation
            if field_config['type'] == int and not isinstance(value, int):
                try:
                    student_data[field_name] = int(value)
                except ValueError:
                    errors.append(f"Field '{field_name}' must be an integer")
                    continue
            elif field_config['type'] == float and not isinstance(value, (int, float)):
                try:
                    student_data[field_name] = float(value)
                except ValueError:
                    errors.append(f"Field '{field_name}' must be a number")
                    continue
            
            # Range validation
            if 'min' in field_config and value < field_config['min']:
                errors.append(f"Field '{field_name}' must be at least {field_config['min']}")
            if 'max' in field_config and value > field_config['max']:
                errors.append(f"Field '{field_name}' must be at most {field_config['max']}")
            
            # Options validation
            if 'options' in field_config and value not in field_config['options']:
                errors.append(f"Field '{field_name}' must be one of: {field_config['options']}")
        
        # Validate optional fields if provided
        for field_name, field_config in self.optional_fields.items():
            if field_name in student_data:
                value = student_data[field_name]
                
                if field_config['type'] == int and not isinstance(value, int):
                    try:
                        student_data[field_name] = int(value)
                    except ValueError:
                        errors.append(f"Optional field '{field_name}' must be an integer")
                elif field_config['type'] == float and not isinstance(value, (int, float)):
                    try:
                        student_data[field_name] = float(value)
                    except ValueError:
                        errors.append(f"Optional field '{field_name}' must be a number")
        
        return len(errors) == 0, errors
    
    def process_student_input(self, student_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process and transform student input into model-ready format
        
        Args:
            student_data: Raw student input data
            
        Returns:
            DataFrame ready for model prediction
        """
        # Validate input first
        is_valid, errors = self.validate_input(student_data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")
        
        # Create base DataFrame
        processed_data = self._create_base_features(student_data)
        
        # Estimate missing features
        processed_data = self._estimate_missing_features(processed_data, student_data)
        
        # Apply feature engineering
        processed_data = self._apply_feature_engineering(processed_data)
        
        # Ensure all required model features are present
        processed_data = self._prepare_for_model(processed_data)
        
        return processed_data
    
    def _create_base_features(self, student_data: Dict[str, Any]) -> pd.DataFrame:
        """Create base features from student input"""
        
        # Calculate current year and years since intake
        current_year = datetime.now().year
        years_since_intake = current_year - student_data['intake_year']
        
        base_features = {
            'student_id': student_data['student_id'],
            'gender': student_data['gender'],
            'age_at_enrollment': student_data['age_at_enrollment'],
            'province': student_data['province'],
            'district': student_data['district'],
            'z_score_AL': student_data['z_score_AL'],
            'pathway': student_data['pathway'],
            'intake_year': student_data['intake_year'],
            'years_since_intake': years_since_intake,
        }
        
        return pd.DataFrame([base_features])
    
    def _estimate_missing_features(self, df: pd.DataFrame, student_data: Dict[str, Any]) -> pd.DataFrame:
        """Estimate missing features based on available information"""
        
        # Estimate GPA if not provided
        if 'current_gpa' in student_data and student_data['current_gpa'] is not None:
            df['cumulative_gpa'] = student_data['current_gpa']
        else:
            # Estimate GPA based on A/L performance and semester
            base_gpa = 2.0 + (student_data['z_score_AL'] - 1.45) * 1.5
            semester_factor = min(student_data['current_semester'] / 8.0, 1.0)
            df['cumulative_gpa'] = base_gpa * (0.9 + 0.1 * semester_factor)
            df['cumulative_gpa'] = np.clip(df['cumulative_gpa'], 2.0, 4.0)
        
        # Process internship information
        if 'completed_internships' in student_data:
            df['internship_count'] = student_data['completed_internships']
            
            if 'internship_ratings' in student_data and student_data['internship_ratings']:
                df['avg_internship_rating'] = np.mean(student_data['internship_ratings'])
            else:
                # Estimate rating based on GPA
                df['avg_internship_rating'] = min(3.0 + df['cumulative_gpa'].iloc[0] * 0.5, 5.0)
            
            if 'total_internship_months' in student_data:
                df['total_internship_days'] = student_data['total_internship_months'] * 30
                df['avg_internship_duration'] = df['total_internship_days'] / max(df['internship_count'].iloc[0], 1)
            else:
                # Estimate 3 months per internship on average
                df['avg_internship_duration'] = 90
                df['total_internship_days'] = df['internship_count'] * 90
        else:
            # Estimate based on semester progression
            semester = student_data['current_semester']
            if semester >= 6:  # Likely to have had internships by semester 6
                df['internship_count'] = max(1, semester - 5)
            else:
                df['internship_count'] = 0
            df['avg_internship_rating'] = 4.0
            df['avg_internship_duration'] = 90
            df['total_internship_days'] = df['internship_count'] * 90
        
        # Process project information
        if 'completed_projects' in student_data:
            df['project_count'] = student_data['completed_projects']
            df['completed_projects'] = student_data['completed_projects']
            
            if 'project_technologies' in student_data and student_data['project_technologies']:
                df['total_technologies'] = len(set(student_data['project_technologies']))
            else:
                # Estimate based on pathway and project count
                df['total_technologies'] = min(df['project_count'].iloc[0] * 2 + 3, 15)
        else:
            # Estimate based on semester
            semester = student_data['current_semester']
            df['project_count'] = max(1, semester // 2)
            df['completed_projects'] = df['project_count']
            df['total_technologies'] = df['project_count'] * 2 + 2
        
        df['avg_project_duration'] = 45  # Default 45 days per project
        
        # Process certifications
        if 'certifications_earned' in student_data:
            df['certification_count'] = student_data['certifications_earned']
        else:
            # Estimate based on pathway and semester
            high_cert_pathways = ['Artificial Intelligence', 'Data Science', 'Cyber Security']
            if student_data['pathway'] in high_cert_pathways:
                df['certification_count'] = max(0, student_data['current_semester'] // 3)
            else:
                df['certification_count'] = max(0, student_data['current_semester'] // 4)
        
        # Capstone information
        if 'capstone_domain' in student_data and student_data['capstone_domain']:
            df['domain'] = student_data['capstone_domain']
            df['technologies_used'] = 'Python,JavaScript,SQL'  # Default tech stack
            df['outcome'] = 'Completed'
        else:
            # Estimate based on semester
            if student_data['current_semester'] >= 7:
                df['domain'] = 'AI/ML' if student_data['pathway'] == 'Artificial Intelligence' else 'Web Development'
                df['technologies_used'] = 'Python,JavaScript,SQL'
                df['outcome'] = 'Completed' if student_data['current_semester'] >= 8 else 'In Progress'
            else:
                df['domain'] = None
                df['technologies_used'] = None
                df['outcome'] = None
        
        # Set placement status (current students are not yet placed)
        df['placed'] = False
        df['starting_salary_lkr'] = None
        df['company_name'] = None
        df['job_title'] = None
        df['employment_type'] = None
        
        return df
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as used in training"""
        
        # Use the trained feature engineer to create engineered features
        engineered_df = self.feature_engineer.engineer_features(df)
        
        return engineered_df
    
    def _prepare_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required model features are present with appropriate defaults"""
        
        # Get the features that were selected during training
        if hasattr(self.feature_engineer, 'selected_features') and self.feature_engineer.selected_features:
            required_features = self.feature_engineer.selected_features
        else:
            # Fallback to common features if selected features not available
            required_features = [
                'cumulative_gpa', 'z_score_AL', 'age_at_enrollment', 'years_since_intake',
                'internship_count', 'project_count', 'certification_count',
                'experience_score', 'high_achiever', 'has_internship'
            ]
        
        # Ensure all required features exist
        for feature in required_features:
            if feature not in df.columns:
                # Set reasonable defaults for missing features
                if 'count' in feature or 'score' in feature:
                    df[feature] = 0
                elif 'gpa' in feature.lower():
                    df[feature] = 3.0
                elif feature.startswith('has_') or feature.startswith('is_'):
                    df[feature] = 0
                else:
                    df[feature] = 0
        
        # Select only the required features
        available_features = [f for f in required_features if f in df.columns]
        model_ready_df = df[available_features].copy()
        
        # Apply the same encoding as used in training
        if hasattr(self.feature_engineer, 'label_encoders'):
            for col, encoder in self.feature_engineer.label_encoders.items():
                if col in model_ready_df.columns:
                    try:
                        model_ready_df[col] = encoder.transform(model_ready_df[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        model_ready_df[col] = 0
        
        # Handle missing values
        model_ready_df = model_ready_df.fillna(0)
        
        return model_ready_df
    
    def predict_career_outcomes(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict career outcomes for a student
        
        Args:
            student_data: Student input data
            
        Returns:
            Dictionary containing predictions and insights
        """
        try:
            # Process input data
            processed_df = self.process_student_input(student_data)
            
            # Make prediction
            salary_prediction = self.trained_model.predict(processed_df)[0]
            
            # Calculate confidence intervals (simplified approach)
            base_uncertainty = 50000  # Base uncertainty in LKR
            experience_factor = processed_df.get('experience_score', [0])[0]
            gpa_factor = processed_df.get('cumulative_gpa', [3.0])[0]
            
            # Lower uncertainty for students with more experience and higher GPA
            uncertainty = base_uncertainty * (1.2 - 0.1 * experience_factor - 0.1 * gpa_factor)
            uncertainty = max(uncertainty, 20000)  # Minimum uncertainty
            
            # Generate insights and recommendations
            insights = self._generate_insights(processed_df, student_data)
            recommendations = self._generate_recommendations(processed_df, student_data)
            
            return {
                'predicted_salary': {
                    'amount': float(salary_prediction),
                    'currency': 'LKR',
                    'confidence_interval': {
                        'lower': float(salary_prediction - uncertainty),
                        'upper': float(salary_prediction + uncertainty)
                    }
                },
                'insights': insights,
                'recommendations': recommendations,
                'student_profile': {
                    'experience_score': float(processed_df.get('experience_score', [0])[0]),
                    'academic_performance': self._categorize_gpa(processed_df.get('cumulative_gpa', [3.0])[0]),
                    'pathway': student_data['pathway'],
                    'completion_status': f"{student_data['current_semester']}/8 semesters"
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'message': 'Unable to generate prediction. Please check your input data.'
            }
    
    def _generate_insights(self, processed_df: pd.DataFrame, student_data: Dict[str, Any]) -> list[str]:
        """Generate insights about the student's profile"""
        insights = []
        
        gpa = processed_df.get('cumulative_gpa', [3.0])[0]
        experience_score = processed_df.get('experience_score', [0])[0]
        internship_count = processed_df.get('internship_count', [0])[0]
        
        # Academic insights
        if gpa >= 3.5:
            insights.append("Strong academic performance will positively impact your career prospects")
        elif gpa <= 2.5:
            insights.append("Consider focusing on improving academic performance in remaining semesters")
        
        # Experience insights
        if experience_score >= 5:
            insights.append("Excellent practical experience profile - you're well-prepared for the job market")
        elif experience_score <= 2:
            insights.append("Consider gaining more practical experience through internships and projects")
        
        # Pathway insights
        high_demand = ['Artificial Intelligence', 'Data Science', 'Cyber Security']
        if student_data['pathway'] in high_demand:
            insights.append("Your pathway is in high demand in the current job market")
        
        # Internship insights
        if internship_count == 0 and student_data['current_semester'] >= 5:
            insights.append("Consider applying for internships to gain industry experience")
        elif internship_count >= 2:
            insights.append("Multiple internships demonstrate strong industry engagement")
        
        return insights
    
    def _generate_recommendations(self, processed_df: pd.DataFrame, student_data: Dict[str, Any]) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        semester = student_data['current_semester']
        experience_score = processed_df.get('experience_score', [0])[0]
        cert_count = processed_df.get('certification_count', [0])[0]
        
        # Semester-based recommendations
        if semester <= 4:
            recommendations.append("Focus on building a strong foundation and maintaining good grades")
            recommendations.append("Start exploring internship opportunities for upcoming breaks")
        elif semester <= 6:
            recommendations.append("Apply for internships to gain practical experience")
            recommendations.append("Begin working on substantial projects in your specialization area")
        else:
            recommendations.append("Focus on completing final projects and capstone work")
            recommendations.append("Start applying for full-time positions")
        
        # Experience-based recommendations
        if experience_score < 3:
            recommendations.append("Participate in more coding competitions and hackathons")
            recommendations.append("Contribute to open-source projects to build your portfolio")
        
        # Certification recommendations
        if cert_count == 0:
            pathway_certs = {
                'Artificial Intelligence': ['AWS Machine Learning', 'Google Cloud ML', 'TensorFlow Developer'],
                'Data Science': ['Google Data Analytics', 'Microsoft Azure Data Scientist', 'Tableau Desktop'],
                'Cyber Security': ['CompTIA Security+', 'CISSP', 'CEH'],
                'Scientific Computing': ['MATLAB Certification', 'Python Scientific Computing'],
                'Standard': ['AWS Solutions Architect', 'Oracle Java', 'Microsoft Azure Fundamentals']
            }
            certs = pathway_certs.get(student_data['pathway'], ['Industry-relevant certifications'])
            recommendations.append(f"Consider pursuing certifications like: {', '.join(certs[:2])}")
        
        return recommendations
    
    def _categorize_gpa(self, gpa: float) -> str:
        """Categorize GPA performance"""
        if gpa >= 3.7:
            return "Excellent"
        elif gpa >= 3.3:
            return "Good"
        elif gpa >= 2.7:
            return "Average"
        else:
            return "Below Average"


class StudentPredictionAPI:
    """
    API wrapper for student career prediction
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the API
        
        Args:
            model_path: Path to saved model files
        """
        self.input_handler = None
        self.model_loaded = False
        
    def load_model(self, feature_engineer, trained_model):
        """
        Load trained model and feature engineer
        
        Args:
            feature_engineer: Trained FeatureEngineer instance
            trained_model: Trained CareerPredictionModel instance
        """
        self.input_handler = StudentInputHandler(feature_engineer, trained_model)
        self.model_loaded = True
        
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input form schema for UI generation"""
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        return self.input_handler.get_input_form_schema()
    
    def predict(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for student
        
        Args:
            student_data: Student input data
            
        Returns:
            Prediction results with insights and recommendations
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        return self.input_handler.predict_career_outcomes(student_data)


# Example usage function
def example_usage():
    """
    Example of how to use the StudentInputHandler
    """
    # Example student data
    sample_student_data = {
        'student_id': 'CS2021001',
        'gender': 'Male',
        'age_at_enrollment': 19,
        'province': 'Western',
        'district': 'Colombo',
        'z_score_AL': 1.85,
        'pathway': 'Artificial Intelligence',
        'intake_year': 2021,
        'current_semester': 6,
        'current_gpa': 3.2,
        'completed_internships': 1,
        'internship_ratings': [4.5],
        'total_internship_months': 3,
        'completed_projects': 4,
        'project_technologies': ['Python', 'TensorFlow', 'React', 'SQL'],
        'certifications_earned': 1
    }
    
    print("Sample Student Input:")
    print("=" * 50)
    for key, value in sample_student_data.items():
        print(f"{key}: {value}")
    
    print("\nThis data would be processed and used for prediction")
    print("The system will estimate missing features and generate predictions")

if __name__ == "__main__":
    example_usage()