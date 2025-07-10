import pandas as pd
import os
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_columns(df, required_cols, name):
    """Validate that required columns exist in the DataFrame."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {name}: {missing_cols}")

def load_data(config_path='config.yaml'):
    """Load all datasets from CSV files."""
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        files = config['data_paths']
        
        datasets = {}
        required_columns = {
            'students': ['student_id', 'gender', 'district', 'province', 'age_at_enrollment', 'z_score_AL', 'intake_year', 'pathway', 'years_since_intake', 'is_graduated'],
            'gpa': ['student_id', 'semester_number', 'cumulative_gpa'],
            'internships': ['student_id', 'employer', 'role_title', 'start_date', 'end_date', 'performance_rating'],
            'projects': ['project_id', 'student_id', 'project_title', 'technologies_used', 'start_date', 'end_date', 'outcome'],
            'capstone': ['student_id', 'pathway', 'domain', 'technologies_used', 'outcome'],
            'certifications': ['student_id', 'certificate_name', 'date_earned', 'type'],
            'placement': ['student_id', 'placed', 'starting_salary_lkr'],
            'courses': ['student_id', 'semester_number', 'term_year', 'course_code', 'credits', 'grade_points'],
            'industry': ['sector', 'avg_hiring_rate', 'salary_benchmark_monthly_lkr'],
            'industry_trends': ['role_title', 'demand_level']
        }
        
        for name, file in files.items():
            if not os.path.exists(file):
                raise FileNotFoundError(f"Dataset file {file} not found")
            datasets[name] = pd.read_csv(file)
            logging.info("Loaded %s: %d records", name, len(datasets[name]))
            if datasets[name].empty:
                raise ValueError(f"Dataset {name} is empty")
            validate_columns(datasets[name], required_columns[name], name)
        
        # Type checking for critical columns
        datasets['placement']['starting_salary_lkr'] = pd.to_numeric(datasets['placement']['starting_salary_lkr'], errors='coerce')
        datasets['gpa']['cumulative_gpa'] = pd.to_numeric(datasets['gpa']['cumulative_gpa'], errors='coerce')
        
        return (datasets['students'], datasets['gpa'], datasets['internships'],
                datasets['projects'], datasets['capstone'], datasets['certifications'],
                datasets['placement'], datasets['courses'], datasets['industry'],
                datasets['industry_trends'])
    except Exception as e:
        logging.error("Failed to load datasets: %s", str(e))
        raise