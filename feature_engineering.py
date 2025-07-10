import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import VarianceThreshold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_final_gpa(gpa_df):
    """Calculate the final cumulative GPA for each student."""
    return gpa_df.groupby('student_id')['cumulative_gpa'].last().reset_index().rename(columns={'cumulative_gpa': 'final_cumulative_gpa'})

def aggregate_internships(internships_df):
    """Aggregate internship data into features."""
    internship_agg = internships_df.groupby('student_id').agg({
        'role_title': 'count',
        'performance_rating': 'mean',
        'start_date': lambda x: (pd.to_datetime(internships_df.loc[x.index, 'end_date']) - pd.to_datetime(x)).dt.days.sum()
    }).reset_index()
    internship_agg.columns = ['student_id', 'num_internships', 'avg_performance_rating', 'total_internship_duration']
    return internship_agg

def aggregate_projects(projects_df):
    """Aggregate project data into features."""
    project_agg = projects_df.groupby('student_id').agg({
        'project_id': 'count',
        'outcome': lambda x: (x == 'Completed').sum(),
        'start_date': lambda x: (pd.to_datetime(projects_df.loc[x.index, 'end_date']) - pd.to_datetime(x)).dt.days.sum()
    }).reset_index()
    project_agg['has_deployed_project'] = projects_df.groupby('student_id')['outcome'].apply(
        lambda x: (x == 'Deployed').any()).astype(int).reset_index()['outcome']
    project_agg.columns = ['student_id', 'num_projects', 'num_completed_projects', 'total_project_duration', 'has_deployed_project']
    return project_agg

def process_certifications(certifications_df):
    """Process certifications into counts and binary features."""
    cert_agg = certifications_df.groupby('student_id').size().reset_index(name='num_certifications')
    cert_dummies = pd.get_dummies(certifications_df['certificate_name']).groupby(certifications_df['student_id']).sum().reset_index()
    cert_dummies.columns = ['student_id'] + [f'has_{col.replace(" ", "_")}' for col in cert_dummies.columns[1:]]
    return cert_agg.merge(cert_dummies, on='student_id', how='left').fillna(0)

def aggregate_courses(courses_df):
    """Aggregate course data into features."""
    course_agg = courses_df.groupby('student_id').agg({
        'credits': 'sum',
        'grade_points': 'mean'
    }).reset_index()
    course_agg.columns = ['student_id', 'total_credits', 'avg_grade_points']
    return course_agg

def validate_dataframe(df, required_columns, name):
    """Validate required columns in DataFrame."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {name}: {missing_cols}")

def engineer_features(students, gpa, internships, projects, capstone, certifications, placement, courses, industry, industry_trends):
    """Engineer all features for the model."""
    try:
        # Validate inputs
        validate_dataframe(students, ['student_id', 'gender', 'district', 'province', 'z_score_AL', 'intake_year', 'pathway', 'years_since_intake', 'is_graduated'], 'students')
        validate_dataframe(gpa, ['student_id', 'cumulative_gpa'], 'gpa')
        validate_dataframe(internships, ['student_id', 'role_title', 'performance_rating', 'start_date', 'end_date'], 'internships')
        validate_dataframe(projects, ['student_id', 'project_id', 'outcome', 'start_date', 'end_date'], 'projects')
        validate_dataframe(capstone, ['student_id', 'domain', 'outcome'], 'capstone')
        validate_dataframe(certifications, ['student_id', 'certificate_name'], 'certifications')
        validate_dataframe(placement, ['student_id', 'placed', 'starting_salary_lkr'], 'placement')
        validate_dataframe(courses, ['student_id', 'credits', 'grade_points'], 'courses')
        validate_dataframe(industry, ['salary_benchmark_monthly_lkr'], 'industry')

        # Filter for placed students
        placement = placement[placement['placed'] == True]
        if placement['starting_salary_lkr'].isnull().any() or (placement['starting_salary_lkr'] <= 0).any():
            raise ValueError("Invalid starting_salary_lkr values (NaN or non-positive)")

        # Merge student features
        data = placement[['student_id', 'starting_salary_lkr']].merge(
            students[['student_id', 'gender', 'province', 'pathway', 'z_score_AL', 'intake_year', 'years_since_intake', 'is_graduated']], on='student_id')

        # Add final GPA
        final_gpa = calculate_final_gpa(gpa)
        data = data.merge(final_gpa, on='student_id')

        # Add capstone data
        data = data.merge(capstone[['student_id', 'domain', 'outcome']], on='student_id', how='left')
        data['outcome'] = data['outcome'].fillna('None')

        # Add internship features
        internship_features = aggregate_internships(internships)
        data = data.merge(internship_features, on='student_id', how='left').fillna(
            {'num_internships': 0, 'avg_performance_rating': 0, 'total_internship_duration': 0})
        data['has_internship'] = (data['num_internships'] > 0).astype(int)
        data['total_internship_duration'] = data['total_internship_duration'].clip(lower=0)

        # Add project features
        project_features = aggregate_projects(projects)
        data = data.merge(project_features, on='student_id', how='left').fillna(
            {'num_projects': 0, 'num_completed_projects': 0, 'total_project_duration': 0, 'has_deployed_project': 0})
        data['total_project_duration'] = data['total_project_duration'].clip(lower=0)

        # Add certification features
        cert_features = process_certifications(certifications)
        data = data.merge(cert_features, on='student_id', how='left').fillna(0)

        # Add course features
        course_features = aggregate_courses(courses)
        data = data.merge(course_features, on='student_id', how='left').fillna(
            {'total_credits': 0, 'avg_grade_points': 0})

        # Add industry benchmark
        data['industry_salary_benchmark'] = industry['salary_benchmark_monthly_lkr'].iloc[0] * 12  # Convert to annual

        # Engineer additional features
        data['internship_quality_score'] = data['num_internships'] * data['avg_performance_rating']
        data['project_impact_score'] = data['num_completed_projects'] + 2 * data['has_deployed_project']
        data['gpa_z_score_interaction'] = data['final_cumulative_gpa'] * data['z_score_AL']
        data['is_graduated'] = data['is_graduated'].astype(int)

        # Define feature categories
        categorical_features = ['gender', 'province', 'pathway', 'domain', 'outcome']
        numerical_features = ['z_score_AL', 'years_since_intake', 'final_cumulative_gpa', 'num_internships',
                             'total_internship_duration', 'avg_performance_rating', 'has_internship',
                             'num_projects', 'num_completed_projects', 'total_project_duration',
                             'has_deployed_project', 'num_certifications', 'internship_quality_score',
                             'project_impact_score', 'total_credits', 'avg_grade_points',
                             'industry_salary_benchmark', 'gpa_z_score_interaction']

        # Add certification-related has_ columns, excluding duplicates
        has_columns = [col for col in data.columns if col.startswith('has_') and col not in numerical_features]
        numerical_features.extend(has_columns)

        # Feature selection
        selector = VarianceThreshold(threshold=0.01)
        selected_features = selector.fit_transform(data[numerical_features])
        numerical_features = [numerical_features[i] for i in selector.get_support(indices=True)]
        data[numerical_features] = selected_features

        logging.info("Feature engineering completed: %d samples, %d features", len(data), len(categorical_features + numerical_features))
        return data, categorical_features, numerical_features
    except Exception as e:
        logging.error("Error in feature_engineering: %s", str(e))
        raise