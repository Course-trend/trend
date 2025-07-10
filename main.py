from data_loader import load_data
from feature_engineering import engineer_features
from model import build_models, train_and_evaluate_model
import joblib
import json
import logging
import argparse
import time
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(config_path):
    """Main function to run the prediction system."""
    start_time = time.time()
    try:
        # Load data
        students, gpa, internships, projects, capstone, certifications, placement, courses, industry, industry_trends = load_data(config_path)
        
        # Engineer features
        data, categorical_features, numerical_features = engineer_features(
            students, gpa, internships, projects, capstone, certifications, placement, courses, industry, industry_trends)
        
        # Save dataset summary
        data.describe().to_csv('dataset_summary.csv')
        
        # Prepare features (X) and target (y)
        X = data[categorical_features + numerical_features]
        y = data['starting_salary_lkr']
        logging.info("Prepared data: %d samples, %d features", len(X), len(X.columns))
        
        # Build and train models
        models = build_models(categorical_features, numerical_features)
        trained_model, results = train_and_evaluate_model(models, X, y)
        
        # Save model and results
        joblib.dump(trained_model, 'best_model.pkl')
        with open('model_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        logging.info("Saved model and results")
        
        elapsed_time = time.time() - start_time
        logging.info("Execution completed in %.2f seconds", elapsed_time)
    except Exception as e:
        logging.error("Error in main: %s", str(e))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Salary Prediction System")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)