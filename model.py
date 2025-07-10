import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_models(categorical_features, numerical_features):
    """Build multiple regression models for comparison."""
    try:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numerical_features)
            ])
        
        models = {
            'XGBoost': Pipeline([
                ('preprocessor', preprocessor),
                ('selector', SelectFromModel(Ridge(), threshold='median')),
                ('regressor', XGBRegressor(n_estimators=100, max_depth=3, random_state=42))
            ]),
            'Ridge': Pipeline([
                ('preprocessor', preprocessor),
                ('selector', SelectFromModel(Ridge(), threshold='median')),
                ('regressor', Ridge(alpha=1.0, random_state=42))
            ]),
            'RandomForest': Pipeline([
                ('preprocessor', preprocessor),
                ('selector', SelectFromModel(Ridge(), threshold='median')),
                ('regressor', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
            ])
        }
        logging.info("Built models: %s", list(models.keys()))
        return models
    except Exception as e:
        logging.error("Error building models: %s", str(e))
        raise

def tune_model(model, X, y):
    """Tune hyperparameters for the best model using GridSearchCV."""
    try:
        param_grid = {}
        if isinstance(model.named_steps['regressor'], XGBRegressor):
            param_grid = {
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [2, 3],
                'regressor__learning_rate': [0.01, 0.1]
            }
        elif isinstance(model.named_steps['regressor'], Ridge):
            param_grid = {
                'regressor__alpha': [0.1, 1.0, 10.0]
            }
        elif isinstance(model.named_steps['regressor'], RandomForestRegressor):
            param_grid = {
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [3, 5]
            }
        
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid_search.fit(X, y)
            logging.info("Best parameters for %s: %s", type(model.named_steps['regressor']).__name__, grid_search.best_params_)
            return grid_search.best_estimator_
        return model
    except Exception as e:
        logging.error("Error tuning model: %s", str(e))
        raise

def train_and_evaluate_model(models, X, y):
    """Train and evaluate multiple models using cross-validation."""
    try:
        if len(X) != len(y):
            raise ValueError(f"Mismatch between X ({len(X)}) and y ({len(y)}) lengths")
        
        best_model = None
        best_mae = float('inf')
        results = {}
        
        for name, model in models.items():
            logging.debug("Evaluating model: %s", name)
            # 3-fold CV
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
            mae = -np.mean(scores)
            rmse = np.sqrt(-cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error').mean())
            r2 = cross_val_score(model, X, y, cv=3, scoring='r2').mean()
            mape = -cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_percentage_error').mean()
            
            results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
            logging.info("Model: %s, MAE: %.2f LKR, RMSE: %.2f LKR, R2: %.4f, MAPE: %.4f", name, mae, rmse, r2, mape)
            
            if mae < best_mae:
                best_mae = mae
                best_model = model
                logging.debug("New best model: %s with MAE: %.2f", name, mae)
        
        # Tune the best model
        logging.info("Tuning best model: %s", type(best_model.named_steps['regressor']).__name__)
        best_model = tune_model(best_model, X, y)
        
        # Fit the best model
        best_model.fit(X, y)
        
        # Predictions and residuals
        y_pred = best_model.predict(X)
        residuals = y - y_pred
        pd.DataFrame({'Actual': y, 'Predicted': y_pred, 'Residual': residuals}).to_csv('predictions.csv', index=False)
        
        # Plot predicted vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('Actual Salary (LKR)')
        plt.ylabel('Predicted Salary (LKR)')
        plt.title('Predicted vs Actual Salaries')
        plt.savefig('predicted_vs_actual.png')
        plt.close()
        
        # Feature importance
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        selected_features = best_model.named_steps['selector'].get_support()
        feature_names = feature_names[selected_features]
        
        try:
            if isinstance(best_model.named_steps['regressor'], (XGBRegressor, RandomForestRegressor)):
                importances = best_model.named_steps['regressor'].feature_importances_
                logging.info("Using feature_importances_ for tree-based model")
            elif isinstance(best_model.named_steps['regressor'], Ridge):
                importances = np.abs(best_model.named_steps['regressor'].coef_)
                importances = importances / importances.sum()  # Normalize
                logging.info("Using normalized absolute coefficients for Ridge model")
            else:
                logging.warning("No feature importance available for %s", type(best_model.named_steps['regressor']).__name__)
                importances = None
            
            if importances is not None:
                feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
                feature_importance.to_csv('feature_importances.csv', index=False)
                logging.info("Top 10 Feature Importances:\n%s", feature_importance.head(10))
        except Exception as e:
            logging.error("Error extracting feature importances: %s", str(e))
        
        # Log best model
        best_model_name = min(results, key=lambda x: results[x]['MAE'])
        logging.info("Best Model: %s with MAE: %.2f LKR", best_model_name, results[best_model_name]['MAE'])
        
        return best_model, results
    except Exception as e:
        logging.error("Error in train_and_evaluate_model: %s", str(e))
        raise