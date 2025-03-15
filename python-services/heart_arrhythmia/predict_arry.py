import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
import joblib


df = pd.read_csv('heart.csv')

# Exploratory Data Analysis
def perform_eda(df):
    print("Dataset shape:", df.shape)
    print("\nData overview:")
    print(df.describe())
    
    print("\nClass distribution:")
    print(df['ARRHYTHMIA'].value_counts(normalize=True) * 100)
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    
  
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(df.columns[:-1]):
        plt.subplot(2, 3, i+1)
        sns.histplot(data=df, x=feature, hue='ARRHYTHMIA', kde=True, element='step')
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    
    return

# Split the data
def prepare_data(df):
    # Features and target
    X = df.drop('ARRHYTHMIA', axis=1)
    y = df['ARRHYTHMIA']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# Model selection and evaluation
def evaluate_models(X_train, X_test, y_train, y_test):
    # Create pipelines for different models
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(random_state=42))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(random_state=42, probability=True))
        ])
    }
    
    # Train and evaluate each model
    results = {}
    cv_scores = {}
    
    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        
        # Cross-validation
        cv_score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc').mean()
        cv_scores[name] = cv_score
        
        # Test evaluation
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:,1]
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        print(f"{name} CV ROC-AUC: {cv_score:.4f}")
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    # Find best model
    best_model_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest model: {best_model_name} with ROC-AUC: {cv_scores[best_model_name]:.4f}")
    
    return results, best_model_name

# Hyperparameter tuning for the best model
def tune_model(best_model_name, X_train, y_train):
    param_grids = {
        'Logistic Regression': {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear', 'saga']
        },
        'Random Forest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        },
        'SVM': {
            'model__C': [0.1, 1, 10, 100],
            'model__gamma': ['scale', 'auto', 0.01, 0.1],
            'model__kernel': ['rbf', 'linear']
        }
    }
    
    # Create base pipeline
    if best_model_name == 'Logistic Regression':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ])
    elif best_model_name == 'Random Forest':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=42))
        ])
    elif best_model_name == 'Gradient Boosting':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(random_state=42))
        ])
    else:  # SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(random_state=42, probability=True))
        ])
    
    # Grid search
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grids[best_model_name], 
        cv=5, 
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Visualize model performance
def visualize_model_performance(results, best_model_name, X_test, y_test):
    # ROC curve for all models
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    
    # Precision-Recall curve for best model
    best_result = results[best_model_name]
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, best_result['y_prob'])
    plt.plot(recall, precision, label=f'{best_model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('precision_recall_curve.png')
    
    # Confusion matrix for best model
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.savefig('confusion_matrix.png')
    
    # Feature importance (if applicable)
    best_pipeline = best_result['pipeline']
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        feature_names = X_test.columns
        importances = best_pipeline.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importances - {best_model_name}')
        plt.bar(range(X_test.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
    
    return

# Main function to orchestrate the entire process
def main():
    print("Loading and exploring the dataset...")
    df = pd.read_csv('arrhythmia_dataset.csv')
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        df = df.dropna()  # Simple approach - drop rows with missing values
    
    # Perform EDA
    perform_eda(df)
    
    # Split the data
    print("\nPreparing data for modeling...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Evaluate different models
    print("\nEvaluating different models...")
    results, best_model_name = evaluate_models(X_train, X_test, y_train, y_test)
    
    # Tune the best model
    best_model = tune_model(best_model_name, X_train, y_train)
    
    # Final evaluation on test data
    print("\nFinal evaluation of the best model...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]
    
    print("\nBest model performance:")
    print(classification_report(y_test, y_pred))
    
    # Update results with the tuned model
    results[best_model_name] = {
        'pipeline': best_model,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    # Visualize model performance
    visualize_model_performance(results, best_model_name, X_test, y_test)
    
    # Save the best model
    joblib.dump(best_model, 'arrhythmia_prediction_model.pkl')
    print("\nBest model saved as 'arrhythmia_prediction_model.pkl'")

    # Create a function for making predictions
    def predict_arrhythmia_risk(heart_rate, body_fat, basal_energy, total_calories, weight, steps):
        data = [[heart_rate, body_fat, basal_energy, total_calories, weight, steps]]
        df_input = pd.DataFrame(data, columns=['HEART_RATE', 'BODY_FAT_PERCENTAGE', 'BASAL_ENERGY_BURNED', 
                                              'TOTAL_CALORIES_BURNED', 'WEIGHT', 'STEPS'])
        
        # Predict probability
        risk_prob = best_model.predict_proba(df_input)[0][1]
        
        # Return probability and risk category
        risk_category = "High" if risk_prob > 0.7 else "Medium" if risk_prob > 0.3 else "Low"
        
        return risk_prob, risk_category

    # Example usage
    print("\nExample prediction:")
    heart_rate = 85
    body_fat = 29
    basal_energy = 1450
    total_calories = 2100
    weight = 82
    steps = 5500
    
    risk_prob, risk_category = predict_arrhythmia_risk(heart_rate, body_fat, basal_energy, total_calories, weight, steps)
    print(f"Person with HR={heart_rate}, BF%={body_fat}, BE={basal_energy}, TC={total_calories}, W={weight}, Steps={steps}")
    print(f"Arrhythmia Risk Probability: {risk_prob:.4f} ({risk_category} Risk)")

if __name__ == "__main__":
    main()