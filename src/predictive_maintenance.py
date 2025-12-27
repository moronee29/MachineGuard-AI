"""
Predictive Maintenance System - Main Script
Standalone Python application for industrial machine failure prediction.
Run this script to generate all outputs without Jupyter.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from datetime import datetime

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier
import joblib

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def print_header(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")

def create_directories():
    """Create necessary output directories."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    print_header("STEP 1: DATA LOADING & PREPROCESSING")
    
    # Load dataset - try config first, then fallback
    print("Loading dataset...")
    try:
        import config
        data_file = config.PATHS['data_file']
    except:
        data_file = 'data/generated/predictive_maintenance_data.csv'
    
    df = pd.read_csv(data_file)
    print(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    
    # Data quality check
    print("Data Quality Check:")
    print(f"  - Total samples: {len(df):,}")
    print(f"  - Machines: {df['machine_id'].nunique()}")
    print(f"  - Failures: {df['failure'].sum()} ({df['failure'].mean()*100:.2f}%)")
    print(f"  - Normal: {(df['failure']==0).sum()} ({(df['failure']==0).mean()*100:.2f}%)")
    
    # Handle missing values
    print("\nHandling missing values...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('failure')
    
    for col in numeric_cols:
        df[col] = df.groupby('machine_id')[col].fillna(method='ffill').fillna(method='bfill')
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    print(f"✓ Missing values handled (remaining: {df.isnull().sum().sum()})")
    
    return df

def perform_eda(df):
    """Perform exploratory data analysis."""
    print_header("STEP 2: EXPLORATORY DATA ANALYSIS")
    
    feature_cols = [
        'vibration', 'temperature', 'pressure', 'runtime_hours',
        'voltage', 'current', 'acoustic_emission', 'rotation_speed',
        'torque', 'power_consumption'
    ]
    
    # Statistical summary
    print("Statistical Summary:")
    print(df[feature_cols].describe().round(2))
    
    # Target distribution visualization
    print("\nCreating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    failure_counts = df['failure'].value_counts()
    axes[0].bar(['Normal (0)', 'Failure (1)'], failure_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Target Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(failure_counts.values):
        axes[0].text(i, v + 50, f"{v:,}\n({v/len(df)*100:.1f}%)", 
                    ha='center', fontweight='bold', fontsize=11)
    
    colors = ['#2ecc71', '#e74c3c']
    axes[1].pie(failure_counts.values, labels=['Normal', 'Failure'], autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Failure Rate', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/target_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/target_distribution.png")
    plt.close()
    
    # Correlation matrix
    print("Generating correlation matrix...")
    plt.figure(figsize=(14, 12))
    correlation_matrix = df[feature_cols + ['failure']].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/correlation_heatmap.png")
    plt.close()
    
    print("\nTop correlations with failure:")
    failure_corr = correlation_matrix['failure'].drop('failure').sort_values(ascending=False)
    print(failure_corr)
    
    return feature_cols

def engineer_features(df, feature_cols):
    """Create engineered features."""
    print_header("STEP 3: FEATURE ENGINEERING")
    
    df_sorted = df.sort_values(['machine_id', 'timestamp']).copy()
    
    # Rolling average features
    print("Creating rolling average features...")
    rolling_features = []
    for col in ['vibration', 'temperature', 'pressure']:
        rolling_col = f"{col}_rolling_avg"
        df_sorted[rolling_col] = df_sorted.groupby('machine_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        rolling_features.append(rolling_col)
    
    # Statistical features
    print("Creating statistical features...")
    stat_features = []
    for col in ['vibration', 'current']:
        std_col = f"{col}_rolling_std"
        df_sorted[std_col] = df_sorted.groupby('machine_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
        )
        stat_features.append(std_col)
    
    # Derived features
    df_sorted['power_efficiency'] = df_sorted['power_consumption'] / (df_sorted['torque'] + 1)
    df_sorted['temp_vib_interaction'] = df_sorted['temperature'] * df_sorted['vibration']
    stat_features.extend(['power_efficiency', 'temp_vib_interaction'])
    
    all_features = feature_cols + rolling_features + stat_features
    print(f"✓ Total features created: {len(all_features)}")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    X_temp = df_sorted[all_features].copy()
    y_temp = df_sorted['failure'].copy()
    
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_temp.fit(X_temp, y_temp)
    
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'].values, color='#3498db')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance Score', fontweight='bold', fontsize=12)
    plt.title('Top 15 Feature Importance', fontweight='bold', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: outputs/feature_importance.png")
    plt.close()
    
    # Select top features
    feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
    selected_features = feature_importance[feature_importance['cumulative_importance'] <= 0.95]['feature'].tolist()
    
    if len(selected_features) < 10:
        selected_features = feature_importance.head(10)['feature'].tolist()
    
    print(f"\n✓ Selected {len(selected_features)} features for modeling")
    
    return df_sorted, selected_features

def train_models(df, selected_features):
    """Train and evaluate ML models."""
    print_header("STEP 4: PREDICTIVE MODELING")
    
    # Prepare data
    X = df[selected_features].copy()
    y = df['failure'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}\n")
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n📊 Random Forest Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_rf):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_rf_proba):.4f}")
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train, verbose=False)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    y_pred_xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n📊 XGBoost Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_xgb):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_xgb):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_xgb_proba):.4f}")
    
    # Confusion matrices
    print("\nGenerating confusion matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
               xticklabels=['Normal', 'Failure'],
               yticklabels=['Normal', 'Failure'])
    axes[0].set_title('Random Forest\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontweight='bold')
    
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
               xticklabels=['Normal', 'Failure'],
               yticklabels=['Normal', 'Failure'])
    axes[1].set_title('XGBoost\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/confusion_matrices.png")
    plt.close()
    
    # ROC curves
    print("Generating ROC curves...")
    plt.figure(figsize=(10, 8))
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_proba)
    auc_rf = roc_auc_score(y_test, y_pred_rf_proba)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})', linewidth=2.5)
    
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba)
    auc_xgb = roc_auc_score(y_test, y_pred_xgb_proba)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.4f})', linewidth=2.5)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)
    
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves: Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/roc_curves.png")
    plt.close()
    
    # Model selection
    f1_rf = f1_score(y_test, y_pred_rf)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    
    if f1_xgb > f1_rf:
        best_model = xgb_model
        best_model_name = "XGBoost"
        best_predictions = y_pred_xgb
        best_probabilities = y_pred_xgb_proba
    else:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_predictions = y_pred_rf
        best_probabilities = y_pred_rf_proba
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    
    # Save models
    print("\nSaving models...")
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    with open('models/selected_features.txt', 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    
    print("✓ Models saved to models/ directory")
    
    return best_model_name, best_predictions, best_probabilities, y_test, X_test.index

def create_dashboard(best_model_name, best_predictions, best_probabilities, y_test, test_indices, df):
    """Create predictions and interactive dashboards."""
    print_header("STEP 5: PREDICTIONS & DASHBOARDS")
    
    # Create prediction results
    test_machine_ids = df.loc[test_indices, 'machine_id'].values
    
    results_df = pd.DataFrame({
        'Machine_ID': test_machine_ids,
        'Actual_Status': ['Failure' if x == 1 else 'Normal' for x in y_test],
        'Predicted_Status': ['Failure' if x == 1 else 'Normal' for x in best_predictions],
        'Failure_Probability': best_probabilities,
        'Risk_Level': pd.cut(best_probabilities, 
                            bins=[0, 0.3, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High'])
    })
    
    results_df['Correct_Prediction'] = results_df['Actual_Status'] == results_df['Predicted_Status']
    
    # Identify high-risk machines
    high_risk_threshold = 0.7
    high_risk_machines = results_df[results_df['Failure_Probability'] > high_risk_threshold].copy()
    high_risk_machines = high_risk_machines.sort_values('Failure_Probability', ascending=False)
    
    print(f"Predictions generated for {len(results_df):,} machines")
    print(f"High-risk machines (>{high_risk_threshold*100:.0f}%): {len(high_risk_machines)}")
    
    # Save results
    results_df.to_csv('results/prediction_results.csv', index=False)
    high_risk_machines.to_csv('results/high_risk_machines.csv', index=False)
    print("\n✓ Saved: results/prediction_results.csv")
    print("✓ Saved: results/high_risk_machines.csv")
    
    # Modern color palette
    colors = {
        'primary': '#667eea',
        'success': '#11998e',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#3498db'
    }
    
    # Config to disable interactive features
    plot_config = {
        'displayModeBar': False,  # Hide toolbar
        'staticPlot': True,  # Disable ALL interactions (no hover)
        'displaylogo': False
    }
    
    # =================================================================
    # DASHBOARD 1: MODEL PERFORMANCE (for data scientists)
    # =================================================================
    print("\nCreating Model Performance dashboard...")
    
    fig_model = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '📊 Confusion Matrix',
            '📈 Model Metrics Comparison',
            '⚖️ Actual vs Predicted',
            '📉 Probability Distribution'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'histogram'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Confusion Matrix as bars
    confusion_data = pd.crosstab(results_df['Actual_Status'], results_df['Predicted_Status'])
    fig_model.add_trace(
        go.Bar(
            name='Predicted Normal',
            x=confusion_data.index,
            y=confusion_data.get('Normal', [0, 0]),
            marker_color=colors['success'],
            text=confusion_data.get('Normal', [0, 0]),
            textposition='outside',
            textfont=dict(size=16, color='white')
        ),
        row=1, col=1
    )
    fig_model.add_trace(
        go.Bar(
            name='Predicted Failure',
            x=confusion_data.index,
            y=confusion_data.get('Failure', [0, 0]),
            marker_color=colors['danger'],
            text=confusion_data.get('Failure', [0, 0]),
            textposition='outside',
            textfont=dict(size=16, color='white')
        ),
        row=1, col=1
    )
    
    # 2. Model Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, best_predictions),
        'Precision': precision_score(y_test, best_predictions),
        'Recall': recall_score(y_test, best_predictions),
        'F1-Score': f1_score(y_test, best_predictions)
    }
    fig_model.add_trace(
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker=dict(color=colors['info'], line=dict(color='rgba(0,0,0,0.2)', width=2)),
            text=[f"{x:.3f}" for x in metrics.values()],
            textposition='outside',
            textfont=dict(size=16, color='white')
        ),
        row=1, col=2
    )
    
    # 3. Prediction Distribution
    pred_counts = results_df['Predicted_Status'].value_counts()
    fig_model.add_trace(
        go.Bar(
            x=pred_counts.index,
            y=pred_counts.values,
            marker=dict(color=[colors['success'], colors['danger']]),
            text=pred_counts.values,
            textposition='outside',
            textfont=dict(size=16, color='white')
        ),
        row=2, col=1
    )
    
    # 4. Probability Distribution
    fig_model.add_trace(
        go.Histogram(
            x=results_df['Failure_Probability'],
            nbinsx=30,
            marker=dict(color=colors['primary'], line=dict(color='rgba(0,0,0,0.3)', width=1))
        ),
        row=2, col=2
    )
    
    fig_model.update_layout(
        title=dict(
            text=f"🔬 Model Performance Analysis - {best_model_name}",
            font=dict(size=32, color='white', family='Arial Black'),
            x=0.5, xanchor='center'
        ),
        showlegend=True,
        legend=dict(font=dict(size=14, color='white'), bgcolor='rgba(0,0,0,0.3)'),
        height=900,
        paper_bgcolor='#0f0f1e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white', size=12)
    )
    
    fig_model.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
    fig_model.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
    
    for annotation in fig_model['layout']['annotations']:
        annotation['font'] = dict(size=18, color='white', family='Arial Black')
    
    model_html = fig_model.to_html(config=plot_config, include_plotlyjs='cdn')
    model_html = add_custom_css(model_html)
    
    with open(os.path.join('dashboards', 'model_performance.html'), 'w') as f:
        f.write(model_html)
    print("✓ Saved: dashboards/model_performance.html")
    
    # =================================================================
    # DASHBOARD 2: PRODUCTION MONITORING (for operations)
    # =================================================================
    print("Creating Production Monitoring dashboard...")
    
    fig_prod = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '🎯 Risk Level Distribution',
            '⚠️ Top 15 High-Risk Machines',
            '📊 Prediction Summary',
            '💡 Maintenance Priorities'
        ),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'table'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Risk Pie Chart
    risk_counts = results_df['Risk_Level'].value_counts()
    fig_prod.add_trace(
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(
                colors=[colors['success'], colors['warning'], colors['danger']],
                line=dict(color='#000', width=3)
            ),
            textinfo='label+percent+value',
            textfont=dict(size=14, color='white'),
            hole=0.4
        ),
        row=1, col=1
    )
    
    # 2. Top High-Risk Machines
    top_risk = high_risk_machines.head(15) if len(high_risk_machines) > 0 else pd.DataFrame({'Machine_ID': ['None'], 'Failure_Probability': [0]})
    fig_prod.add_trace(
        go.Bar(
            y=top_risk['Machine_ID'],
            x=top_risk['Failure_Probability'],
            orientation='h',
            marker=dict(
                color=top_risk['Failure_Probability'],
                colorscale=[[0, colors['warning']], [1, colors['danger']]],
                showscale=True,
                colorbar=dict(title="Risk Level", titlefont=dict(color='white', size=14), tickfont=dict(color='white'))
            ),
            text=[f"{x:.1%}" for x in top_risk['Failure_Probability']],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ),
        row=1, col=2
    )
    
    # 3. Summary Stats
    summary_data = {
        'Total Machines': [len(results_df)],
        'High Risk': [len(high_risk_machines)],
        'Medium Risk': [(results_df['Risk_Level'] == 'Medium').sum()],
        'Low Risk': [(results_df['Risk_Level'] == 'Low').sum()]
    }
    fig_prod.add_trace(
        go.Bar(
            x=list(summary_data.keys()),
            y=[v[0] for v in summary_data.values()],
            marker=dict(color=[colors['info'], colors['danger'], colors['warning'], colors['success']]),
            text=[v[0] for v in summary_data.values()],
            textposition='outside',
            textfont=dict(size=16, color='white')
        ),
        row=2, col=1
    )
    
    # 4. Priority Table
    if len(high_risk_machines) > 0:
        table_data = high_risk_machines.head(10)[['Machine_ID', 'Failure_Probability', 'Risk_Level']]
    else:
        table_data = pd.DataFrame({'Machine_ID': ['No high-risk machines'], 'Failure_Probability': ['N/A'], 'Risk_Level': ['✓ All Safe']})
    
    fig_prod.add_trace(
        go.Table(
            header=dict(
                values=['<b>Machine</b>', '<b>Risk %</b>', '<b>Priority</b>'],
                fill_color=colors['danger'],
                font=dict(color='white', size=14),
                align='center'
            ),
            cells=dict(
                values=[
                    table_data['Machine_ID'].values,
                    [f"{x:.1%}" if isinstance(x, float) else str(x) for x in table_data['Failure_Probability'].values],
                    table_data['Risk_Level'].values
                ],
                fill_color=[['#1a1a2e', '#2a2a3e'] * 10],
                font=dict(color='white', size=13),
                align='center',
                height=30
            )
        ),
        row=2, col=2
    )
    
    fig_prod.update_layout(
        title=dict(
            text="🏭 Production Monitoring Dashboard",
            font=dict(size=32, color='white', family='Arial Black'),
            x=0.5, xanchor='center'
        ),
        showlegend=False,
        height=900,
        paper_bgcolor='#0f0f1e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white', size=12)
    )
    
    fig_prod.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
    fig_prod.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
    
    for annotation in fig_prod['layout']['annotations']:
        annotation['font'] = dict(size=18, color='white', family='Arial Black')
    
    prod_html = fig_prod.to_html(config=plot_config, include_plotlyjs='cdn')
    prod_html = add_custom_css(prod_html)
    
    with open(os.path.join('dashboards', 'production_dashboard.html'), 'w') as f:
        f.write(prod_html)
    print("✓ Saved: dashboards/production_dashboard.html")
    
    return results_df, high_risk_machines

def add_custom_css(html_content):
    """Add custom CSS for premium dark theme."""
    custom_css = """
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
        }
        .plotly-graph-div {
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            backdrop-filter: blur(10px);
            background: rgba(26, 26, 46, 0.8);
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
    </style>
    """
    return html_content.replace('</head>', custom_css + '</head>')



def print_summary(results_df, high_risk_machines):
    """Print final summary."""
    print_header("PROJECT COMPLETION SUMMARY")
    
    print("✅ ALL TASKS COMPLETED SUCCESSFULLY\n")
    
    print("📁 Generated Files:")
    print("  Models:")
    print("    - models/random_forest_model.pkl")
    print("    - models/xgboost_model.pkl")
    print("    - models/scaler.pkl")
    print("    - models/selected_features.txt")
    print("\n  Visualizations:")
    print("    - outputs/target_distribution.png")
    print("    - outputs/correlation_heatmap.png")
    print("    - outputs/feature_importance.png")
    print("    - outputs/confusion_matrices.png")
    print("    - outputs/roc_curves.png")
    print("\n  Results:")
    print("    - results/prediction_results.csv")
    print("    - results/high_risk_machines.csv")
    print("    - model_performance.html (ML Metrics Dashboard)")
    print("    - production_dashboard.html (Operations Dashboard)")
    
    print("\n📊 Predictions Summary:")
    print(f"  - Total machines analyzed: {len(results_df):,}")
    print(f"  - High-risk machines: {len(high_risk_machines)}")
    print(f"  - Medium-risk: {(results_df['Risk_Level'] == 'Medium').sum()}")
    print(f"  - Low-risk: {(results_df['Risk_Level'] == 'Low').sum()}")
    
    print("\n🚀 Next Steps:")
    print("  1. Open model_performance.html - for ML metrics & analysis")
    print("  2. Open production_dashboard.html - for operational monitoring")
    print("  3. Review high_risk_machines.csv for maintenance priorities")
    print("  4. Check outputs/ folder for all visualizations")
    
    print("\n" + "="*70)
    print("🎉 PREDICTIVE MAINTENANCE SYSTEM READY FOR DEPLOYMENT!")
    print("="*70)

def main():
    """Main execution function."""
    print("\n" + "🏭 PREDICTIVE MAINTENANCE SYSTEM" + "\n")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create directories
    create_directories()
    
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data()
    
    # Step 2: EDA
    feature_cols = perform_eda(df)
    
    # Step 3: Feature engineering
    df_engineered, selected_features = engineer_features(df, feature_cols)
    
    # Step 4: Train models
    best_model_name, best_predictions, best_probabilities, y_test, test_indices = train_models(
        df_engineered, selected_features
    )
    
    # Step 5: Create dashboard
    results_df, high_risk_machines = create_dashboard(
        best_model_name, best_predictions, best_probabilities, 
        y_test, test_indices, df_engineered
    )
    
    # Print summary
    print_summary(results_df, high_risk_machines)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
