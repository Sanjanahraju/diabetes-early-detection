"""
Data Analysis & Visualization for Type 2 Diabetes Dataset
==========================================================
Performs comprehensive Exploratory Data Analysis (EDA) and generates
publication-quality visualizations for the VTU project report.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Set professional plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'visualizations')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes_enhanced.csv')


def load_data():
    """Load the enhanced diabetes dataset."""
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def statistical_summary(df):
    """Print comprehensive statistical summary."""
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    
    print("\n--- Numerical Features ---")
    print(df.describe().round(2).to_string())
    
    print("\n--- Categorical Features ---")
    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().to_string())
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found [OK]")
    else:
        print(missing[missing > 0])
    
    print("\n--- Class Distribution ---")
    print(df['diabetes'].value_counts())
    print(f"\nDiabetes prevalence: {df['diabetes'].mean()*100:.2f}%")


def plot_class_distribution(df):
    """Plot diabetes class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    counts = df['diabetes'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = axes[0].bar(['Non-Diabetic (0)', 'Diabetic (1)'], counts.values, color=colors, 
                       edgecolor='white', linewidth=2, width=0.6)
    axes[0].set_title('Class Distribution', fontweight='bold', fontsize=16)
    axes[0].set_ylabel('Count')
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
                    f'{count:,}\n({count/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=['Non-Diabetic', 'Diabetic'], colors=colors,
               autopct='%1.1f%%', startangle=90, explode=(0, 0.08),
               textprops={'fontsize': 13, 'fontweight': 'bold'},
               shadow=True)
    axes[1].set_title('Diabetes Prevalence', fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/class_distribution.png')
    plt.close()
    print("[OK] class_distribution.png saved")


def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical features."""
    # Encode categoricals for correlation
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    
    fig, ax = plt.subplots(figsize=(16, 13))
    corr_matrix = df_encoded.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', linewidths=0.5,
                square=True, ax=ax,
                annot_kws={'size': 8},
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'})
    
    ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/correlation_heatmap.png')
    plt.close()
    print("[OK] correlation_heatmap.png saved")


def plot_feature_distributions(df):
    """Plot distributions of numerical features split by diabetes status."""
    num_cols = ['age', 'pregnancies', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
                'stress_level', 'sleep_hours', 'waist_circumference',
                'cholesterol_total', 'bp_systolic', 'bp_diastolic']
    
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    axes = axes.flatten()
    
    colors = {'Non-Diabetic': '#3498db', 'Diabetic': '#e74c3c'}
    
    for idx, col in enumerate(num_cols):
        ax = axes[idx]
        for label, color in colors.items():
            subset = df[df['diabetes'] == (1 if label == 'Diabetic' else 0)]
            ax.hist(subset[col], bins=40, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(num_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Feature Distributions by Diabetes Status', fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/feature_distributions.png')
    plt.close()
    print("[OK] feature_distributions.png saved")


def plot_categorical_distributions(df):
    """Plot categorical feature distributions."""
    cat_features = ['gender', 'smoking_history', 'family_history', 
                    'physical_activity', 'diet_quality', 'alcohol_consumption']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(cat_features):
        ax = axes[idx]
        ct = pd.crosstab(df[col], df['diabetes'], normalize='index') * 100
        ct.columns = ['Non-Diabetic %', 'Diabetic %']
        ct.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'], edgecolor='white', width=0.7)
        ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Percentage')
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=30)
    
    fig.suptitle('Categorical Feature Analysis by Diabetes Status', fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/categorical_distributions.png')
    plt.close()
    print("[OK] categorical_distributions.png saved")


def plot_boxplots(df):
    """Box plots for key numerical features by diabetes status."""
    features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                'bp_systolic', 'cholesterol_total', 'waist_circumference', 'sleep_hours']
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(features):
        ax = axes[idx]
        df.boxplot(column=col, by='diabetes', ax=ax,
                   patch_artist=True,
                   boxprops=dict(facecolor='#3498db', alpha=0.7),
                   medianprops=dict(color='#e74c3c', linewidth=2))
        ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Diabetes Status')
        ax.get_figure().suptitle('')
    
    fig.suptitle('Feature Box Plots: Non-Diabetic vs Diabetic', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/boxplots.png')
    plt.close()
    print("[OK] boxplots.png saved")


def plot_pairplot(df):
    """Pairplot for key clinical features."""
    key_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
    subset = df[key_features].sample(3000, random_state=42)  # Sample for performance
    
    g = sns.pairplot(subset, hue='diabetes', palette={0: '#3498db', 1: '#e74c3c'},
                     diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15},
                     diag_kws={'fill': True, 'alpha': 0.5})
    g.figure.suptitle('Pairwise Feature Relationships', fontsize=18, fontweight='bold', y=1.02)
    g.savefig(f'{OUTPUT_DIR}/pairplot.png')
    plt.close()
    print("[OK] pairplot.png saved")


def plot_diabetes_by_age_bmi(df):
    """Scatter: Age vs BMI colored by diabetes status."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sample = df.sample(5000, random_state=42)
    scatter = ax.scatter(sample['age'], sample['bmi'], 
                        c=sample['diabetes'], cmap='RdYlGn_r',
                        alpha=0.5, s=20, edgecolors='none')
    ax.set_xlabel('Age', fontsize=14)
    ax.set_ylabel('BMI', fontsize=14)
    ax.set_title('Diabetes Risk: Age vs BMI', fontweight='bold', fontsize=16)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Diabetes Status', fontsize=12)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Non-Diabetic', 'Diabetic'])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/age_vs_bmi_scatter.png')
    plt.close()
    print("[OK] age_vs_bmi_scatter.png saved")


def run_full_analysis():
    """Run the complete EDA pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = load_data()
    
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    statistical_summary(df)
    
    print("\n--- Generating Visualizations ---")
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_categorical_distributions(df)
    plot_boxplots(df)
    plot_pairplot(df)
    plot_diabetes_by_age_bmi(df)
    
    print("\n" + "=" * 60)
    print(f"ALL VISUALIZATIONS SAVED TO: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    run_full_analysis()
