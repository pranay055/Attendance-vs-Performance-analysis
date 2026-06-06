import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set page layout configuration for the browser tab
st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

def load_data():
    primary_path = "data/student_data_v1.csv"
    fallback_path = "data/student_data_v1.csv.txt"
    target_path = primary_path if os.path.exists(primary_path) else (fallback_path if os.path.exists(fallback_path) else None)
    
    if target_path is None:
        st.error("❌ Data file 'student_data_v1.csv' could not be found inside the 'data' folder.")
        return None
    return pd.read_csv(target_path)

def clean_and_prepare_data(df):
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    df["Attendance_Pct"] = pd.to_numeric(df["Attendance_Pct"], errors='coerce').fillna(df["Attendance_Pct"].mean())
    df["Exam_Marks"] = pd.to_numeric(df["Exam_Marks"], errors='coerce').fillna(df["Exam_Marks"].median())
    
    df["Attendance_Category"] = df["Attendance_Pct"].apply(
        lambda x: "Excellent (>90%)" if x >= 90 else ("Good (75-89%)" if x >= 75 else ("Shortage (65-74%)" if x >= 65 else "Critical (<65%)"))
    )
    df["Is_Pass"] = df["Exam_Marks"] >= 40

    def get_quadrant(row):
        if row["Attendance_Pct"] >= 75 and row["Exam_Marks"] >= 40: return "Safe Zone (Good Attendance + Passed)"
        elif row["Attendance_Pct"] < 75 and row["Exam_Marks"] < 40: return "High Risk (Poor Attendance + Failed)"
        elif row["Attendance_Pct"] < 75 and row["Exam_Marks"] >= 40: return "Detained but Capable (Low Attendance + Passed)"
        else: return "Struggling (High Attendance + Failed)"

    df["Risk_Quadrant"] = df.apply(get_quadrant, axis=1)
    return df

# --- WEB APPLICATION FRONTEND LAYER ---
st.title("📈 Student Attendance vs. Performance Analytics Portal")
st.markdown("Interactive Business Intelligence dashboard mapping student risk quadrants and class metrics.")
st.markdown("---")

df_raw = load_data()
if df_raw is not None:
    df = clean_and_prepare_data(df_raw)
    
    # Calculate Statistical Metrics
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["Attendance_Pct"], df["Exam_Marks"])
    weak_df = df[df["Risk_Quadrant"] == "High Risk (Poor Attendance + Failed)"][["Student_ID", "Student_Name", "Attendance_Pct", "Exam_Marks"]]
    
    # KPI Highlights Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Dataset Sample Size", value=f"{len(df)} Students")
    with col2:
        st.metric(label="Pearson Correlation (R)", value=f"{r_value:.2f}", delta="Strong Positive Correlation")
    with col3:
        st.metric(label="Critical High Risk Students", value=f"{len(weak_df)} Alerted", delta="- Action Required", delta_color="inverse")
    
    st.markdown("---")
    
    # Visual Layout Split (Left: Charts, Right: Tables)
    left_ui, right_ui = st.columns([1.2, 1])
    
    with left_ui:
        st.subheader("📊 Visual Distribution Engine")
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot 1: Linear Regression
        sns.regplot(ax=axes[0], data=df, x="Attendance_Pct", y="Exam_Marks", scatter_kws={"color": "#2c3e50"}, line_kws={"color": "#e74c3c", "linewidth": 2.5})
        axes[0].set_title(f"Attendance vs Marks Trend Line (R² = {r_value**2:.2f})", weight="bold")
        
        # Plot 2: Scatter Quadrants
        sns.scatterplot(ax=axes[1], data=df, x="Attendance_Pct", y="Exam_Marks", hue="Risk_Quadrant", palette="Set1", s=120)
        axes[1].axhline(y=40, color="gray", linestyle="--", alpha=0.7, label="Pass Cutoff (40 Marks)")
        axes[1].axvline(x=75, color="purple", linestyle="--", alpha=0.7, label="Attendance Cutoff (75%)")
        axes[1].set_title("Student Performance Risk Quadrants Mapping", weight="bold")
        axes[1].legend(loc="lower right", fontsize='small')
        
        plt.tight_layout()
        st.pyplot(fig)

    with right_ui:
        st.subheader("🚨 Risk Management Registry")
        st.dataframe(weak_df.reset_index(drop=True), use_container_width=True)
        
        # Download Link for Placement Tracking CSV
        csv_data = weak_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Export High Risk Registry (CSV)", data=csv_data, file_name="high_risk_students.csv", mime="text/csv")
        
        st.markdown("---")
        st.subheader("📊 Class Performance Summary Matrix")
        summary_matrix = df.groupby("Attendance_Category").agg(
            Total_Students=("Student_ID", "count"),
            Average_Marks=("Exam_Marks", "mean")
        ).reset_index()
        st.dataframe(summary_matrix, use_container_width=True)
        
        st.info(f"💡 **Drop Impact Insight:** For every 10% structural drop in class attendance, student score metrics are projected to fall by ~{slope*10:.1f} marks based on linear distribution models.")