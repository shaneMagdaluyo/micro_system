"""
Risk Score Analysis System
Streamlit application for analyzing risk score distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Must be the first Streamlit command
st.set_page_config(
    page_title="Risk Score Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    data = {
        'ID': range(1, 101),
        'Name': [f'Person_{i}' for i in range(1, 101)],
        'Age': np.random.randint(18, 80, 100),
        'Risk_Score': np.random.normal(70, 15, 100),
        'Category': np.random.choice(['Low', 'Medium', 'High'], 100, p=[0.3, 0.4, 0.3])
    }
    return pd.DataFrame(data)


def get_statistics(df):
    """Calculate basic statistics for Risk_Score."""
    risk_scores = df['Risk_Score']
    stats = {
        'Count': len(risk_scores),
        'Mean': round(risk_scores.mean(), 2),
        'Median': round(risk_scores.median(), 2),
        'Std Dev': round(risk_scores.std(), 2),
        'Min': round(risk_scores.min(), 2),
        'Max': round(risk_scores.max(), 2),
        'Q25': round(risk_scores.quantile(0.25), 2),
        'Q75': round(risk_scores.quantile(0.75), 2),
        'Skewness': round(risk_scores.skew(), 2),
        'Kurtosis': round(risk_scores.kurtosis(), 2)
    }
    return stats


def create_histogram(df):
    """Create a histogram of the Risk_Score column."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram with KDE
    sns.histplot(data=df, x='Risk_Score', kde=True, bins=30, 
                color='skyblue', edgecolor='black', alpha=0.7, ax=ax)
    
    # Add mean and median lines
    risk_scores = df['Risk_Score']
    ax.axvline(risk_scores.mean(), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {risk_scores.mean():.2f}')
    ax.axvline(risk_scores.median(), color='green', linestyle='--', 
              linewidth=2, label=f'Median: {risk_scores.median():.2f}')
    
    # Add titles and labels
    ax.set_title('Distribution of Risk Score', fontsize=16, fontweight='bold')
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_boxplot(df):
    """Create a box plot of the Risk_Score column."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create box plot
    sns.boxplot(data=df, y='Risk_Score', color='lightblue', width=0.3, ax=ax)
    
    # Add individual points for better visualization
    sns.stripplot(data=df, y='Risk_Score', color='red', alpha=0.3, size=4, jitter=True, ax=ax)
    
    # Add titles and labels
    ax.set_title('Box Plot of Risk Score', fontsize=16, fontweight='bold')
    ax.set_ylabel('Risk Score', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def create_combined_visualization(df):
    """Create combined histogram and box plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    risk_scores = df['Risk_Score']
    
    # Histogram
    sns.histplot(data=df, x='Risk_Score', kde=True, bins=30,
                color='skyblue', edgecolor='black', alpha=0.7, ax=ax1)
    ax1.axvline(risk_scores.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {risk_scores.mean():.2f}')
    ax1.axvline(risk_scores.median(), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {risk_scores.median():.2f}')
    ax1.set_title('Distribution of Risk Score (Histogram)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Risk Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    sns.boxplot(data=df, y='Risk_Score', color='lightblue', width=0.3, ax=ax2)
    sns.stripplot(data=df, y='Risk_Score', color='red', alpha=0.3, size=3, jitter=True, ax=ax2)
    ax2.set_title('Distribution of Risk Score (Box Plot)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Risk Score', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Risk Score Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def home_page():
    """Home page content."""
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4; margin-bottom: 2rem;'>
            📊 Risk Score Analysis System
        </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 2rem; border-radius: 15px; height: 200px;'>
                <h3>📊 Histogram Analysis</h3>
                <p>View the distribution of risk scores with frequency analysis and KDE curve</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        color: white; padding: 2rem; border-radius: 15px; height: 200px;'>
                <h3>📦 Box Plot</h3>
                <p>Identify outliers, quartiles, and the spread of your risk score data</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        color: white; padding: 2rem; border-radius: 15px; height: 200px;'>
                <h3>📈 Statistics</h3>
                <p>Get detailed statistical measures including mean, median, and standard deviation</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation hint
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 **Navigate to 'Analyze Risk Score' from the sidebar to start your analysis!**")
    
    # Sample data preview
    st.markdown("---")
    st.subheader("📋 Sample Data Preview")
    if 'data' not in st.session_state:
        st.session_state.data = create_sample_data()
    st.dataframe(st.session_state.data.head(10), use_container_width=True)


def analyze_page():
    """Analysis page content."""
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4; margin-bottom: 2rem;'>
            📈 Risk Score Distribution Analysis
        </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data source selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Data Source")
        data_option = st.radio(
            "Choose data source:",
            ["Use Sample Data", "Upload Custom Dataset"],
            horizontal=True
        )
    
    # Handle data loading
    if data_option == "Upload Custom Dataset":
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV, Excel, or JSON)",
            type=['csv', 'xlsx', 'json'],
            help="File must contain a 'Risk_Score' column"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    st.session_state.data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    st.session_state.data = pd.read_json(uploaded_file)
                
                st.success(f"✅ Dataset loaded successfully! ({len(st.session_state.data)} rows)")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                st.session_state.data = create_sample_data()
    else:
        if 'data' not in st.session_state or st.button("🔄 Generate New Sample Data"):
            st.session_state.data = create_sample_data()
            st.success("✅ Sample data generated successfully!")
    
    # Check if Risk_Score column exists
    if 'data' in st.session_state:
        df = st.session_state.data
        
        if 'Risk_Score' not in df.columns:
            st.error("❌ The dataset must contain a 'Risk_Score' column!")
            st.write("Available columns:", df.columns.tolist())
            return
        
        st.markdown("---")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Statistics section
        st.subheader("📊 Statistical Summary")
        stats = get_statistics(df)
        
        # Display statistics in columns
        cols = st.columns(5)
        metrics = [
            ("Count", stats['Count'], cols[0]),
            ("Mean", stats['Mean'], cols[1]),
            ("Median", stats['Median'], cols[2]),
            ("Std Dev", stats['Std Dev'], cols[3]),
            ("Min/Max", f"{stats['Min']} / {stats['Max']}", cols[4])
        ]
        
        for label, value, col in metrics:
            with col:
                st.metric(label=label, value=value)
        
        # Additional statistics
        with st.expander("📈 View Detailed Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Q25 (First Quartile):** {stats['Q25']}")
                st.write(f"**Q75 (Third Quartile):** {stats['Q75']}")
            with col2:
                st.write(f"**Skewness:** {stats['Skewness']}")
                st.write(f"**Kurtosis:** {stats['Kurtosis']}")
            with col3:
                st.write(f"**Total Records:** {stats['Count']}")
                st.write(f"**Range:** {stats['Max'] - stats['Min']:.2f}")
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📊 Visualizations")
        
        # Tab selection for different views
        tab1, tab2, tab3 = st.tabs(["📊 Histogram", "📦 Box Plot", "🔍 Combined View"])
        
        with tab1:
            st.markdown("### Distribution of Risk Score (Histogram)")
            st.markdown("*With KDE curve, mean, and median indicators*")
            fig_hist = create_histogram(df)
            st.pyplot(fig_hist)
            
        with tab2:
            st.markdown("### Box Plot of Risk Score")
            st.markdown("*Shows quartiles, outliers, and data distribution*")
            fig_box = create_boxplot(df)
            st.pyplot(fig_box)
            
        with tab3:
            st.markdown("### Combined Analysis View")
            st.markdown("*Histogram and Box Plot side by side*")
            fig_combined = create_combined_visualization(df)
            st.pyplot(fig_combined)
        
        # Download option
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name="risk_score_data.csv",
                mime="text/csv"
            )


def about_page():
    """About page content."""
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4; margin-bottom: 2rem;'>
            ℹ️ About This System
        </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div style='background: #f8f9fa; padding: 2rem; border-radius: 15px;'>
                <h2 style='color: #667eea;'>Risk Score Analysis System</h2>
                <p style='font-size: 1.1rem; line-height: 1.8;'>
                    This system provides comprehensive analysis and visualization of risk score distributions.
                    It helps you understand the patterns, outliers, and statistical properties of your risk assessment data.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px;'>
                <h3 style='color: #333;'>Features</h3>
                <ul style='line-height: 2;'>
                    <li>📊 Histogram visualization with KDE curve</li>
                    <li>📦 Box plot analysis for outlier detection</li>
                    <li>📈 Comprehensive statistical summary</li>
                    <li>📁 Support for CSV, Excel, and JSON formats</li>
                    <li>🔄 Interactive data analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px;'>
                <h3 style='color: #333;'>How to Use</h3>
                <ol style='line-height: 2;'>
                    <li>Navigate to <b>Analyze Risk Score</b></li>
                    <li>Upload your dataset or use sample data</li>
                    <li>View statistical summary and charts</li>
                    <li>Interpret results for risk assessment</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "📈 Analyze Risk Score", "ℹ️ About"],
        key="navigation"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
            <p>Risk Score Analysis System v1.0</p>
            <p>Built with Streamlit</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "📈 Analyze Risk Score":
        analyze_page()
    elif page == "ℹ️ About":
        about_page()


if __name__ == "__main__":
    main()
