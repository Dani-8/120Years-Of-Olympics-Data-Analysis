import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🏆 Olympic Medal Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FFD700, #C0C0C0, #CD7F32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-high {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .prediction-medium {
        background: linear-gradient(135deg, #C0C0C0, #87CEEB);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .prediction-low {
        background: linear-gradient(135deg, #CD7F32, #D2691E);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# --- TITLE AND HEADER ---
st.markdown('<h1 class="main-header">🏆 Olympic Medal Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
    Advanced AI-powered prediction system using historical Olympic data (1896-2016)
    <br>
    Predict medal chances based on athlete characteristics and performance factors
</div>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Load and preprocess Olympic dataset with sample data if file not found."""
    try:
        df = pd.read_csv('athlete_events.csv')
        return df
    except FileNotFoundError:
        st.warning("📁 'athlete_events.csv' not found. Using simulated data for demonstration.")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 10000
        
        countries = ['USA', 'CHN', 'RUS', 'GER', 'JPN', 'GBR', 'FRA', 'ITA', 'AUS', 'CAN']
        sports = ['Swimming', 'Athletics', 'Gymnastics', 'Basketball', 'Wrestling', 'Cycling', 'Boxing', 'Weightlifting']
        genders = ['M', 'F']
        medals = ['Gold', 'Silver', 'Bronze', None, None, None, None]  # Most athletes don't win medals
        
        sample_data = {
            'ID': range(1, n_samples + 1),
            'Name': [f'Athlete_{i}' for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(genders, n_samples),
            'Age': np.random.normal(25, 4, n_samples).astype(int),
            'Height': np.random.normal(175, 10, n_samples),
            'Weight': np.random.normal(70, 15, n_samples),
            'NOC': np.random.choice(countries, n_samples),
            'Sport': np.random.choice(sports, n_samples),
            'Medal': np.random.choice(medals, n_samples),
            'Year': np.random.choice(range(1992, 2017, 4), n_samples)
        }
        
        df = pd.DataFrame(sample_data)
        df['Height'] = df['Height'].clip(150, 220)
        df['Weight'] = df['Weight'].clip(45, 150)
        df['Age'] = df['Age'].clip(15, 45)
        
        return df

# Load data
df = load_data()

# Sidebar for navigation
st.sidebar.title("🎯 Navigation")
page_selection = st.sidebar.radio(
    "Choose Analysis Section:",
    ["📊 Dashboard Overview", "📈 Advanced Analytics", "🤖 Medal Prediction", "📋 Athlete Database"]
)

# --- DASHBOARD OVERVIEW ---
if page_selection == "📊 Dashboard Overview":
    st.header("📊 Olympic Games Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_athletes = len(df)
        st.markdown(f"""
        <div class="metric-container">
            <h3>{total_athletes:,}</h3>
            <p>Total Athletes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_medals = df['Medal'].notna().sum()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{total_medals:,}</h3>
            <p>Total Medals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_countries = df['NOC'].nunique()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{total_countries}</h3>
            <p>Countries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_sports = df['Sport'].nunique()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{total_sports}</h3>
            <p>Sports</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏅 Medal Distribution by Country")
        medal_by_country = df[df['Medal'].notna()].groupby('NOC').size().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=medal_by_country.values,
            y=medal_by_country.index,
            orientation='h',
            color=medal_by_country.values,
            color_continuous_scale='Viridis',
            title="Top 10 Countries by Medal Count"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏃‍♂️ Athletes by Sport")
        sport_counts = df['Sport'].value_counts().head(10)
        
        fig = px.pie(
            values=sport_counts.values,
            names=sport_counts.index,
            title="Distribution of Athletes Across Sports"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Age and Physical Characteristics Analysis
    st.subheader("📏 Physical Characteristics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x='Age',
            nbins=30,
            title="Age Distribution of Olympic Athletes",
            color_discrete_sequence=['#FF6B6B']
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df.dropna(subset=['Height', 'Weight']),
            x='Height',
            y='Weight',
            color='Sex',
            title="Height vs Weight Distribution",
            opacity=0.6
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# --- ADVANCED ANALYTICS ---
elif page_selection == "📈 Advanced Analytics":
    st.header("📈 Advanced Olympic Analytics")
    
    # Medal success rate analysis
    st.subheader("🎯 Success Rates Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate by sport
        sport_analysis = df.groupby('Sport').agg({
            'Medal': ['count', lambda x: x.notna().sum()]
        }).round(3)
        sport_analysis.columns = ['Total_Athletes', 'Medal_Winners']
        sport_analysis['Success_Rate'] = (sport_analysis['Medal_Winners'] / sport_analysis['Total_Athletes'] * 100).round(2)
        sport_analysis = sport_analysis.sort_values('Success_Rate', ascending=False).head(10)
        
        fig = px.bar(
            x=sport_analysis.index,
            y=sport_analysis['Success_Rate'],
            title="Medal Success Rate by Sport (%)",
            color=sport_analysis['Success_Rate'],
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age performance correlation
        age_medals = df.groupby('Age').agg({
            'Medal': lambda x: x.notna().sum(),
            'ID': 'count'
        }).reset_index()
        age_medals['Success_Rate'] = (age_medals['Medal'] / age_medals['ID'] * 100).round(2)
        age_medals = age_medals[age_medals['ID'] >= 50]  # Filter for statistical significance
        
        fig = px.line(
            age_medals,
            x='Age',
            y='Success_Rate',
            title="Success Rate by Age",
            markers=True
        )
        fig.add_hline(y=age_medals['Success_Rate'].mean(), line_dash="dash", 
                     annotation_text=f"Average: {age_medals['Success_Rate'].mean():.1f}%")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Gender analysis
    st.subheader("⚖️ Gender Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender_medals = df[df['Medal'].notna()].groupby(['Sex', 'Medal']).size().unstack(fill_value=0)
        
        fig = px.bar(
            gender_medals,
            title="Medal Distribution by Gender",
            barmode='group',
            color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Physical characteristics by medal winners
        medal_winners = df[df['Medal'].notna()]
        fig = px.box(
            medal_winners,
            x='Medal',
            y='Height',
            title="Height Distribution by Medal Type",
            color='Medal',
            color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlation Analysis")
    numeric_cols = ['Age', 'Height', 'Weight']
    correlation_data = df[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_data,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Physical Attributes",
        color_continuous_scale='RdBu'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- MEDAL PREDICTION ---
elif page_selection == "🤖 Medal Prediction":
    st.header("🤖 AI Medal Prediction System")
    st.write("Enter athlete details below to predict their medal chances using our advanced algorithm.")
    
    # Enhanced prediction function
    def enhanced_predict_medal_chance(age, height, weight, gender, sport, country):
        """Enhanced medal prediction with multiple factors."""
        
        medal_score = 0.0
        confidence_factors = []
        
        # Sport success factor (40% weight)
        sport_data = df[df['Sport'] == sport]
        if len(sport_data) > 0:
            sport_success_rate = (sport_data['Medal'].notna().sum() / len(sport_data)) * 100
            medal_score += sport_success_rate * 0.4
            confidence_factors.append(f"Sport success rate: {sport_success_rate:.1f}%")
        
        # Country success factor (25% weight)
        country_data = df[df['NOC'] == country]
        if len(country_data) > 0:
            country_success_rate = (country_data['Medal'].notna().sum() / len(country_data)) * 100
            medal_score += country_success_rate * 0.25
            confidence_factors.append(f"Country success rate: {country_success_rate:.1f}%")
        
        # Age factor (20% weight)
        age_optimal = df.groupby('Age')['Medal'].apply(lambda x: x.notna().sum() / len(x) * 100)
        if age in age_optimal.index:
            age_factor = age_optimal[age]
            medal_score += age_factor * 0.2
            confidence_factors.append(f"Age factor: {age_factor:.1f}%")
        
        # Physical attributes factor (15% weight)
        sport_athletes = df[(df['Sport'] == sport) & (df['Sex'] == gender) & df['Medal'].notna()]
        if len(sport_athletes) > 0:
            height_mean = sport_athletes['Height'].mean()
            weight_mean = sport_athletes['Weight'].mean()
            height_score = max(0, 100 - abs(height - height_mean) * 2)
            weight_score = max(0, 100 - abs(weight - weight_mean) * 2)
            physical_score = (height_score + weight_score) / 2
            medal_score += physical_score * 0.15
            confidence_factors.append(f"Physical match: {physical_score:.1f}%")
        
        # Determine medal prediction
        if medal_score >= 15:
            prediction = "🥇 High Medal Probability"
            color_class = "prediction-high"
            medal_type = "Gold Medal Potential"
        elif medal_score >= 8:
            prediction = "🥈 Medium Medal Probability" 
            color_class = "prediction-medium"
            medal_type = "Silver/Bronze Potential"
        else:
            prediction = "🥉 Low Medal Probability"
            color_class = "prediction-low"
            medal_type = "Training Recommended"
        
        return prediction, medal_score, confidence_factors, color_class, medal_type
    
    # Input form with better layout
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Basic Info")
            age = st.slider("Age", 15, 45, 25, help="Athlete's age in years")
            gender = st.selectbox("Gender", ["M", "F"], help="M = Male, F = Female")
        
        with col2:
            st.subheader("📏 Physical Stats")
            height = st.slider("Height (cm)", 150, 220, 175, help="Height in centimeters")
            weight = st.slider("Weight (kg)", 45, 150, 70, help="Weight in kilograms")
        
        with col3:
            st.subheader("🏃‍♂️ Competition Info")
            available_sports = sorted(df['Sport'].unique())
            sport = st.selectbox("Sport", available_sports, help="Select the sport")
            
            available_countries = sorted(df['NOC'].unique())
            country = st.selectbox("Country Code", available_countries, help="3-letter country code")
        
        predict_button = st.form_submit_button("🚀 Predict Medal Chance", use_container_width=True)
    
    if predict_button:
        with st.spinner("🤖 Analyzing athlete profile..."):
            prediction, score, factors, color_class, medal_type = enhanced_predict_medal_chance(
                age, height, weight, gender, sport, country
            )
        
        # Results display
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="{color_class}">
                <h2>{prediction}</h2>
                <h4>{medal_type}</h4>
                <p>Confidence Score: {score:.1f}/100</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence factors
            st.subheader("📋 Analysis Breakdown")
            for factor in factors:
                st.write(f"• {factor}")
        
        with col2:
            # Gauge chart for confidence
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with similar athletes
        st.subheader("👥 Similar Athletes Comparison")
        similar_athletes = df[
            (df['Sport'] == sport) & 
            (df['Sex'] == gender) & 
            (df['Age'].between(age-2, age+2))
        ].head(100)
        
        if len(similar_athletes) > 0:
            similar_medal_rate = (similar_athletes['Medal'].notna().sum() / len(similar_athletes)) * 100
            st.info(f"📊 Athletes with similar profile have a {similar_medal_rate:.1f}% medal success rate")
        
        # Recommendations
        st.subheader("💡 Performance Recommendations")
        recommendations = []
        
        if score < 30:
            recommendations.append("🏋️‍♂️ Focus on intensive training and skill development")
            recommendations.append("📈 Consider working with specialized coaches")
        elif score < 60:
            recommendations.append("🎯 Fine-tune technique and mental preparation")
            recommendations.append("🥇 You're on track for potential medal success!")
        else:
            recommendations.append("🌟 Excellent medal potential - maintain current training")
            recommendations.append("🏆 Consider participating in major competitions")
        
        for rec in recommendations:
            st.write(rec)

# --- ATHLETE DATABASE ---
elif page_selection == "📋 Athlete Database":
    st.header("📋 Olympic Athletes Database")
    st.write("Explore and search through the Olympic athletes database.")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sport_filter = st.selectbox("Filter by Sport", ["All"] + sorted(df['Sport'].unique()))
    
    with col2:
        country_filter = st.selectbox("Filter by Country", ["All"] + sorted(df['NOC'].unique()))
    
    with col3:
        gender_filter = st.selectbox("Filter by Gender", ["All", "M", "F"])
    
    with col4:
        medal_filter = st.selectbox("Medal Winners Only", ["All Athletes", "Medal Winners Only"])
    
    # Apply filters
    filtered_df = df.copy()
    
    if sport_filter != "All":
        filtered_df = filtered_df[filtered_df['Sport'] == sport_filter]
    
    if country_filter != "All":
        filtered_df = filtered_df[filtered_df['NOC'] == country_filter]
    
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df['Sex'] == gender_filter]
    
    if medal_filter == "Medal Winners Only":
        filtered_df = filtered_df[filtered_df['Medal'].notna()]
    
    st.write(f"📊 Showing {len(filtered_df):,} athletes")
    
    # Display data
    display_cols = ['Name', 'Sex', 'Age', 'Height', 'Weight', 'NOC', 'Sport', 'Medal', 'Year']
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    
    st.dataframe(
        filtered_df[available_cols].head(1000), 
        use_container_width=True,
        height=400
    )
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Statistics:**")
        numeric_cols = ['Age', 'Height', 'Weight']
        available_numeric = [col for col in numeric_cols if col in filtered_df.columns]
        if available_numeric:
            st.write(filtered_df[available_numeric].describe())
    
    with col2:
        st.write("**Medal Distribution:**")
        if 'Medal' in filtered_df.columns:
            medal_counts = filtered_df['Medal'].value_counts()
            st.write(medal_counts)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    🏆 Olympic Medal Prediction System | Built with Streamlit & Python<br>
    Data covers Olympic Games from 1896-2016 | For educational purposes
</div>
""", unsafe_allow_html=True)