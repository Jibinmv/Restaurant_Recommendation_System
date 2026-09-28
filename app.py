

import streamlit as st
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess_dataset
from src.recommender import RestaurantRecommender



st.set_page_config(
    page_title="Restaurant Recommendation System",
    page_icon="🍽️",
    layout="wide"
)




@st.cache_resource(show_spinner="Loading and preprocessing data...")
def load_recommender():
    """
    Load the dataset, preprocess it, and build the recommender.
    This is cached so it only runs once per session.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'restaurants.csv')

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at: {data_path}")
        st.stop()

    df = preprocess_dataset(data_path)
    recommender = RestaurantRecommender(df)
    return df, recommender


# Load data and build recommender
df, recommender = load_recommender()




st.sidebar.title("🍽️ Your Preferences")
st.sidebar.markdown("---")

# --- City dropdown ---
# Get unique cities from the dataset, sorted alphabetically
cities = sorted(df['City'].dropna().unique().tolist())
city_options = ['Any City'] + cities
selected_city = st.sidebar.selectbox(
    "🏙️ City",
    options=city_options,
    index=0,
    help="Select your preferred city or 'Any City' for all locations."
)

# --- Cuisine input ---
# Get unique individual cuisines from the dataset
all_cuisines_raw = df['Cuisines'].dropna().str.split(',').explode().str.strip()
unique_cuisines = sorted(all_cuisines_raw.unique().tolist())
cuisine_options = ['Any Cuisine'] + unique_cuisines
selected_cuisine = st.sidebar.selectbox(
    "🍕 Cuisine",
    options=cuisine_options,
    index=0,
    help="Select your preferred cuisine type."
)

# --- Price range selector ---
selected_price = st.sidebar.slider(
    "💰 Price Range",
    min_value=1,
    max_value=4,
    value=2,
    step=1,
    help="1 = Budget, 2 = Moderate, 3 = Expensive, 4 = Premium"
)

# --- Minimum rating selector ---
selected_rating = st.sidebar.slider(
    "⭐ Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=3.5,
    step=0.5,
    help="Only show restaurants with at least this rating."
)

# --- Online delivery selector ---
selected_delivery = st.sidebar.radio(
    "🚗 Online Delivery",
    options=['Yes', 'No'],
    index=0,
    help="Do you want online delivery?"
)

# --- Table booking selector ---
selected_booking = st.sidebar.radio(
    "📋 Table Booking",
    options=['No', 'Yes'],
    index=0,
    help="Do you want table booking?"
)

# --- Number of recommendations ---
top_n = st.sidebar.slider(
    "📊 Number of Recommendations",
    min_value=5,
    max_value=25,
    value=10,
    step=5,
    help="How many restaurants to recommend."
)

st.sidebar.markdown("---")




# Title
st.title("🍽️ Restaurant Recommendation System")


# Recommendation button
recommend_clicked = st.sidebar.button(
    "🔍 Recommend Restaurants",
    use_container_width=True,
    type="primary"
)

if recommend_clicked:
    # Prepare parameters
    city_param = '' if selected_city == 'Any City' else selected_city
    cuisine_param = '' if selected_cuisine == 'Any Cuisine' else selected_cuisine

    # Get recommendations
    with st.spinner("Finding the best restaurants for you..."):
        results = recommender.recommend(
            city=city_param,
            cuisine=cuisine_param,
            price_range=selected_price,
            minimum_rating=selected_rating,
            online_delivery=selected_delivery,
            table_booking=selected_booking,
            top_n=top_n
        )

    # Display results
    if len(results) == 0:
        st.warning(
            "⚠️ No restaurants found matching your preferences. "
            "Try relaxing your filters (lower the minimum rating, "
            "change city/cuisine, etc.)."
        )
    else:
        st.success(f"✅ Found {len(results)} recommendation(s)!")

        # Show user preferences summary
        with st.expander("📋 Your Preferences", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("City", selected_city)
                st.metric("Cuisine", selected_cuisine)
            with col2:
                st.metric("Price Range", f"{selected_price}/4")
                st.metric("Min Rating", f"{selected_rating}/5.0")
            with col3:
                st.metric("Online Delivery", selected_delivery)
                st.metric("Table Booking", selected_booking)

        # Display recommendations table
        st.subheader("🏆 Recommended Restaurants")

        # Format the display dataframe
        display_df = results.copy()
        display_df.columns = [
            'Rank', 'Restaurant', 'City', 'Cuisines', 'Price Range',
            'Rating', 'Votes', 'Online Delivery', 'Table Booking',
            'Similarity Score'
        ]

        # Style the dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Rank': st.column_config.NumberColumn('🏅 Rank', width='small'),
                'Restaurant': st.column_config.TextColumn('🍽️ Restaurant', width='medium'),
                'City': st.column_config.TextColumn('🏙️ City', width='small'),
                'Cuisines': st.column_config.TextColumn('🍕 Cuisines', width='medium'),
                'Price Range': st.column_config.NumberColumn('💰 Price', width='small'),
                'Rating': st.column_config.NumberColumn('⭐ Rating', format='%.1f', width='small'),
                'Votes': st.column_config.NumberColumn('👍 Votes', width='small'),
                'Online Delivery': st.column_config.TextColumn('🚗 Delivery', width='small'),
                'Table Booking': st.column_config.TextColumn('📋 Booking', width='small'),
                'Similarity Score': st.column_config.ProgressColumn(
                    '📊 Similarity',
                    min_value=0,
                    max_value=1,
                    format='%.4f',
                    width='small'
                ),
            }
        )

        # Quick stats
        st.markdown("---")
        st.subheader("📊 Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Rating", f"{results['Aggregate rating'].mean():.2f}")
        with col2:
            st.metric("Avg Similarity", f"{results['Similarity Score'].mean():.4f}")
        with col3:
            price_match = (results['Price range'] == selected_price).mean() * 100
            st.metric("Price Match", f"{price_match:.0f}%")
        with col4:
            if cuisine_param:
                cuisine_match = results['Cuisines'].str.lower().str.contains(
                    cuisine_param.lower(), na=False
                ).mean() * 100
            else:
                cuisine_match = 100.0
            st.metric("Cuisine Match", f"{cuisine_match:.0f}%")

else:
    # Default view when no recommendations yet
    st.info(
        "👈 Set your preferences in the sidebar and click "
        "**Recommend Restaurants** to get started!"
    )

    # Show dataset overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Restaurants", f"{len(df):,}")
    with col2:
        st.metric("Cities", df['City'].nunique())
    with col3:
        unique_cuisines_count = df['Cuisines'].str.split(',').explode().str.strip().nunique()
        st.metric("Cuisine Types", unique_cuisines_count)



st.markdown("---")
st.markdown(
    "💡 **How it works:** Recommendations are generated using "
    "**content-based filtering** based on similarity between your "
    "preferences and restaurant features. The system uses TF-IDF "
    "vectorization for text features (city, cuisines) and cosine "
    "similarity to find the best matching restaurants."
)
st.caption("Cognifyz Technologies | ML Internship Task 2 | Restaurant Recommendation System")
