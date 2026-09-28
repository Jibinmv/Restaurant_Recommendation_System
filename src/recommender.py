

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')




def create_text_features(df: pd.DataFrame) -> pd.Series:
    """
    Combine text-based features (City, Cuisines) into a single string
    for TF-IDF vectorization.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed restaurant dataframe.

    Returns
    -------
    pd.Series
        Combined text feature for each restaurant.
    """
    # Combine City and Cuisines into a single text field
    # This allows TF-IDF to capture city-cuisine combinations
    text_features = (
        df['City'].astype(str).str.lower() + ' ' +
        df['Cuisines'].astype(str).str.lower().str.replace(',', ' ')
    )
    return text_features


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary/categorical features (Has Online delivery, Has Table booking)
    as numeric values (1 = Yes, 0 = No).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed restaurant dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with two new numeric columns:
        'online_delivery_encoded' and 'table_booking_encoded'.
    """
    df = df.copy()

    # Encode "Yes" -> 1, "No" -> 0
    df['online_delivery_encoded'] = (
        df['Has Online delivery'].str.strip().str.lower().map({'yes': 1, 'no': 0})
    ).fillna(0).astype(int)

    df['table_booking_encoded'] = (
        df['Has Table booking'].str.strip().str.lower().map({'yes': 1, 'no': 0})
    ).fillna(0).astype(int)

    return df


def prepare_numerical_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract and scale numerical features:
    Price range, Aggregate rating, Votes, online_delivery_encoded,
    table_booking_encoded.

    We use Min-Max scaling so that all features are on a 0-1 scale,
    preventing any single feature from dominating the similarity score.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with encoded binary features.

    Returns
    -------
    np.ndarray
        Scaled numerical feature matrix (n_restaurants x n_features).
    """
    num_cols = [
        'Price range',
        'Aggregate rating',
        'Votes',
        'online_delivery_encoded',
        'table_booking_encoded'
    ]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[num_cols].values)
    return scaled


# =========================================================================
# RECOMMENDATION ENGINE
# =========================================================================

class RestaurantRecommender:
    """
    Content-based restaurant recommendation system.

    How it works:
    1. Text features (City, Cuisines) are vectorized using TF-IDF.
    2. Numerical features (Price range, Rating, Votes) are scaled.
    3. Binary features (Online delivery, Table booking) are encoded.
    4. All features are combined into a single feature matrix.
    5. A user-preference vector is constructed from the input.
    6. Cosine similarity is computed between the user vector and every
       restaurant vector.
    7. Restaurants are ranked by similarity and filtered by constraints.

    Attributes
    ----------
    df : pd.DataFrame
        The preprocessed restaurant data.
    tfidf : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    tfidf_matrix : sparse matrix
        TF-IDF feature matrix for all restaurants.
    num_matrix : np.ndarray
        Scaled numerical feature matrix.
    combined_matrix : np.ndarray
        Full combined feature matrix (TF-IDF + numerical).
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize and fit the recommender on the restaurant dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed restaurant dataframe.
        """
        self.df = df.copy()

        # --- Step 1: Encode binary features ---
        self.df = encode_binary_features(self.df)

        # --- Step 2: Create and fit TF-IDF on text features ---
        text_features = create_text_features(self.df)
        self.tfidf = TfidfVectorizer(
            max_features=5000,   # Limit vocabulary size
            stop_words='english' # Remove common English stop words
        )
        self.tfidf_matrix = self.tfidf.fit_transform(text_features)
        print(f"[INFO] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        # --- Step 3: Prepare numerical features ---
        self.num_matrix = prepare_numerical_features(self.df)
        print(f"[INFO] Numerical feature matrix shape: {self.num_matrix.shape}")

        # --- Step 4: Combine TF-IDF and numerical into one matrix ---
        # Convert sparse TF-IDF to dense and concatenate with numerical
        self.combined_matrix = np.hstack([
            self.tfidf_matrix.toarray(),
            self.num_matrix
        ])
        print(f"[INFO] Combined feature matrix shape: {self.combined_matrix.shape}")
        print("[INFO] Recommender initialized successfully!")

    def _build_user_vector(self, city: str, cuisine: str,
                           price_range: int, aggregate_rating: float,
                           votes: int, online_delivery: str,
                           table_booking: str) -> np.ndarray:
        """
        Construct a feature vector representing the user's preferences.

        We create a "pseudo-restaurant" row that matches the user's stated
        preferences, then transform it through the same TF-IDF vectorizer
        and scaler pipeline.

        Parameters
        ----------
        city : str
            Preferred city.
        cuisine : str
            Preferred cuisine type.
        price_range : int
            Desired price range (1-4).
        aggregate_rating : float
            Desired minimum aggregate rating.
        votes : int
            Representative vote count (used for feature matching).
        online_delivery : str
            'Yes' or 'No'.
        table_booking : str
            'Yes' or 'No'.

        Returns
        -------
        np.ndarray
            User preference vector matching the combined feature space.
        """
        # --- Text part ---
        user_text = f"{city.lower()} {cuisine.lower().replace(',', ' ')}"
        user_tfidf = self.tfidf.transform([user_text]).toarray()

        # --- Numerical part ---
        # Encode delivery / booking
        delivery_val = 1 if online_delivery.strip().lower() == 'yes' else 0
        booking_val = 1 if table_booking.strip().lower() == 'yes' else 0

        # Scale numerical features using the same range as the data
        # We need to manually scale to [0, 1] using the data's min/max
        price_min = self.df['Price range'].min()
        price_max = self.df['Price range'].max()
        rating_min = self.df['Aggregate rating'].min()
        rating_max = self.df['Aggregate rating'].max()
        votes_min = self.df['Votes'].min()
        votes_max = self.df['Votes'].max()

        # Avoid division by zero
        def safe_scale(val, vmin, vmax):
            if vmax == vmin:
                return 0.0
            return (val - vmin) / (vmax - vmin)

        user_num = np.array([[
            safe_scale(price_range, price_min, price_max),
            safe_scale(aggregate_rating, rating_min, rating_max),
            safe_scale(votes, votes_min, votes_max),
            delivery_val,
            booking_val
        ]])

        # --- Combine ---
        user_vector = np.hstack([user_tfidf, user_num])
        return user_vector

    def recommend(self, city: str = '', cuisine: str = '',
                  price_range: int = 2, minimum_rating: float = 0.0,
                  online_delivery: str = 'No', table_booking: str = 'No',
                  top_n: int = 10) -> pd.DataFrame:
        """
        Recommend restaurants based on user preferences using
        content-based filtering with cosine similarity.

        Strategy:
        1. Build a user-preference vector.
        2. Compute cosine similarity against all restaurants.
        3. Apply hard filters (minimum rating).
        4. Rank by similarity score.
        5. If exact city/cuisine match yields enough results, prefer those;
           otherwise, fall back to similarity ranking across all data.

        Parameters
        ----------
        city : str
            Preferred city (case-insensitive). Empty string means no filter.
        cuisine : str
            Preferred cuisine (case-insensitive). Empty string means no filter.
        price_range : int
            Desired price range (1-4).
        minimum_rating : float
            Minimum aggregate rating filter.
        online_delivery : str
            'Yes' or 'No'.
        table_booking : str
            'Yes' or 'No'.
        top_n : int
            Number of recommendations to return.

        Returns
        -------
        pd.DataFrame
            Top N recommended restaurants with details and similarity scores.
        """
        # Use median votes as a representative value for the user vector
        median_votes = int(self.df['Votes'].median())

        # Build the user preference vector
        user_vector = self._build_user_vector(
            city=city,
            cuisine=cuisine,
            price_range=price_range,
            aggregate_rating=minimum_rating,
            votes=median_votes,
            online_delivery=online_delivery,
            table_booking=table_booking
        )

        # Compute cosine similarity between user and all restaurants
        similarities = cosine_similarity(user_vector, self.combined_matrix)[0]

        # Add similarity scores to a working copy
        results = self.df.copy()
        results['Similarity Score'] = similarities

        # --- Apply minimum rating filter ---
        results = results[results['Aggregate rating'] >= minimum_rating]

        # --- Try filtered results first (city + cuisine match) ---
        filtered = results.copy()

        if city.strip():
            city_match = filtered['City'].str.lower().str.strip() == city.lower().strip()
            city_filtered = filtered[city_match]
            # Only apply city filter if it returns enough results
            if len(city_filtered) >= top_n:
                filtered = city_filtered
            elif len(city_filtered) > 0:
                # Use city-matched results even if fewer than top_n
                filtered = city_filtered
            # else: keep full results (city not found or no matches)

        if cuisine.strip():
            cuisine_match = filtered['Cuisines'].str.lower().str.contains(
                cuisine.lower().strip(), na=False
            )
            cuisine_filtered = filtered[cuisine_match]
            if len(cuisine_filtered) >= min(top_n, 3):
                filtered = cuisine_filtered
            # else: keep current filtered (cuisine not available in filtered set)

        # --- Sort by similarity score (descending) ---
        filtered = filtered.sort_values('Similarity Score', ascending=False)

        # --- Select top N ---
        top_results = filtered.head(top_n)

        # --- Build clean output ---
        output_cols = [
            'Restaurant Name', 'City', 'Cuisines', 'Price range',
            'Aggregate rating', 'Votes', 'Has Online delivery',
            'Has Table booking', 'Similarity Score'
        ]
        output = top_results[output_cols].copy()
        output.insert(0, 'Rank', range(1, len(output) + 1))
        output = output.reset_index(drop=True)

        # Round similarity score for readability
        output['Similarity Score'] = output['Similarity Score'].round(4)

        return output


def print_recommendations(recommendations: pd.DataFrame,
                          user_prefs: dict) -> None:
    """
    Display recommendations in a formatted table with user preferences.

    Parameters
    ----------
    recommendations : pd.DataFrame
        The recommendation results from recommender.recommend().
    user_prefs : dict
        Dictionary of user preferences for display.
    """
    print("\n" + "=" * 80)
    print("  RESTAURANT RECOMMENDATIONS")
    print("=" * 80)

    # Display user preferences
    print("\n  User Preferences:")
    print("  " + "-" * 40)
    for key, value in user_prefs.items():
        print(f"    {key:25s}: {value}")

    # Display results
    if len(recommendations) == 0:
        print("\n  [!] No restaurants found matching your preferences.")
        print("      Try relaxing your filters (lower rating, different city/cuisine).")
    else:
        print(f"\n  Found {len(recommendations)} recommendation(s):\n")
        # Print as a formatted table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.max_colwidth', 30)
        print(recommendations.to_string(index=False))

    print("\n" + "=" * 80)
