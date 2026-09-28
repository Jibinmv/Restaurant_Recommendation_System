
import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess_dataset
from src.recommender import RestaurantRecommender, print_recommendations
from src.evaluation import (
    evaluate_recommendations,
    print_evaluation,
    save_evaluation_report,
    generate_all_visualizations
)



# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'restaurants.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'outputs', 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')

# Number of recommendations per sample
TOP_N = 10


# Define 3 sample user profiles to test the system
SAMPLE_PREFERENCES = [
    {
        'label': 'Sample 1',
        'city': 'New Delhi',
        'cuisine': 'North Indian',
        'price_range': 2,
        'minimum_rating': 3.5,
        'online_delivery': 'Yes',
        'table_booking': 'No'
    },
    {
        'label': 'Sample 2',
        'city': 'Mumbai',
        'cuisine': 'Italian',
        'price_range': 3,
        'minimum_rating': 4.0,
        'online_delivery': 'Yes',
        'table_booking': 'Yes'
    },
    {
        'label': 'Sample 3',
        'city': 'Bangalore',
        'cuisine': 'Chinese',
        'price_range': 2,
        'minimum_rating': 3.5,
        'online_delivery': 'No',
        'table_booking': 'No'
    }
]


def main():
    """
    Run the complete restaurant recommendation pipeline.
    """
    print("\n" + "=" * 70)
   
    
    print("=" * 70)

    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # =================================================================
    # STEP 1: PREPROCESS THE DATASET
    # =================================================================
    df = preprocess_dataset(DATA_PATH)

    # =================================================================
    # STEP 2: BUILD THE RECOMMENDATION ENGINE
    # =================================================================
    print("\n" + "#" * 60)
    print("  STEP 2: BUILDING RECOMMENDATION ENGINE")
    print("#" * 60 + "\n")

    recommender = RestaurantRecommender(df)

    # =================================================================
    # STEP 3: TEST WITH SAMPLE USER PROFILES
    # =================================================================
    print("\n" + "#" * 60)
    print("  STEP 3: TESTING WITH SAMPLE USER PROFILES")
    print("#" * 60)

    all_metrics = []  # Collect evaluation metrics for all samples

    for i, prefs in enumerate(SAMPLE_PREFERENCES, start=1):
        label = prefs['label']

        # Display header
        print(f"\n{'*' * 70}")
        print(f"  TESTING: {label}")
        print(f"  City: {prefs['city']}, Cuisine: {prefs['cuisine']}")
        print(f"  Price: {prefs['price_range']}, Min Rating: {prefs['minimum_rating']}")
        print(f"  Online Delivery: {prefs['online_delivery']}, "
              f"Table Booking: {prefs['table_booking']}")
        print(f"{'*' * 70}")

        # --- Check if the city/cuisine exist in the dataset ---
        city_exists = (
            df['City'].str.lower().str.strip() == prefs['city'].lower().strip()
        ).any()
        cuisine_exists = (
            df['Cuisines'].str.lower().str.contains(
                prefs['cuisine'].lower().strip(), na=False
            )
        ).any()

        if not city_exists:
            print(f"\n  [NOTE] City '{prefs['city']}' not found in dataset.")
            print(f"         The system will use similarity ranking across "
                  f"all cities.")

        if not cuisine_exists:
            print(f"\n  [NOTE] Cuisine '{prefs['cuisine']}' not found "
                  f"in dataset.")
            print(f"         The system will rank by closest matching "
                  f"cuisines.")

        # --- Get recommendations ---
        recommendations = recommender.recommend(
            city=prefs['city'],
            cuisine=prefs['cuisine'],
            price_range=prefs['price_range'],
            minimum_rating=prefs['minimum_rating'],
            online_delivery=prefs['online_delivery'],
            table_booking=prefs['table_booking'],
            top_n=TOP_N
        )

        # --- Display recommendations ---
        user_display = {
            'City': prefs['city'],
            'Cuisine': prefs['cuisine'],
            'Price Range': prefs['price_range'],
            'Minimum Rating': prefs['minimum_rating'],
            'Online Delivery': prefs['online_delivery'],
            'Table Booking': prefs['table_booking']
        }
        print_recommendations(recommendations, user_display)

        # --- Save recommendations to CSV ---
        csv_path = os.path.join(RESULTS_DIR, f'recommendations_sample_{i}.csv')
        recommendations.to_csv(csv_path, index=False)
        print(f"\n[INFO] Recommendations saved to: {csv_path}")

        # --- Evaluate recommendations ---
        metrics = evaluate_recommendations(recommendations, prefs, label)
        all_metrics.append(metrics)
        print_evaluation(metrics)

        # --- Generate visualizations ---
        generate_all_visualizations(recommendations, label, FIGURES_DIR)

   
    print("\n" + "#" * 60)
    print("  STEP 4: COMBINED EVALUATION REPORT")
    print("#" * 60)

    eval_path = os.path.join(RESULTS_DIR, 'recommendation_evaluation.csv')
    save_evaluation_report(all_metrics, eval_path)

    # Display combined evaluation
    eval_df = pd.DataFrame(all_metrics)
    print("\nCombined Evaluation Summary:")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(eval_df.to_string(index=False))
    print("-" * 80)

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n  Results saved to:  {RESULTS_DIR}")
    print(f"  Figures saved to:  {FIGURES_DIR}")
    print(f"\n  Files generated:")
    for f in os.listdir(RESULTS_DIR):
        print(f"    - outputs/results/{f}")
    for f in os.listdir(FIGURES_DIR):
        print(f"    - outputs/figures/{f}")
    print("\n  To launch the interactive Streamlit app:")
    print("    streamlit run app.py")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
