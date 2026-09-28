

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set visual style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def evaluate_recommendations(recommendations: pd.DataFrame,
                             user_prefs: dict,
                             sample_label: str = 'Sample') -> dict:
    """
    Evaluate the quality of a set of recommendations against user preferences.

    Quality Indicators:
    - Number of recommendations returned
    - Average rating of recommended restaurants
    - Average similarity score
    - % matching the requested price range
    - % matching the requested city
    - % matching the requested cuisine
    - % matching online delivery preference
    - % matching table booking preference

    NOTE: These are simple quality indicators for a content-based
    recommendation system. They measure how well the recommendations
    match the stated preferences, NOT real user satisfaction. Real-world
    evaluation would require user studies and A/B testing.

    Parameters
    ----------
    recommendations : pd.DataFrame
        The recommendation output from the recommender.
    user_prefs : dict
        Dictionary with keys: city, cuisine, price_range,
        minimum_rating, online_delivery, table_booking.
    sample_label : str
        Label for this evaluation sample.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    n = len(recommendations)

    if n == 0:
        return {
            'Sample': sample_label,
            'Num Recommendations': 0,
            'Avg Rating': 0,
            'Avg Similarity Score': 0,
            'Price Range Match (%)': 0,
            'City Match (%)': 0,
            'Cuisine Match (%)': 0,
            'Online Delivery Match (%)': 0,
            'Table Booking Match (%)': 0
        }

    # --- Calculate metrics ---

    avg_rating = recommendations['Aggregate rating'].mean()
    avg_similarity = recommendations['Similarity Score'].mean()

    # Price range match
    price_match = (
        (recommendations['Price range'] == user_prefs.get('price_range', 0))
        .sum() / n * 100
    )

    # City match
    city_pref = user_prefs.get('city', '').lower().strip()
    if city_pref:
        city_match = (
            recommendations['City'].str.lower().str.strip() == city_pref
        ).sum() / n * 100
    else:
        city_match = 100.0  # No city filter = all match

    # Cuisine match
    cuisine_pref = user_prefs.get('cuisine', '').lower().strip()
    if cuisine_pref:
        cuisine_match = (
            recommendations['Cuisines'].str.lower().str.contains(
                cuisine_pref, na=False
            )
        ).sum() / n * 100
    else:
        cuisine_match = 100.0

    # Online delivery match
    delivery_pref = user_prefs.get('online_delivery', '').strip().lower()
    if delivery_pref:
        delivery_match = (
            recommendations['Has Online delivery'].str.lower().str.strip()
            == delivery_pref
        ).sum() / n * 100
    else:
        delivery_match = 100.0

    # Table booking match
    booking_pref = user_prefs.get('table_booking', '').strip().lower()
    if booking_pref:
        booking_match = (
            recommendations['Has Table booking'].str.lower().str.strip()
            == booking_pref
        ).sum() / n * 100
    else:
        booking_match = 100.0

    metrics = {
        'Sample': sample_label,
        'Num Recommendations': n,
        'Avg Rating': round(avg_rating, 2),
        'Avg Similarity Score': round(avg_similarity, 4),
        'Price Range Match (%)': round(price_match, 1),
        'City Match (%)': round(city_match, 1),
        'Cuisine Match (%)': round(cuisine_match, 1),
        'Online Delivery Match (%)': round(delivery_match, 1),
        'Table Booking Match (%)': round(booking_match, 1)
    }

    return metrics


def print_evaluation(metrics: dict) -> None:
    """
    Print evaluation metrics in a formatted display.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics dictionary.
    """
    print("\n" + "-" * 60)
    print(f"  EVALUATION: {metrics['Sample']}")
    print("-" * 60)
    for key, value in metrics.items():
        if key == 'Sample':
            continue
        print(f"  {key:35s}: {value}")
    print("-" * 60)

    # Explanation
    print("\n  NOTE: These quality indicators measure how well the")
    print("  recommendations match the stated user preferences.")
    print("  They are NOT a measure of real user satisfaction.")
    print("  Real-world evaluation requires user studies and A/B testing.")


def save_evaluation_report(all_metrics: list, output_path: str) -> None:
    """
    Save combined evaluation metrics to a CSV file.

    Parameters
    ----------
    all_metrics : list of dict
        List of evaluation metric dictionaries from multiple samples.
    output_path : str
        File path to save the CSV.
    """
    eval_df = pd.DataFrame(all_metrics)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    eval_df.to_csv(output_path, index=False)
    print(f"\n[INFO] Evaluation report saved to: {output_path}")


# =========================================================================
# VISUALIZATIONS
# =========================================================================

def plot_similarity_scores(recommendations: pd.DataFrame,
                           sample_label: str,
                           output_dir: str) -> None:
    """
    Plot a horizontal bar chart of recommended restaurants by similarity score.

    Parameters
    ----------
    recommendations : pd.DataFrame
        Recommendation results.
    sample_label : str
        Label for the plot title.
    output_dir : str
        Directory to save the figure.
    """
    if len(recommendations) == 0:
        print(f"[WARN] No data to plot for {sample_label}.")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(recommendations) * 0.5)))

    # Sort by similarity for display
    plot_data = recommendations.sort_values('Similarity Score', ascending=True)

    # Truncate long restaurant names for readability
    names = plot_data['Restaurant Name'].str[:35]

    bars = ax.barh(names, plot_data['Similarity Score'],
                   color=sns.color_palette('viridis', len(plot_data)),
                   edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Similarity Score', fontsize=13)
    ax.set_ylabel('Restaurant', fontsize=13)
    ax.set_title(f'Top Recommended Restaurants - {sample_label}',
                 fontsize=15, fontweight='bold')

    # Add value labels on bars
    for bar, score in zip(bars, plot_data['Similarity Score']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{score:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir,
                            f'similarity_scores_{sample_label.lower().replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {filepath}")


def plot_rating_distribution(recommendations: pd.DataFrame,
                              sample_label: str,
                              output_dir: str) -> None:
    """
    Plot the rating distribution of recommended restaurants.

    Parameters
    ----------
    recommendations : pd.DataFrame
        Recommendation results.
    sample_label : str
        Label for the plot title.
    output_dir : str
        Directory to save the figure.
    """
    if len(recommendations) == 0:
        print(f"[WARN] No data to plot for {sample_label}.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ratings = recommendations['Aggregate rating']

    ax.hist(ratings, bins=np.arange(0, 5.5, 0.5), color='#2196F3',
            edgecolor='white', linewidth=1.2, alpha=0.85)

    ax.axvline(ratings.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean Rating: {ratings.mean():.2f}')

    ax.set_xlabel('Aggregate Rating', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'Rating Distribution of Recommendations - {sample_label}',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir,
                            f'rating_distribution_{sample_label.lower().replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {filepath}")


def plot_similarity_distribution(recommendations: pd.DataFrame,
                                  sample_label: str,
                                  output_dir: str) -> None:
    """
    Plot the similarity score distribution of recommended restaurants.

    Parameters
    ----------
    recommendations : pd.DataFrame
        Recommendation results.
    sample_label : str
        Label for the plot title.
    output_dir : str
        Directory to save the figure.
    """
    if len(recommendations) == 0:
        print(f"[WARN] No data to plot for {sample_label}.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    scores = recommendations['Similarity Score']

    ax.hist(scores, bins=15, color='#4CAF50', edgecolor='white',
            linewidth=1.2, alpha=0.85)

    ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean Similarity: {scores.mean():.4f}')

    ax.set_xlabel('Similarity Score', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'Similarity Score Distribution - {sample_label}',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir,
                            f'similarity_distribution_{sample_label.lower().replace(" ", "_")}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {filepath}")


def generate_all_visualizations(recommendations: pd.DataFrame,
                                 sample_label: str,
                                 output_dir: str) -> None:
    """
    Generate all three visualizations for a recommendation set.

    Parameters
    ----------
    recommendations : pd.DataFrame
        Recommendation results.
    sample_label : str
        Label for the sample.
    output_dir : str
        Directory to save figures.
    """
    plot_similarity_scores(recommendations, sample_label, output_dir)
    plot_rating_distribution(recommendations, sample_label, output_dir)
    plot_similarity_distribution(recommendations, sample_label, output_dir)
