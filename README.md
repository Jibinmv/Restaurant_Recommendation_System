
#  🍽️ Restaurant Recommendation System

## Project Objective

This project implements a **content-based restaurant recommendation system** as part of the Cognifyz Technologies Machine Learning Internship (Task 2). The system recommends restaurants based on user preferences by analyzing restaurant features and finding the most similar matches using machine learning techniques.

## Dataset

The project uses the `restaurants.csv` dataset containing information about **9,551 restaurants** across 141 cities worldwide. The dataset includes:

- Restaurant names and locations
- Cuisines served
- Price range (1-4)
- Aggregate ratings and votes
- Online delivery and table booking availability

## Features Used for Recommendation

| Feature | Type | Role |
|---------|------|------|
| City | Text | TF-IDF vectorized |
| Cuisines | Text | TF-IDF vectorized |
| Price range | Numerical | Min-Max scaled |
| Aggregate rating | Numerical | Min-Max scaled |
| Votes | Numerical | Min-Max scaled |
| Has Online delivery | Binary | Encoded (Yes=1, No=0) |
| Has Table booking | Binary | Encoded (Yes=1, No=0) |

**Features NOT used:** Restaurant ID, Address, Locality, Rating color, Rating text.

## How It Works

### 1. Data Preprocessing
- Load CSV with encoding handling
- Clean column names (BOM removal)
- Handle missing values (Cuisines → "Unknown", numericals → median)
- Normalize text (strip whitespace, standardize casing)

### 2. Content-Based Filtering

Content-based filtering recommends items based on the similarity between item features and user preferences. Unlike collaborative filtering (which needs user-item interaction history), content-based filtering works with item attributes alone.

### 3. TF-IDF Vectorization

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts text data into numerical vectors:

- **Term Frequency (TF)**: How often a word appears in a document
- **Inverse Document Frequency (IDF)**: How rare a word is across all documents
- **TF-IDF = TF × IDF**: Words that are frequent in a document but rare overall get higher scores

We apply TF-IDF to combined City and Cuisines text to capture location-cuisine patterns.

### 4. Cosine Similarity

**Cosine similarity** measures the angle between two vectors:

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

- Value ranges from 0 (completely different) to 1 (identical)
- A user preference vector is constructed and compared against all restaurant vectors
- Restaurants with higher cosine similarity are better matches

### 5. Feature Combination

The final feature matrix combines:
- TF-IDF vectors (text features)
- Min-Max scaled numerical features
- Binary encoded categorical features

## Project Structure

```
Cognifyz_Task2_Restaurant_Recommendation/
├── data/
│   └── restaurants.csv          # Dataset
├── outputs/
│   ├── figures/                 # Visualizations
│   └── results/                 # CSV results
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Data loading & cleaning
│   ├── recommender.py           # Recommendation engine
│   └── evaluation.py            # Evaluation & visualization
├── main.py                      # Main script
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Main Script

```bash
python main.py
```

This will:
1. Preprocess the dataset
2. Build the recommendation engine
3. Test with 3 sample user profiles
4. Evaluate recommendation quality
5. Save results and visualizations

### Run the Streamlit App

```bash
streamlit run app.py
```

This launches an interactive web interface where you can:
- Select your preferred city, cuisine, price range, etc.
- Click "Recommend Restaurants" to get personalized recommendations
- View results in a clean, interactive table

## Example User Preferences

### Sample 1
- City: New Delhi
- Cuisine: North Indian
- Price Range: 2
- Minimum Rating: 3.5
- Online Delivery: Yes
- Table Booking: No

### Sample 2
- City: Mumbai
- Cuisine: Italian
- Price Range: 3
- Minimum Rating: 4.0
- Online Delivery: Yes
- Table Booking: Yes

### Sample 3
- City: Bangalore
- Cuisine: Chinese
- Price Range: 2
- Minimum Rating: 3.5
- Online Delivery: No
- Table Booking: No

## Expected Output

For each sample, the system displays:
- Ranked list of recommended restaurants
- Restaurant details (name, city, cuisines, price range, rating, votes)
- Delivery and booking availability
- Similarity score (0 to 1)

Evaluation metrics include:
- Average rating and similarity score
- Percentage match for each preference criterion

## Limitations

1. **No user interaction data**: Content-based filtering only uses item features, not user behavior or ratings history.
2. **Cold start for new cuisines**: If a cuisine type isn't in the dataset, recommendations will fall back to similarity-based ranking.
3. **City coverage**: The dataset is heavily weighted toward New Delhi (~5,400 restaurants). Cities with fewer restaurants may produce less diverse recommendations.
4. **Static dataset**: Recommendations don't update with new restaurants or changing user preferences over time.
5. **No personalization over time**: The system doesn't learn from user feedback or past interactions.
6. **Simple quality metrics**: Evaluation metrics measure preference matching, not actual user satisfaction.

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **scikit-learn** - TF-IDF, cosine similarity, scaling
- **matplotlib & seaborn** - Visualizations
- **streamlit** - Web application interface
- **numpy** - Numerical computing

## Author

Cognifyz Technologies Machine Learning Internship - Task 2
=======
🍽️ Restaurant Recommendation System

A machine learning project that recommends restaurants based on restaurant attributes and user preferences. The project processes restaurant data, prepares relevant features, and generates recommendation results through a structured recommendation pipeline.

🎯 Objective
Build a restaurant recommendation system that helps users discover suitable restaurants using information such as:

Location

Cuisine

Price range

Restaurant ratings

Customer votes

Online delivery

Table booking

Other available restaurant attributes

🔄 Workflow
Restaurant Dataset
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Preparation
        ↓
Recommendation Logic / Model
        ↓
Restaurant Matching
        ↓
Recommended Restaurants

📊 Dataset
The project uses a restaurant dataset containing information such as:

Restaurant Name

City

Locality

Cuisines

Average Cost for Two

Price Range

Aggregate Rating

Votes

Table Booking

Online Delivery

Delivery Availability

Main dataset: restaurants.csv

✨ Key Features
Restaurant data preprocessing

Handling of restaurant attributes

User-oriented restaurant recommendations

Cuisine-based filtering

Location-based filtering

Rating and price consideration

Recommendation result generation

Structured output and analysis

📁 Project Structure
Restaurant_Recommendation/
│
├── data/
│   └── Dataset and supporting data
│
├── outputs/
│   └── Generated results and visualizations
│
├── ScreenShots/
│   └── Project screenshots
│
├── src/
│   └── Source code
│
├── venv/
│   └── Local Python virtual environment
│
├── .gitignore
├── app.py
├── main.py
├── README.md
├── requirements.txt
└── restaurants.csv
venv/ is a local virtual environment and should not be uploaded to GitHub.

🛠️ Technologies
Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

Streamlit

🚀 Installation
1. Clone the repository
git clone https://github.com/Jibinmv/Restaurant_Recommendation_System.git
cd Restaurant_Recommendation_System
2. Create a virtual environment
python -m venv venv
3. Activate it on Windows
venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
▶️ Usage
Run the main processing pipeline:

python main.py
If the Streamlit interface is included in the current implementation, launch it with:

python -m streamlit run app.py
Then open the local URL displayed in the terminal.

📌 Recommendation Factors
The recommendation system can use multiple restaurant attributes to identify restaurants that match user requirements, including:

Preferred cuisine

Preferred location

Price range

Restaurant rating

Customer votes

Online delivery availability

Table booking availability

📂 Outputs
Generated results, visualizations, and analysis files are stored in:

outputs/
🔮 Future Improvements
Personalized user profiles

Content-based recommendation

Collaborative filtering

Hybrid recommendation models

Improved ranking algorithms

Review sentiment analysis

Location-aware recommendations

Interactive recommendation dashboard

Explainable recommendations

🎓 Learning Outcomes
This project provides practical experience in:

Data preprocessing

Feature engineering

Recommendation systems

Machine learning

Data analysis

Model development

Python project organization

Streamlit application development

Git and GitHub

👨‍💻 Author
Jibin
B.Tech Computer Science Engineering Student

📜 Disclaimer
This project is developed for educational purposes. Recommendations are generated from the available dataset and may vary depending on the quality, completeness, and characteristics of the data.

⭐ If you find this project useful, consider giving the repository a star.

>>>>>>> c5b0e7c5aaeb1ecf092ce8c2444b49874453db9d
