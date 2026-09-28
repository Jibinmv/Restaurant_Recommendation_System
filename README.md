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
git clone https://github.com/YOUR-USERNAME/Restaurant_Recommendation_System.git
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

