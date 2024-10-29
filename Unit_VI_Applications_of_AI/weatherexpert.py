#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources (only run these once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")  # Added download for omw-1.4

# Define a simple dataset of weather rules
weather_rules = {
    "sunny,warm": "It is likely to be a nice day. Enjoy outdoor activities!",
    "rainy,cold": "Expect rain and chilly weather. Carry an umbrella!",
    "sunny,cool": "A cool but sunny day is expected. Perfect for a walk!",
    "cloudy,warm": "It might be cloudy, but the weather will be warm.",
    "stormy": "Severe weather warning! Stay indoors!",
    "snowy": "Expect snowfall. Drive safely and stay warm!",
}

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_input(text):
    # Tokenize the input
    tokens = word_tokenize(text.lower())
    # Remove stopwords and lemmatize
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]
    return filtered_tokens


def forecast_weather(conditions):
    # Join conditions to match rule keys
    conditions_joined = ",".join(conditions)
    conditions_sorted = ",".join(sorted(conditions))
    # Check against both joined and sorted versions
    forecast = weather_rules.get(
        conditions_joined,
        weather_rules.get(
            conditions_sorted,
            "Weather conditions unclear. Please provide more details.",
        ),
    )
    return forecast


# Accept user input
user_input = input("Describe the weather conditions (e.g., 'sunny, warm'): ")

# Remove quotes and strip whitespace
user_input = user_input.replace("'", "").replace('"', "").strip()

# Preprocess the user input
user_conditions = preprocess_input(user_input)

# Display the processed conditions and weather forecast
print("Processed Conditions:", user_conditions)
print("Weather Forecast:", forecast_weather(user_conditions))


# In[ ]:
