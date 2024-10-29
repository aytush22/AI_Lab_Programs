#!/usr/bin/env python
# coding: utf-8

# In[14]:


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

# Define a simple dataset of disease rules (this would typically be more extensive)
disease_rules = {
    "fever,cough": "Common Cold",
    "fever,cough,body ache": "Flu",
    "fever,headache,rash": "Dengue",
    "fever,body ache,weakness": "Malaria",
    "cough,shortness of breath": "Asthma",
    "chest pain,shortness of breath": "Heart Disease",
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


def diagnose(symptoms):
    # Join symptoms to match rule keys
    symptoms_joined = ",".join(symptoms)
    symptoms_sorted = ",".join(sorted(symptoms))
    # Check against both joined and sorted versions
    diagnosis = disease_rules.get(
        symptoms_joined,
        disease_rules.get(
            symptoms_sorted, "Unknown condition. Please consult a doctor."
        ),
    )
    return diagnosis


# Accept user input
user_input = input("Describe your symptoms (e.g., 'fever, cough'): ")

# Remove quotes and strip whitespace
user_input = user_input.replace("'", "").replace('"', "").strip()

# Preprocess the user input
user_symptoms = preprocess_input(user_input)

# Display the processed symptoms and diagnosis
print("Processed Symptoms:", user_symptoms)
print("Possible Diagnosis:", diagnose(user_symptoms))


# In[ ]:


# In[ ]:
