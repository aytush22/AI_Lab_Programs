#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Load dataset
df = pd.read_csv("playsheet_dataset.csv")

# Initialize individual LabelEncoders for each column
outlook_encoder = LabelEncoder()
temp_encoder = LabelEncoder()
humidity_encoder = LabelEncoder()
windy_encoder = LabelEncoder()

# Encode each feature
inputs = df.drop("Play", axis="columns")
inputs["outlook_n"] = outlook_encoder.fit_transform(inputs["Outlook"])
inputs["temp_n"] = temp_encoder.fit_transform(inputs["Temp"])
inputs["humidity_n"] = humidity_encoder.fit_transform(inputs["Humidity"])
inputs["windy_n"] = windy_encoder.fit_transform(inputs["Windy"])

# Drop original categorical columns
inputs_n = inputs.drop(["Outlook", "Temp", "Humidity", "Windy"], axis="columns")

# Define the target variable
target = df["Play"]

# Apply Gaussian Naive Bayes
classifier = GaussianNB()
classifier.fit(inputs_n, target)

# Print to confirm the classifier training is complete
print("GaussianNB Model Trained")


# In[6]:


# Example of predicting with the trained model
predicted = classifier.predict(inputs_n)
print(predicted)


# In[7]:


inputs


# In[8]:


inputs_n


# In[ ]:
