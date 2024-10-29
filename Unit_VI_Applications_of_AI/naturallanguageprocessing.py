#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources (only run these once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Sample text
text = "Natural Language Processing is a fascinating field of AI that focuses on understanding human language."

# 1. Word Tokenization
words = word_tokenize(text)
print("Words:", words)

# 2. Stopwords Removal
stop_words = set(stopwords.words("english"))
filtered_words = [
    word for word in words if word.isalnum() and word.lower() not in stop_words
]
print("Filtered Words (Stopwords Removed):", filtered_words)

# 3. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("Stemmed Words:", stemmed_words)

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("Lemmatized Words:", lemmatized_words)


# In[ ]:
