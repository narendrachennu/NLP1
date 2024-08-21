#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
df.head()


# In[3]:


import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()


# In[4]:


corpus = []
for i in range(len(df)):
    rp = re.sub('[a_zA_Z]',' ',df['message'][i])
    rp = rp.lower()
    rp = rp.split()
    rp = [ps.stem(word) for word in rp if not word in set(stopwords.words('english'))]
    rp = ' '.join(rp)
    corpus.append(rp)


# In[5]:


# Preprocess data
X = df['message']
y = df['label']


# In[6]:


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Build the pipeline

# In[7]:


pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])


# In[8]:


# Train the model
pipeline.fit(X_train, y_train)


# In[9]:


# Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))


# # Save the model

# In[10]:


import joblib
joblib.dump(pipeline, 'spam_classifier_model.pkl')


# # streamlit app

# In[12]:


import streamlit as st
import joblib

# Load the pre-trained model
model = joblib.load('spam_classifier_model.pkl')

# Streamlit app title
st.title('Email Spam Classifier')

# Input form for user email
email_text = st.text_area("Enter the email content here:")

if st.button('Classify'):
    if email_text:
        # Predict the label
        prediction = model.predict([email_text])
        label = 'Spam' if prediction[0] == 'spam' else 'Not Spam'
        st.write(f'The email is classified as: **{label}**')
    else:
        st.write("Please enter email content for classification.")


# In[ ]:




