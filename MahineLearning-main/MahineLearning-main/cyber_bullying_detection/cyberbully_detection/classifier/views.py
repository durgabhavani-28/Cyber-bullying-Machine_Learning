from django.shortcuts import render
from django.http import HttpResponse
from .forms import InputTextForm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
import string
import numpy as np
import pandas as pd
import nltk
from django.shortcuts import render, redirect
from django.http import HttpResponse
import csv

def clean_text(text):
    text = str(text).lower()
    text = tokenize_remove_punctuation(text)
    text = [word for word in text if not any(c.isdigit() for c in word)]
    text = remove_stopwords(text)
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tagging(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet(t[1])) for t in pos_tags]
    text = [t for t in text if len(t) > 1]
    return " ".join(text)
def tokenize_remove_punctuation(text):
    clean_text = []
    text = text.split(" ")
    for word in text:
        word = list(word)
        new_word = []
        for c in word:
            if c not in string.punctuation:
                new_word.append(c)
        word = "".join(new_word)
        clean_text.append(word)
    return clean_text
def remove_stopwords(text):
    clean_text = []
    for word in text:
        if word not in stopwords.words('english'):
            clean_text.append(word)
    return clean_text

def pos_tagging(text):
    try:
        tagged = nltk.pos_tag(text)
        return tagged
    except Exception as e:
        print(e)

def get_wordnet(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[pd.to_numeric(df['tagging'], errors='coerce').notnull()]
    df = df.dropna(axis=0)
    df.reset_index(inplace=True, drop=True)
    df['Processed_Comment'] = df['comments'].map(clean_text)
    return df

def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['Processed_Comment'], df['tagging'], random_state=42, test_size=0.20)
    count_vector = CountVectorizer()
    X_train = count_vector.fit_transform(X_train)
    X_test = count_vector.transform(X_test)
    model = LogisticRegression(C=10, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000000)
    model.fit(X_train, y_train)
    return model, count_vector
report={}
def classify_text(request):
    if request.method == 'POST':
        form = InputTextForm(request.POST)
        if form.is_valid():
            input_text = form.cleaned_data['input_text']
            preprocessed_input = clean_text(input_text)
            input_vector = count_vector.transform([preprocessed_input]) 
            predicted_category = model.predict(input_vector)[0]
            report[0]=[input_text,predicted_category]
            if predicted_category == 1:
                result = "Offensive"
            else:
                result = "Non-Offensive"
            return render(request, 'classifier/result.html', {'input_text': input_text, 'result': result, 'form': form})
    else:
        return render(request,'classifier/result.html')
df = load_and_preprocess_data('classifier/archive/sus.csv')
model, count_vector = train_model(df)

def render_result(request):
    return render(request, 'classifier/result.html')
def home(request):
    return render(request, 'classifier/index.html')
def render_about(request):
    return render(request, 'classifier/classify.html')
import csv

import subprocess
from django.shortcuts import redirect

def report_comment(request):
    if request.method == 'POST':
        report_data = report[0]
        report_data[1] = 1 - report_data[1]  
        csv_file_path = 'classifier/archive/sus.csv'

        with open(csv_file_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(report_data)
        report.clear()
        return redirect('render_result')

