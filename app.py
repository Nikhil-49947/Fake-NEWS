
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import PassiveAggressiveClassifier
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string


# initialise flask
app = Flask(__name__, template_folder='templates')
model = pickle.load(open('fakeNews.pkl', 'rb'))
df = pd.read_csv('https://media.githubusercontent.com/media/Uday0456/Fake-NEWS/master/news.csv')




#Extracting 'reviews' for processing
news_data=df.copy()
news_data=news_data[['news']].reset_index(drop=True)


x=news_data['news']
y=df['output']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=7)
tfidf = TfidfVectorizer(max_features=5000,ngram_range=(2,2),lowercase=True,stop_words='english')
def fake_news_detect(news):
    tfid_x_train = tfidf.fit_transform(x_train)
    tfid_x_test = tfidf.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfidf.transform(input_data)
    prediction = model.predict(vectorized_input_data)
    return prediction



# launch home page
@app.route('/', methods=['GET'])
def home():
    return render_template('fakenews.html')


@app.route('/', methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        pred=fake_news_detect(message)
        print(pred)
        return render_template('fakenews.html',prediction=pred)
    else:
        return render_template('fakenews.html',prediction='Something went wrong')


if __name__ == '__main__':
    app.run(debug=True)
