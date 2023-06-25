from flask import Flask, render_template, request
from googletrans import Translator
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request, json
from nltk.corpus import wordnet
from decimal import Decimal
from preprocess import data_cleaning
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def getAntonym(text):
    if text.find("not")>=0:
        text=text.lower()
        i=0
        while i<len(text)-3:
            if text[i]=="n" and text[i+1]=="o" and text[i+2]=="t":
                b=i
                if i+3<len(text) and text[i+3]==" ":
                    word=""
                    i+=4
                    while i<len(text) and text[i]!=" ":
                        word+=text[i]
                        i+=1
                    print(word)
                    if word=="like":
                        antonym="disliked"
                    elif word=="liked":
                        antonym="disliked"
                    elif word=="love":
                        antonym="hated"
                    elif word=="loved":
                        antonym="hated"
                    elif word=='good':
                        antonym='bad'
                    elif word=='bad':
                        antonym='good'
                    elif word=='rated':
                        antonym='unrated'
                    elif word=='rate':
                        antonym='unrate'
                    else:
                        print("here")
                        for syn in wordnet.synsets(word):
                            for lemma in syn.lemmas():
                                print(1,lemma)
                                for antonym in lemma.antonyms():
                                    print("done")
                                    antonym=antonym.name()
                                    break
                    text=text.replace("not "+word,antonym)
                    print(text)
                    i=b+len(antonym)-1
            else:
                i+=1
    return text

def predictMovie(text):
    text=text.replace("okay","ok")
    text=getAntonym(text)
    with open('models/lstm_movie_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=20)
    predictor = tf.keras.models.load_model('models/lstm_movie.h5')
    score=predictor.predict(np.array(text))[0][0]*100
    score=round(float(score),2)
    return score

def predictFood(text):
    with open('models/foodtokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=100, padding='post', truncating='post')
    predictor = tf.keras.models.load_model('models/food.h5')
    score=predictor.predict([text])[0][0]*100
    score=round(float(score),2)
    return score

def predictBook(text):
    text=text.replace("okay","ok")
    text=getAntonym(text)
    with open('models/lstm_book_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=20)
    predictor = tf.keras.models.load_model('models/lstm_book.h5')
    score=predictor.predict(np.array(text))[0][0]*100
    score=round(float(score),2)
    return score

def fetchComment(videoid):
    result=[]
    for i in range(len(videoid)):
            if videoid[i]=='?' and videoid[i+1]=="v" and videoid[i+2]=="=":
                videoid=videoid[i+3:]
                break
    url=f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={videoid}&maxResults=100&textFormat=plainText&key="+os.getenv("yt_key")
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    for i in data['items']:
        c=i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        if c!="":
            result.append(c)
    flag=0
    while data.get("nextPageToken",0) and flag==0:
        url+=f"&pageToken={data['nextPageToken']}"
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())
        for i in data['items']:
            c=i["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            if c!="":
                result.append(c)
            if len(result)==300:
                flag=1
                break
        print(result)
        for i in range(len(result)):
            result[i]=Translator().translate(result[i],dest='en').text
    return result

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method=='POST':
        print(os.getenv("yt_key"))
        data = request.form
        files = request.files
        text=data["inputtext"]
        result=[{'text':"Please enter text or upload file",'sentiment':'N/A','score':'N/A'}]
        if text!="":
            result[0]['text']=text
            if data['language']!='english':
                text=Translator().translate(text,src='hi',dest='en').text
            text=data_cleaning(text)
            print(text)
            if data['domain']=='movie':
                score=predictMovie(text)
            elif data['domain']=='food':
                score=predictFood(text)
            elif data['domain']=='book':
                score=predictBook(text)
            else:
                predictor = tf.keras.models.load_model('models/movie.h5',custom_objects={'KerasLayer':hub.KerasLayer})
                score=predictor.predict([text])
            result[0]['score']=score
            if score<45:
                result[0]['sentiment']='Negative'
                result[0]['score']=round(score-45,4)
            elif score>55:
                result[0]['sentiment']='Positive'
                result[0]['score']=round(score-55,4)
            else:
                result[0]['sentiment']='Neutral'
                result[0]['score']=0
        else:
            f=files.get('inputfile',0)
            if f:
                f= pd.read_excel(f)
                pos,neg,neutral=0,0,0
                for i in f:
                    mylist=f[i].tolist()
                    mylist.insert(0,i)
                for i in range(len(mylist)):
                    text=mylist[i]
                    if i:
                        result.append({})
                    result[i]['text']=text
                    if data['domain']=='movie':
                        score=predictMovie(text)
                    elif data['domain']=='food':
                        score=predictFood(text)
                    elif data['domain']=='book':
                        score=predictBook(text)
                    else:
                        predictor = tf.keras.models.load_model('models/movie.h5',custom_objects={'KerasLayer':hub.KerasLayer})
                        score=predictor.predict([text])
                    result[i]['score']=score
                    if score<45:
                        result[i]['sentiment']='Negative'
                        result[i]['score']=round(score-45,4)
                        neg+=1
                    elif score>55:
                        result[i]['sentiment']='Positive'
                        result[i]['score']=round(score-55,4)
                        pos+=1
                    else:
                        result[i]['sentiment']='Neutral'
                        result[i]['score']=0
                        neutral+=1
                return render_template('index.html',result=result,piedata=[pos,neg,neutral])
            else:
                return render_template('index.html',result=result)
        return render_template('index.html',result=result)
    return render_template('index.html')

@app.route('/package', methods=['POST', 'GET'])
def package():
    if request.method=='POST':
        data = request.form
        files = request.files
        text=data["inputtext"]
        result=[{'text':"Please enter text or upload file",'sentiment':'N/A','score':'N/A'}]
        if text!="":
            result[0]['text']=text
            if data['language']!='english':
                text=Translator().translate(text,src='hi',dest='en').text
            text=data_cleaning(text)
            print(text)
            if data['domain']=='movie':
                score=predictMovie(text)
            elif data['domain']=='food':
                score=predictFood(text)
            elif data['domain']=='book':
                score=predictBook(text)
            else:
                predictor = tf.keras.models.load_model('models/movie.h5',custom_objects={'KerasLayer':hub.KerasLayer})
                score=predictor.predict([text])
            result[0]['score']=score
            if score<45:
                result[0]['sentiment']='Negative'
                result[0]['score']=round(score-45,4)
            elif score>55:
                result[0]['sentiment']='Positive'
                result[0]['score']=round(score-55,4)
            else:
                result[0]['sentiment']='Neutral'
                result[0]['score']=0
        else:
            f=files.get('inputfile',0)
            if f:
                f= pd.read_excel(f)
                pos,neg,neutral=0,0,0
                for i in f:
                    mylist=f[i].tolist()
                    mylist.insert(0,i)
                for i in range(len(mylist)):
                    text=mylist[i]
                    if i:
                        result.append({})
                    result[i]['text']=text
                    if data['domain']=='movie':
                        score=predictMovie(text)
                    elif data['domain']=='food':
                        score=predictFood(text)
                    elif data['domain']=='book':
                        score=predictBook(text)
                    else:
                        predictor = tf.keras.models.load_model('models/movie.h5',custom_objects={'KerasLayer':hub.KerasLayer})
                        score=predictor.predict([text])
                    result[i]['score']=score
                    if score<45:
                        result[i]['sentiment']='Negative'
                        result[i]['score']=round(score-45,4)
                        neg+=1
                    elif score>55:
                        result[i]['sentiment']='Positive'
                        result[i]['score']=round(score-55,4)
                        pos+=1
                    else:
                        result[i]['sentiment']='Neutral'
                        result[i]['score']=0
                        neutral+=1
                return render_template('index.html',result=result,piedata=[pos,neg,neutral])
            else:
                return json.dumps(result)
        return json.dumps(result)
    return render_template('index.html')

@app.route('/youtube', methods=['POST', 'GET'])
def youtube():
    if request.method=='POST':
        result=[{'text':"Please enter text or upload file",'sentiment':'N/A','score':'N/A'}]
        videoid=str(request.form['videoid'])
        comments=fetchComment(videoid)
        print(comments)
        return render_template('youtube.html',result=result)
    return render_template('youtube.html')
 
if __name__ == '__main__':
    app.run(debug=True)