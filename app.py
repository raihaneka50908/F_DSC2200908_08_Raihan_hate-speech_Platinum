from flask import Flask,jsonify
from flask import request
from flasgger import Swagger,LazyString,LazyJSONEncoder# Ubah dari LazyString menjadi LazyJSONEncoder
from flasgger import swag_from
import pandas as pd

import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import re
from keras.models import load_model

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nlp_id.lemmatizer import Lemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

app=Flask(__name__)
app.json_encoder=LazyJSONEncoder
swagger_template=dict(
    info={
        'title':LazyString(lambda:'Dokumentasi API Untuk Klasifikasi Sentiment'),
        'version':LazyString(lambda:'1.0.0'),
        'description':LazyString(lambda:'Dokumentasi API'),
    },
    host=LazyString(lambda:request.host)
)

swagger_config={
    'headers':[],
    'specs':[
        {
            'endpoint':'docs',
            'route':'/docs.json',
        }
    ],
    'static_url_path':'/flasgger_static',
    'swagger_ui':True,
    'specs_route':'/docs/'
}

swagger=Swagger(app,template=swagger_template,config=swagger_config)

#####################################
alay_dict=pd.read_csv("Program Platinum Challange/DATA/new_kamusalay.csv",encoding='latin1',header=None)
alay_dict=alay_dict.rename(columns={0:'Original',1:'Baku'})

kasar_dict=pd.read_csv("Program Platinum Challange/DATA/abusive.csv",encoding='latin1')
kasar_dict['Kata_Sensor']="a1b2c3d4e5f6" #Inisiasi kata ganti untuk kata-kata yang kasar dengan kata "disensor"

#print(kasar_dict)

alay_dict_map = dict(zip(alay_dict['Original'], alay_dict['Baku']))
kasar_dict_map = dict(zip(kasar_dict['ABUSIVE'],kasar_dict['Kata_Sensor']))

file_h5_dl= "/Volumes/WorkWorkWor/Folder_Fold_Recovery/For Learning/Bootcamp Binar/Model Deep Learning h5/model_dl_yang_telah_dilatih.h5"
file_pkl_mlpc = "Program Platinum Challange/model_MLPClassifier.pkl"

model_deep_learning_lstm_h5 = load_model(file_h5_dl)

######################################
with open(file_pkl_mlpc,"rb") as file:
    model1=pickle.load(file)

with open("Program Platinum Challange/vectorizer.pkl","rb") as file:
    vectorizer = pickle.load(file)
######################################
#####################################

######################################
def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])


def sensor_kata_kasar(text):
    return ' '.join([kasar_dict_map[word] if word in kasar_dict_map else word for word in text.split(' ')])


def preprocess(TextYangInginDiPreProcess):
    #Tahap Pertama Adalah Membuat semua huruf menjadi huruf kecil atau lower
    text = TextYangInginDiPreProcess.lower()

    #Tahap Kedua adalah menghilangkan non alpha numeric character pada text
    text = re.sub('[^0-9a-zA-Z]+',' ',text)

    #Tahap Ketiga adalah menghilangkan char tidak penting
    text=re.sub('\n',' ',text) #Menghilangkan new line pada data
    text=re.sub('rt',' ',text) #Menghilangkan kata-kata retweet
    text=re.sub('user',' ',text) #Menghilangkan kata-kata user
    text=re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) #Menghilangkan  URL
    text=re.sub(' +',' ',text) #Menghilangkan ekstra spasi

    #Tahap keempat adalah membuat map terhadap kata-kata "alay" dan mengubah nya menjadi kata yang baku
    text=normalize_alay(text)

    #Tahap kelima adalah mensensor kata kasar dengan kata "disensor"
    text=sensor_kata_kasar(text)
    return text

def menghilangkan_kata_umum_dengan_stopword(text):
    # Mengambil daftar stopwords bahasa Indonesia dari NLTK
    stop_words = set(stopwords.words('indonesian'))

    # Memisahkan teks menjadi kata-kata
    words = text.split()

    # Menghapus stopwords dari teks
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Menggabungkan kata-kata yang tersisa menjadi teks tanpa stopwords
    filtered_text = ' '.join(filtered_words)

    return filtered_text

def mengurangi_variasi_kata_dengan_steeming(teks):
    Stemming = StemmerFactory()
    Fact = Stemming.create_stemmer()
    teks = Fact.stem(teks)
    return teks

def preprocess_semua_kata(teks):
    teks = preprocess(teks)
    teks = menghilangkan_kata_umum_dengan_stopword(teks)
    teks = mengurangi_variasi_kata_dengan_steeming(teks)
    return teks

max_features = 100000
def tokenisasi_kata(kata,max_features=max_features):
  tokenizer = Tokenizer(num_words=max_features,split=' ',lower=None)
  tokenizer.fit_on_texts(kata)
  kata_token = tokenizer.texts_to_sequences(kata)
  pad_kata = pad_sequences(kata_token)
  return pad_kata,tokenizer.word_index

######################################

@swag_from("docs/hello_world.yml",methods=['GET'])
@app.route('/',methods=['GET'])
def Hello_world():
    json_response={
        'status_code':200,
        'description':"Menyapa",
        'data':'Hello World',
    }
    response_data=jsonify(json_response)
    return response_data

#Masukan Teks
@swag_from("docs/lstm.yml",methods=['POST'])
@app.route("/lstm",methods=['POST'])
def dnn_lstm(model=model_deep_learning_lstm_h5):
  sentiment = ['negative','neutral','positive']
  max_features = 100000
  teks = request.form.get('text')
  teks = preprocess_semua_kata(teks)
  tokenizer = Tokenizer(num_words=max_features,split=' ',lower=None)
  tokenizer.fit_on_texts(teks)
  kata_token = tokenizer.texts_to_sequences(teks)
  pad_kata = pad_sequences(kata_token)
  prediksi = model.predict(pad_kata)
  polarity = np.argmax(prediksi[0])
  jenis_sentimen = sentiment[polarity]

  json_response = {
      'status_code':200,
      'description': 'Hasil Analisis Sentimen Dengan DNN LSTM',
      'jenis_sentimen':jenis_sentimen
  }
  response_data = jsonify(json_response)
  return response_data

#Masukan Teks
@swag_from("docs/cnn.yml",methods=['POST'])
@app.route("/cnn",methods=['POST'])
def nn_mlpc(model=model1):
    text = request.form.get('text')
    preprocessed_text = preprocess_semua_kata(text)
    text_vector = vectorizer.transform([preprocessed_text])
    sentiment_pred = model.predict(text_vector)

    json_response={
        'status_code':200,
        'description':'Hasil Analisis Sentimen Dengan MLPClassifier',
        'jenis_sentimen':str(sentiment_pred)
    }

    response_data = jsonify(json_response)
    return response_data

#Masukan File CSV
@swag_from("docs/lstm_csv.yml",methods=['POST'])
@app.route("/lstm_csv",methods=['POST'])
def dnn_lstm_csv(model=model_deep_learning_lstm_h5):
    max_features = 100000
    sentiment = ['negative','neutral','positive']
    file = request.files.getlist('file')[0]
    df = pd.read_csv(file,encoding='latin1')
    df = df[:5]
    df['Tweet'] = df['Tweet'].apply(preprocess_semua_kata)
    sentimen = []
    tokenizer = Tokenizer(num_words=max_features,split=' ',lower=None)
    for i in df['Tweet']:
        tokenizer.fit_on_texts(i)
        kata_token = tokenizer.texts_to_sequences(i)
        pad_kata = pad_sequences(kata_token)
        prediksi = model.predict(pad_kata)
        polarity = np.argmax(prediksi[0])
        jenis_sentimen = sentiment[polarity]
        sentimen.append(jenis_sentimen)
    
    json_response={
        'status_code':200,
        'description':'Hasil Analisis Sentimen Dengan MLPClassifier',
        'jenis_sentimen':sentimen
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/cnn_csv.yml",methods=['POST'])
@app.route("/cnn_csv",methods=['POST'])
#Masukan File CSV
def nn_mlpc_csv(model=model1):
    file = request.files.getlist('file')[0]
    df = pd.read_csv(file,encoding='latin1')
    df = df[:5]
    df['Tweet'] = df['Tweet'].apply(preprocess_semua_kata)
    sentimen = []
    for i in df['Tweet']:
        text_vector = vectorizer.transform([i])
        sentiment_pred = model.predict(text_vector)
        sentiment_pred = sentiment_pred[0]
        sentimen.append(sentiment_pred)
    
    json_response={
        'status_code':200,
        'description':'Hasil Analisis Sentimen Dengan MLPClassifier',
        'jenis_sentimen':sentimen
    }
    response_data = jsonify(json_response)
    return response_data






if __name__=="__main__":
    app.run()