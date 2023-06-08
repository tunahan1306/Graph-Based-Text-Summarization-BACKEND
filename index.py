from flask import Flask, request
from flask_cors import CORS
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
from scipy.spatial.distance import cosine
from rouge import Rouge


nlp = spacy.load("en_core_web_sm")  # İngilizce modelini yükleyin (Örnek: "en_core_web_sm" İngilizce modeli için)


FILE_PATH = 'file-path/glove.6B.300d.txt'

nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')

embedding_index = {}
with open(FILE_PATH , 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embedding_index[word] = vector


stemmer = PorterStemmer()
stopwords_list = stopwords.words('english')

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/createGraph", methods=['POST'])
def createGraph():
    array = request.json['array']
    baslik = request.json['baslik']
    gelen_skor = request.json['skor']
    gelen_benzerlik = request.json['benzerlik']

    # Başlık Doğal Dil İşleme Adımlarının Uygulanması
    # - word_tokenize işlemi
    baslik_kelimeler = word_tokenize(baslik)
    # - Noktalama İşaretlerinin Kaldırılması
    temiz_kelimeler = [kelime for kelime in baslik_kelimeler if kelime not in string.punctuation]
    temiz_metin = ' '.join(temiz_kelimeler)
    # - Stopwords Kaldırılması
    baslik_kelimeler = word_tokenize(temiz_metin)
    filtreli_kelimeler = [kelime for kelime in baslik_kelimeler if kelime.lower() not in stopwords_list]
    filtreli_metin = ' '.join(filtreli_kelimeler)
    # - Kelimelerin kökünün bulunması işlemi
    baslik_kelimeler = word_tokenize(filtreli_metin)
    baslik_stemler = [stemmer.stem(kelime) for kelime in baslik_kelimeler]

    # tf - idf Kelimeleri
    tfidf_dict = calculate_tf_idf(preprocess_sentence(' '.join(array)))
    sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)

    top_10_percent = sorted_tfidf[:int(len(sorted_tfidf) * 0.1)]

    top_words = [word for word, tfidf in top_10_percent]

    texts = []
    dialogs = []
    
    for i in range(len(array)):
        #Özel İsimlerin Bulunması
        ozel_isimler = []
        numerik_veriler = []
        doc = nlp(array[i])
        kelimeler = word_tokenize(array[i])
        filtreli_kelimeler = [kelime for kelime in kelimeler if kelime.lower() not in stopwords_list]
        stemler = [stemmer.stem(kelime) for kelime in kelimeler]

        #Özel İsimleri Bulma
        for token in doc:
            if token.ent_type_ == "PERSON" or token.ent_type_ == "GPE" or token.ent_type_ == "ORG":
                ozel_isimler.append(token.text)
        
        #Numerik Veri Bulma
        for kelime in kelimeler:
            if kelime.isdigit():
                numerik_veriler.append(kelime)
        
        #Doğal Dil İşleme Adımlarının Uygulanması
        # - Noktalama İşaretlerinin Kaldırılması
        temiz_kelimeler = [kelime for kelime in kelimeler if kelime not in string.punctuation]
        temiz_metin = ' '.join(temiz_kelimeler)
        # - Stopwords Kaldırılması
        kelimeler = word_tokenize(temiz_metin)
        filtreli_kelimeler = [kelime for kelime in kelimeler if kelime.lower() not in stopwords_list]
        filtreli_metin = ' '.join(filtreli_kelimeler)
        # - Kelimelerin kökünün bulunması işlemi
        kelimeler = word_tokenize(filtreli_metin)
        stemler = [stemmer.stem(kelime) for kelime in kelimeler]
        
        #Özel İsim Skor Bulma
        ozelIsimSkor = len(ozel_isimler) / len(doc)
        #Numerik Veri Skor Bulma
        numerikVeriSkor = len(numerik_veriler) / len(kelimeler)
        #Başlık ile Aynı Kelime Skoru
        ortak = ortak_kelimeler(baslik_stemler, stemler)
        baslikSkor = len(ortak) / len(filtreli_kelimeler)
        

        #tf - idf skor bulma
        ortak = ortak_kelimeler(top_words, stemler)
        tfSkor = len(ortak) / len(doc)

        #cumle Skoru bulma
        cumleSkor = ozelIsimSkor + numerikVeriSkor + baslikSkor + tfSkor 
        

        texts.append({'key':i , 
                      'name':array[i] , 
                      #'ozel_isimler':ozel_isimler , 
                      'ozelIsimSkor':ozelIsimSkor , 
                      #'numerik_veriler':numerik_veriler , 
                      'numerikVeriSkor':numerikVeriSkor , 
                      #'stemler':stemler , 
                      #'baslik_stemler':baslik_stemler , 
                      #'ortak':ortak , 
                      'baslikSkor':baslikSkor , 
                      'tfSkor':tfSkor , 
                      #'top_words':top_words , 
                      'cumleSkor':cumleSkor , 
                      'color':'white',
                    })
        
    dugumSayisi = {}

    for text in texts:
        dugumSayisi[str(text['key'])] = 0
    
    for i in range(len(array)-1):
        for j in range(i+1,len(array),1):
            sentence1 = array[i]
            sentence2 = array[j]
            tokens1 = word_tokenize(sentence1)
            tokens2 = word_tokenize(sentence2)

            word_vectors1 = [embedding_index[word] for word in tokens1 if word in embedding_index]
            word_vectors2 = [embedding_index[word] for word in tokens2 if word in embedding_index]

            similarity = 1 - cosine(np.mean(word_vectors1, axis=0), np.mean(word_vectors2, axis=0))

            if similarity >= gelen_benzerlik:
                dugumSayisi[str(i)] += 1
                dugumSayisi[str(j)] += 1

            dialogs.append({'from':i ,
                            'to':j , 
                            'text':similarity,
                            'color':'green'
                            })
        
    #Düğüm Sayısının texts lere eklenmesi ve Cümle Skorunun tekrar hesaplanması
    for text in texts:
        if str(text['key']) in dugumSayisi:
            text['dugumSayisi'] =  dugumSayisi[str(text['key'])]
            text['cumleSkor'] += dugumSayisi[str(text['key'])] / (len(dugumSayisi) - 1)
        
        if text['cumleSkor'] >= gelen_skor:
            text['color'] = 'cyan'

    #Özet Oluşturma
    siralama = sorted(texts, key=lambda x: x["cumleSkor"], reverse=True)
    ilk_yari_dizi = siralama[:len(siralama)//2]
    siralama_new = sorted(ilk_yari_dizi, key=lambda x: x["key"], reverse=False)
    
    ozet = ''

    for text in siralama_new:
        ozet += text['name'] + '.'
    
    return {'texts':texts , 'dialogs':dialogs ,'gelen_benzerlik':gelen_benzerlik , 'gelen_skor':gelen_skor , 'dugumSayisi':dugumSayisi , 'ozet':ozet}

@app.route("/ozet", methods=['POST'])
def kiyasla():
    ozet = request.json['ozet']
    ozet_asil = request.json['ozet_asil']

    rouge = Rouge()
    scores = rouge.get_scores(ozet, ozet_asil)

    # Precision, Recall ve F1-score değerlerini alın
    rouge_1_precision = scores[0]['rouge-1']['p']
    rouge_1_recall = scores[0]['rouge-1']['r']
    rouge_1_f1_score = scores[0]['rouge-1']['f']

    # Yüzdelik benzerlik skoru hesaplama
    percentage_score = rouge_1_f1_score * 100

    deger = "{:.2f}%".format(percentage_score)
    return { 'scores':scores , 'deger':deger}

def ortak_kelimeler(dizi1, dizi2):
    set1 = set(dizi1)
    set2 = set(dizi2)
    ortak_kelimeler = set1.intersection(set2)
    return list(ortak_kelimeler)

def preprocess_sentence(sentence):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(sentence.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)

def calculate_tf_idf(sentence):
    preprocessed_sentence = preprocess_sentence(sentence)

    corpus = [preprocessed_sentence]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_values = tfidf_matrix.toarray()[0]

    tfidf_dict = dict(zip(feature_names, tfidf_values))

    return tfidf_dict

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)