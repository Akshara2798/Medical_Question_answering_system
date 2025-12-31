from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings 
warnings.filterwarnings('ignore') 

def Concept2Vec(sentences):

    model = Word2Vec.load("concept2vec_model.bin")
    vector = model.wv[sentences]
    return vector

def Cosine_Similarity(x,y):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    if len(y.shape) == 1:
        y = y.reshape(1, -1)
    if x.size == 0 or y.size == 0:
        return 0.0
    
    similarities = cosine_similarity(x, y)
    
    similarity = np.mean(similarities)
    return similarity
    


def General_similarity(question,answer):
    vectorizer = CountVectorizer().fit([question, answer])
    vector_question, vector_answer = vectorizer.transform([question, answer])
    

    cosine_sim = cosine_similarity(vector_question, vector_answer)
    
    return cosine_sim[0][0]