import pandas as pd
import numpy as np 
import spacy  # Importing spaCy library for natural language processing
from spacy.matcher import PhraseMatcher  # Importing PhraseMatcher from spaCy's matcher module
import re
from gensim.models import Word2Vec
from Concept2Vec import *
from SNOMED_CT import *
from transformers import TFBertModel
from tensorflow.keras.models import load_model
import keras
from Performance import *
import warnings 
warnings.filterwarnings('ignore') 
import tensorflow as tf
from RBTM import *
from keras.models import load_model
from Existing import *
import time

def Testing():
    #Data Preprocess
    class LemmaChase:
        def __init__(self):
            self.rules = [
                (r'(s|es|ies)$', ''),
                (r'([^aeiou])ies$', r'\1y'),
                (r'([aeiou]y)s$', r'\1'),
                (r'(us|ss|is)$', ''),
                (r'(s|x|ch|sh)$', ''),
                (r'(ie|us|ss|is)$', ''),
            ]
    
        def lemmatize(self, word):
            for pattern, replacement in self.rules:
                word = re.sub(pattern, replacement, word)
            return word
    
    
    print("******************************************************")
    print()
    print("             QUESTION ANSWERING SYSTEM                ")
    print()
    print("******************************************************")
    print()
    start_time = time.time()
    data=pd.read_csv("Datasets/overall.csv")
    questions=data['Question']
    answers=data['Answer']
    print()
    Questine=input("Enter The Question     : ").upper()
    print()
    
    '********************Preprocessing****************************'
    #---------------LemmaChase Lemmatizer-------------------------#
    lemmatizer = LemmaChase()
    vectorized_lemmatize = np.vectorize(lemmatizer.lemmatize)
    lemmatized_data = vectorized_lemmatize(Questine)
    lemmatized_data=lemmatized_data.item()
    split1=lemmatized_data.replace("?"," ")
    split2=split1.replace("("," ")
    split3=split2.replace(")"," ")
    split4=split3.replace("/"," ")
    questins=list(questions)
    questins=[x.upper() for x in questins]
    answers=list(answers)
    
    
    print("***************** Answers***************************")
    print()
    ques_pos = [index for index, value in enumerate(questins) if value == Questine]
    related_ans=[]
    a=0
    for ques_in in ques_pos:
        ans=answers[ques_in]
        related_ans.append(ans)
        print()
       
        print(f"Ans  {a+1}                  :\n{related_ans[a]}")
        print()
        a=a+1
    
    '******************** Annotation****************************'
    #---------------------------SNOMED-CT-------------------------#
    #Systematized Nomenclature of Medicine - Clinical Terms (SNOMED-CT)
    print()
    
    print("Finding Domain Related Words.....") 
    related_ans_=[x.upper() for x in related_ans]
    matched_ques_terms = SNOMED_CT([split3])
    print()
    print("Domain Realated Question Words        :")
    print()
    print(matched_ques_terms)
    print()
    print("Domain Realated Answer Words          :")
    matched_ans_terms=[]
    for ans in related_ans_:
        split1=ans.replace("."," ")
        split2=split1.replace("("," ")
        split3=split2.replace(")"," ")
        split4=split3.replace("/"," ")
        matched_a_terms=SNOMED_CT([split4])
        matched_ans_terms.append(matched_a_terms)
    
    print()
    print()
    print(matched_ans_terms)
    '******************** Semantic Enhancement ****************************'
    #---------------------------Concept2Vec Model -------------------------#
    print()
    print("Converting into vector.....")
    domain_questine_vector=Concept2Vec(matched_ques_terms)
    
    
    domain_ans_vectors=[]
    for i in matched_ans_terms:
        domain_ans_vector=Concept2Vec(i)
        domain_ans_vectors.append(domain_ans_vector)
    
    '******************** Cosine Similarity Calculation********************'    
    #-----------Cosine similarity on domain related words-----------------#    
    dom_similarity_scores=[] 
    for cosine in  domain_ans_vectors:
        cosine_sim=Cosine_Similarity(domain_questine_vector,cosine)
        dom_similarity_scores.append(cosine_sim)
        
    
    #-----------Cosine similarity on Genereal ques and ans-----------------# 
    gen_similarity_scores=[] 
    for general in  related_ans:
        cosine_sim=General_similarity(split3,general)
        gen_similarity_scores.append(cosine_sim)
    
    '******************** Cosine Similarity Calculation********************'
    #--------------------Hybrid gradient regression based transformer model (RBTM) ----------#    
    
    
    similarity=[]
    CNN_LSTM_similarity=[]
    print()
    print("Similarity Values ")
    cmt=0
    for gen,dom in zip(gen_similarity_scores,dom_similarity_scores):
        with keras.utils.custom_object_scope({'TFBertModel': TFBertModel}):
            model = load_model("Model/RBTM.h5")
        general_similarity = tf.constant([[gen]])
        domain_similarity = tf.constant([[dom]])
        num_additional_features = 10  
        additional_input_example = tf.random.uniform((1, num_additional_features))
        logits = model([general_similarity, domain_similarity, additional_input_example])
        logits_ = tf.constant([], dtype=tf.float32)
        value = logits.numpy()[0, 0] 
        print(f"Ans {cmt+1}  :   {value}")
        similarity.append(value)
        print()
        end_time = time.time()   
    
        #Existing 
        cnn_lstm=cnn_lstm_predict(general_similarity,domain_similarity)
        bilstm=bilstm_predict(general_similarity,domain_similarity)
        autoencoder=autoencode_predict(general_similarity,domain_similarity)
        print()
        
        cmt+=1
        
        
        
    highest_similarity_score=max(similarity)
    posision=similarity.index(highest_similarity_score)
    print("As per the similarity score the selected Answer is ") 
    print()
    print()
    print(related_ans[posision]) 
    
    cm_time = end_time - start_time
    return cm_time
    
    
Ex_time=Testing()   
Plot(Ex_time)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      
