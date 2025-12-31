import pandas as pd
import re
import numpy as np
from Concept2Vec import *
from sklearn.model_selection import train_test_split
from RBTM import *
from Existing import *
import warnings 
warnings.filterwarnings('ignore') 



def Main():
    
    '********************Preprocessing****************************'
    #---------------LemmaChase Lemmatizer-------------------------#
    traing_data=pd.read_csv("Datasets/train_data.csv")
    
    questine=traing_data['Questines']
    answer=traing_data['Answers']
    
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
    
    dataset=pd.read_csv("Datasets/Train_data.csv")
    data=dataset[["Questines","Answers","Topic","Split"]]
    data=data.to_numpy()
    lemmatizer = LemmaChase()
    vectorized_lemmatize = np.vectorize(lemmatizer.lemmatize)
    lemmatized_data = vectorized_lemmatize(data.flatten()).reshape(data.shape)
    DF = pd.DataFrame(lemmatized_data) 
    # np.save("files/Preprocess data.npy",lemmatized_data)
    
    #finding domain related words
    
    preprocessed_data=pd.read_csv("files/Preprocess_data.csv",encoding='latin')
    
    
    '******************** Annotation****************************'
    #---------------------------SNOMED-CT-------------------------#
    #Systematized Nomenclature of Medicine - Clinical Terms (SNOMED-CT)
    question=preprocessed_data.iloc[:, 1]
    question=list(question)
    answer=preprocessed_data.iloc[:, 2]
    answer=list(answer)
    
    domain_ques_word=[]
    domain_ans_word=[]
    
    for ques,ans in zip(question,answer):
        split1=ques.replace("."," ")
        split2=split1.replace("("," ")
        split3=split2.replace(")"," ")
        split4=split3.replace("/"," ")
        domain_ques=SNOMED_CT([split4])
        split5=ans.replace("."," ")
        split6=split5.replace("("," ")
        split7=split6.replace(")"," ")
        split8=split7.replace("/"," ")
        domain_ans=SNOMED_CT([split8])
        domain_ques_word.append(domain_ques)
        domain_ans_word.append(ans)
     
       
       
    
    '******************** Semantic Enhancement****************************'
    #---------------------------Concept2Vec Model -------------------------#
    domain_ques_word=np.load("files/domain_ques_words.npy",allow_pickle=True) 
    domain_ans_word=np.load("files/domain_ans_words.npy",allow_pickle=True)
    questine_vectors=[]
    answer_vectors=[]
    
    for ques_vec,ans_vec in zip(domain_ques_word,domain_answer_word):
        ques_vector=Concept2Vec(ques_vec)
        questine_vectors.append(ques_vector)
        ans_vector=Concept2Vec(ans_vec)
        answer_vectors.append(ans_vector)
        print(b)
       
    
    # np.save("files/Qus_vectors.npy",questine_vectors)
    # np.save("files/Answer_vectors.npy",answer_vectors)
    
    #Cosine similarity findling
    '******************** Cosine Similarity Calculation********************'    
    #-----------Cosine similarity on domain related words-----------------#    
    dom_cosine_similarities=[] 
    for ques_,ans_ in  zip(questine_vectors,answer_vectors):
        cosine_sim=Cosine_Similarity(ques_,ans_)
        dom_cosine_similarities.append(cosine_sim)
        print(c)
        
    # np.save("files/dom_cosine_similarities.npy",dom_cosine_similarities)
    preprocessed_data=pd.read_csv("files/Preprocess_data.csv",encoding='latin')
    
    question=preprocessed_data.iloc[:, 1]
    question=list(question)
    answer=preprocessed_data.iloc[:, 2]
    answer=list(answer)
    
    #-----------Cosine similarity on Genereal ques and ans-----------------# 
    general_cosine_similarities=[]
    for ques__,ans__ in  zip(question,answer):
        split1=ques__.replace("."," ")
        split2=split1.replace("("," ")
        split3=split2.replace(")"," ")
        split4=split3.replace("/"," ")
        
        split5=ans__.replace("."," ")
        split6=split5.replace("("," ")
        split7=split6.replace(")"," ")
        split8=split7.replace("/"," ")
        
        cosine_sim=General_similarity(split4,split8)
        general_cosine_similarities.append(cosine_sim)
        print(d)
       
    
    
    '******************** Cosine Similarity Calculation********************'
    #--------------------Hybrid gradient regression based transformer model (RBTM) ----------#    
    
    general_cosine_similarities=np.array(general_cosine_similarities)
    dom_cosine_similarities=np.array(dom_cosine_similarities)
    
    general_cosine_similarities=general_cosine_similarities[:len(dom_cosine_similarities)]
    general_cosine_similarities=np.array(general_cosine_similarities)
    dom_cosine_similarities=np.array(dom_cosine_similarities)
    x_train,x_test,y_train,y_test=train_test_split(general_cosine_similarities,dom_cosine_similarities,test_size=0.2)
    model = RBTM(x_train)
    history = fit_model(model, x_train, y_train, x_test, y_test)
    
    #Existing
    #cnn Lstm 
    #bilstm
    #autoencoder
    
    cnn_lstm=cnn_lstm_model(x_train,y_train)
    bilstm=bilstm_model(x_train,y_train)
    auto_encoder=autoencoder_model(x_train,y_train)
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
