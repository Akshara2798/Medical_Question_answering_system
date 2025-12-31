import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Bidirectional, Conv1D, MaxPooling1D, LSTM, Dense, concatenate, Reshape, TimeDistributed
from tensorflow.keras.models import load_model
import warnings 
warnings.filterwarnings('ignore') 
# Define the CNN+LSTM Model
def cnn_lstm_model(general_similarity, domain_similarity):
    input_a = Input(shape=(1,))
    input_b = Input(shape=(1,))
    
    input_shape = (1, 1)
    reshaped_a = Reshape(input_shape)(input_a)
    reshaped_b = Reshape(input_shape)(input_b)
    
    cnn_a = create_cnn_layers()
    encoded_a = cnn_a(reshaped_a)

    cnn_b = create_cnn_layers()
    encoded_b = cnn_b(reshaped_b)
    
    merged = concatenate([encoded_a, encoded_b])
    
    lstm_input = Reshape((-1, 32))(merged)
    
    lstm_out = LSTM(32)(TimeDistributed(Dense(32))(lstm_input))
    
    prediction = Dense(1, activation='sigmoid')(lstm_out)
    
    model = Model(inputs=[input_a, input_b], outputs=prediction)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(general_similarity,domain_similarity,epochs=100,batch_size=32)
    return model
   

# Define the Shared CNN Layers
def create_cnn_layers():
    input_layer = Input(shape=(1, 1))
    x = Conv1D(64, 1, activation='relu')(input_layer)  # Use a kernel size of 1
    x = MaxPooling1D(1)(x)
    x = Conv1D(32, 1, activation='relu')(x)  # Use a kernel size of 1
    x = MaxPooling1D(1)(x)
    x = LSTM(16, return_sequences=True)(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=x)
    
    return model


def cnn_lstm_predict(general_similarity,domain_similarity):
    cnn_lstm_model=load_model("Model/CNN_LSTM.h5")
    cnn_lstm_prediction = cnn_lstm_model.predict([general_similarity, domain_similarity],verbose=0)[0][0]
    return cnn_lstm_prediction

#%%
def bilstm_model(general_similarity, domain_similarity):
    # Input layers for the similarity scores
    input_a = Input(shape=(1,))
    input_b = Input(shape=(1,))
    
    # Merge the input tensors
    merged = concatenate([input_a, input_b])
    
    # Reshape for BiLSTM
    merged_reshaped = Reshape((-1, 1))(merged)
    
    # Bidirectional LSTM layer
    bilstm_out = Bidirectional(LSTM(32))(merged_reshaped)
    
    # Output layer
    prediction = Dense(1, activation='sigmoid')(bilstm_out)
    
    # Create the model
    model = Model(inputs=[input_a, input_b], outputs=prediction)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(general_similarity,domain_similarity,epochs=100,batch_size=32)
    return model

def bilstm_predict(general_similarity,domain_similarity):
    bilstm_model=load_model("Model/BiLSTM.h5")
    bilstm_prediction = bilstm_model.predict([general_similarity, domain_similarity],verbose=0)[0][0]
    return bilstm_prediction

# %%

def autoencoder_model(general_similarity, domain_similarity):
   
    input_a = Input(shape=(1,))
    input_b = Input(shape=(1,))
    
    # Encoder
    encoded_a = Dense(32, activation='relu')(input_a)
    encoded_b = Dense(32, activation='relu')(input_b)
    
    # Merge the encoded representations
    merged = concatenate([encoded_a, encoded_b])
    
    # Decoder
    decoded = Dense(1, activation='sigmoid')(merged)
    
    # Create the autoencoder model
    autoencoder = Model(inputs=[input_a, input_b], outputs=decoded)
    
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    autoencoder.fit(general_similarity,domain_similarity,epochs=100,batch_size=32)
    return autoencoder

def autoencode_predict(general_similarity,domain_similarity):
    autoencoder_model=load_model("Model/autoencoder.h5")
    autoencode_prediction = autoencoder_model.predict([general_similarity, domain_similarity],verbose=0)[0][0]
    return autoencode_prediction


