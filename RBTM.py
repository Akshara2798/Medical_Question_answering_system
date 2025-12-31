import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.models import Model
import warnings 
warnings.filterwarnings('ignore') 


def RBTM(bert_model_name='bert-base-uncased'):
    # Define input layers
    num_classes=1
    num_additional_features = 10
    input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    additional_input = Input(shape=(num_additional_features,), dtype=tf.float32, name="additional_input")

    # Load pre-trained BERT model
    bert_model = TFBertModel.from_pretrained(bert_model_name)
    
    # Freeze BERT layers
    for layer in bert_model.layers:
        layer.trainable = False

    # Get BERT outputs
    bert_outputs = bert_model(input_ids, attention_mask=attention_mask)[0]
    pooled_output = bert_outputs[:, 0, :]  # Use the [CLS] token

    # Concatenate BERT output with additional features
    combined = Concatenate()([pooled_output, additional_input])

    # Add regression layer
    regression_output = Dense(num_classes, activation='linear', name='regression')(combined)

    # Define the model
    model = Model(inputs=[input_ids, attention_mask, additional_input], outputs=regression_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
   

    return model
    
def fit_model(model, x_train, y_train, x_val, y_val):  
    history = model.fit([x_train, y_train, x_train],epochs=100,batch_size=32)
    # history.save("Model/RBTM.h5")
    return history
    


