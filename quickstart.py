# git clone https://github.com/Mehrads/NLP-method
# pip install -r requirements.txt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import neattext.functions as nfx
import matplotlib.pyplot as plt
import plotly.express as plx
from sklearn.metrics import classification_report
import keras
from keras.layers import Embedding,Dense,LSTM,Bidirectional,GlobalMaxPooling1D,Input,Dropout
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tqdm import tqdm
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
text = "Am I weird I don't get affected by compliments if it's coming from someone I know irl but I feel really good when internet strangers do it"
def clean_text(text):
    text_length=[]
    cleaned_text=[]
    for sent in tqdm(text):
        sent=sent.lower()
        sent=nfx.remove_special_characters(sent)
        sent=nfx.remove_stopwords(sent)
        text_length.append(len(sent.split()))
        cleaned_text.append(sent)
    return cleaned_text,text_length

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            # This Dense layer's units will be dynamically set in the call method
            tf.keras.layers.Dense(head_size),  # Placeholder, will adjust dynamically
        ])
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        # Dynamically set the output dimension of the last Dense layer to match input dimension
        self.ffn.layers[-1].units = inputs.shape[-1]
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)



sample_text, sample_length = clean_text(text)
tokenizer=Tokenizer()

sample_text_seq=tokenizer.texts_to_sequences(sample_text)
sample_text_pad=pad_sequences(sample_text_seq,maxlen=50)


saved_model = load_model('hybrid_model.h5', custom_objects={'TransformerEncoder': TransformerEncoder})
sample_input = sample_text_pad
prediction = saved_model.predict(sample_input)

# Interpret the prediction
threshold = 0.5
classified_output = "suicide" if prediction[0][0] > threshold else "non-suicide"
print(f"Classification result: {classified_output}")

