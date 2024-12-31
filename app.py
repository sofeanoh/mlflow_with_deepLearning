#%% import
import pandas as pd
import numpy as np
import pickle
import streamlit as st 
import mlflow

#%% Create function for loading resources
@st.cache_resource #for caching purpose
def load_pickle(filepath):
    with open(filepath, "rb") as f:
        pickle_object = pickle.load(f)
    return pickle_object

#%% 
@st.cache_resource
def load_model(model_uri):
    model = mlflow.tensorflow.load_model(model_uri)
    return model

encoder = load_pickle('news_encoder.pkl')
model = load_model("runs:/39f68009d6e0464e81390f115f1dc864")

#%% 
st.title("News Classification")
with st.form("news_form"):
    news_input = st.text_area("Input your news aeticle here", height=200)
    submit = st.form_submit_button()
    
# structure the input, but here remember we are inputting one instead of a batch of 32, so wee need to expand dim
model_input = np.array(news_input, dtype=object)
model_input = np.expand_dims(model_input, axis=0)

#%%model prediction

prediction = np.argmax(model.predict(model_input), axis=1)
class_prediction = encoder.inverse_transform(prediction)

if submit:
    st.write(f"A(n) {class_prediction[0]} news")
# %%
