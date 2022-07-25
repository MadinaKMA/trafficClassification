import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

st.title("Classification of things during traffic")
file_image = st.file_uploader('Upload image:', type=['png', 'jpeg', 'gif', 'svg'])
if file_image:
    st.image(file_image)
    img=PILImage.create(file_image)
    model=load_learner('transport_model_project.pkl')
    pred, pred_id, probs = model.predict(img)
    st.success(f'Prediction: {pred}')
    st.info(f'Probability: {probs[pred_id]*100:.1f}%')
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)