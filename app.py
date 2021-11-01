import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

st.title('Heart Failure Prediction')
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

row1col1, row1col2 = st.columns(2)

st.sidebar.title('Navigation')


default_visual_or_predict = 0
visual_or_predict = 0
with row1col1:
    if st.button('Visualize'):
        visual_or_predict = 0
with row1col2:
    if st.button('Predict'):
        visual_or_predict = 1

if visual_or_predict == 0:
    discrete_features, continuous_features = [], []
    for feature in data.columns:
        if feature == 'DEATH_EVENT':
            label = ['DEATH_EVENT']
        elif len(data[feature].unique()) >= 10:
            continuous_features.append(feature)
        else:
            discrete_features.append(feature)
    print('Discrete: ', discrete_features, '\n', 'Continuous', continuous_features)

    row2col1, row2col2 = st.columns(2)
    with row2col1:
        st.header('Discrete Features')
        for feature in discrete_features:
            st.write(feature)
    with row2col2:
        st.header('Continuous Features')
        for feature in continuous_features:
            st.write(feature)

    st.header('Counts and Count Difference')
    row3col1, row3col2 = st.columns([1, 2.5])
    with row3col1:
        st.subheader('Select a feature')
        option1 = st.radio('', discrete_features + label)
    with row3col2:
        st.subheader(option1)
        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        plt.xticks(fontsize=12)
        sns.countplot(ax=ax[0], x=option1, data=data)
        if option1 != 'DEATH_EVENT':
            sns.countplot(ax=ax[1], x=option1, hue='DEATH_EVENT', data=data)
        st.pyplot(fig)

    st.header('Correlation Matrix')
    row4col1, row4col2 = st.columns([6, 1])
    correlate = 0
    with row4col1:
        selected_features = st.multiselect('Features', data.columns)
    with row4col2:
        if st.button('Correlate'):
            correlate = 1
    select_all_features = st.checkbox('Select All Features')
    if select_all_features:
        selected_features = list(data.columns)
    if selected_features and (select_all_features or correlate):
        correlation = data[selected_features].corr()
        correlate = 0
        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(correlation, annot=True, annot_kws={"size": 7})
        st.pyplot(fig)

    st.header('Probability Density')
    row5col1, row5col2 = st.columns([1, 2])
    with row5col1:
        st.subheader('Select a feature')
        option2 = st.radio('', continuous_features)
    with row5col2:
        st.subheader(option2)
        fig = plt.figure(figsize=(8, 6))
        plt.xticks(fontsize=12)
        sns.kdeplot(x=option2, hue='DEATH_EVENT', data=data, fill=True)
        st.pyplot(fig)

elif visual_or_predict == 1:
    with st.form(key='my_form'):
        text_input = st.text_input(label='Enter some text')
        submit_button = st.form_submit_button(label='Predict')
