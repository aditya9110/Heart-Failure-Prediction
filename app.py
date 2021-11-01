import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2, SelectFromModel, SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE

st.title('Heart Failure Prediction')
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

labelled_data = data.copy()
labelled_data['anaemia'] = labelled_data['anaemia'].map({0: 'No', 1: 'Yes'})
labelled_data['diabetes'] = labelled_data['diabetes'].map({0: 'No', 1: 'Yes'})
labelled_data['high_blood_pressure'] = labelled_data['high_blood_pressure'].map({0: 'No', 1: 'Yes'})
labelled_data['sex'] = labelled_data['sex'].map({0: 'Female', 1: 'Male'})
labelled_data['smoking'] = labelled_data['smoking'].map({0: 'No', 1: 'Yes'})
labelled_data['DEATH_EVENT'] = labelled_data['DEATH_EVENT'].map({0: 'Alive', 1: 'Dead'})

st.sidebar.title('Navigation')
page = st.sidebar.radio('What would you like to do', ['Home', 'Visualize', 'Predict'])

if page == 'Home':
    st.write('* Cardiovascular diseases (CVDs) are the leading cause of death worldwide, killing an estimated 17.9 '
             'million people each year, accounting for 31% of all deaths.')
    st.write('* CVDs are a common cause of heart failure, and this dataset contains 12 variables that can be used to '
             'predict heart failure mortality.')
    st.write('* Most cardiovascular illnesses can be avoided by implementing population-wide programmes to address '
             'behavioural risk factors such as cigarette use, poor diet and obesity, physical inactivity, and'
             ' problematic alcohol consumption.')
    st.write('* People with cardiovascular disease or who are at high cardiovascular risk (due to the existence of'
             ' one or more risk factors such as hypertension, diabetes, hyperlipidemia, or previously existing illness)'
             ' require early identification and care, which can be greatly aided by a machine learning model.')

    st.subheader('Glance of the Data')
    st.dataframe(data.head())

    st.write('[Data Source - Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)')

if page == 'Visualize':
    # discrete vs continuous features
    discrete_features, continuous_features = [], []
    for feature in data.columns:
        if feature == 'DEATH_EVENT':
            label = ['DEATH_EVENT']
        elif len(data[feature].unique()) >= 10:
            continuous_features.append(feature)
        else:
            discrete_features.append(feature)
    # print('Discrete: ', discrete_features, '\n', 'Continuous', continuous_features)

    row1col1, row1col2 = st.columns(2)
    with row1col1:
        st.header('Discrete Features')
        for feature in discrete_features:
            st.write(feature)
    with row1col2:
        st.header('Continuous Features')
        for feature in continuous_features:
            st.write(feature)

    # count plot
    st.header('Counts and Count Difference')
    row2col1, row2col2 = st.columns([1, 2.5])
    with row2col1:
        st.subheader('Select a feature')
        option1 = st.radio('', discrete_features + label)
    with row2col2:
        if option1 == 'DEATH_EVENT':
            st.subheader(option1)
            fig = plt.figure(figsize=(6, 4))
            plt.xticks(fontsize=12)
            sns.countplot(x=option1, data=labelled_data)
        else:
            st.subheader(option1)
            fig, ax = plt.subplots(1, 2, figsize=(6, 4))
            plt.xticks(fontsize=12)
            sns.countplot(ax=ax[0], x=option1, data=labelled_data)
            sns.countplot(ax=ax[1], x=option1, hue='DEATH_EVENT', data=labelled_data)
        st.pyplot(fig)

    # correlation matrix
    st.header('Correlation Matrix')
    row3col1, row3col2 = st.columns([6, 1])
    correlate = 0
    with row3col1:
        selected_features = st.multiselect('Features', data.columns)
    with row3col2:
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

    # kde plot
    st.header('Probability Density')
    row4col1, row4col2 = st.columns([1, 2])
    with row4col1:
        st.subheader('Select a feature')
        option2 = st.radio('', continuous_features)
    with row4col2:
        st.subheader(option2)
        fig = plt.figure(figsize=(8, 6))
        plt.xticks(fontsize=12)
        sns.kdeplot(x=option2, hue='DEATH_EVENT', data=data, fill=True)
        st.pyplot(fig)

    st.header('Feature Selection')
    best_features = SelectKBest(chi2, k=10)
    features_ranking = best_features.fit(data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT'])
    # features_ranking_df = pd.Series(features_ranking, data.columns[:-1])
    ranking_dictionary = {}
    for i in range(len(features_ranking.scores_)):
        ranking_dictionary[data.columns[i]] = round(features_ranking.scores_[i], 3)
    print(ranking_dictionary)
    st.bar_chart(pd.Series(ranking_dictionary))

if page == 'Predict':
    with st.form(key='my_form'):
        name_input = st.text_input(label='Enter Name')
        age = st.slider('Age', min_value=30, max_value=100)
        sex = st.selectbox('Sex', ['Male', 'Female'])

        row1col1, row1col2, row1col3, row1col4 = st.columns(4)
        with row1col1:
            anaemia = st.radio('Anaemia', ['Yes', 'No'])
        with row1col2:
            diabetes = st.radio('Diabetes', ['Yes', 'No'])
        with row1col3:
            smoking = st.radio('Smoking', ['Yes', 'No'])
        with row1col4:
            high_blood_pressure = st.radio('High Blood Pressure', ['Yes', 'No'])

        creatinine_phosphokinase = st.slider('Level of CPK Enzyme (mcg/L)', min_value=10, max_value=8000)
        ejection_fraction = st.slider('Percentage of blood leaving the heart at each contraction (percentage)',
                                      min_value=10, max_value=100)
        platelets = st.slider('Platelets in the blood (kiloplatelets/mL)', min_value=20, max_value=900)
        serum_creatinine = st.slider('Level of serum creatinine in the blood (mg/dL)', min_value=0.5, max_value=10.0)
        serum_sodium = st.slider('Level of serum sodium in the blood (mEq/L)', min_value=100, max_value=150)
        time = st.slider('Follow-up period (days)', min_value=0, max_value=300)

        submit_button = st.form_submit_button(label='Predict')
