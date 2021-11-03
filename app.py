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
from sklearn.feature_selection import chi2, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE

row0col1, row0col2 = st.columns([1, 5])
with row0col1:
    st.image('1249-heart-beat-outline.gif')
with row0col2:
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

elif page == 'Visualize':
    st.header('Independent Variables')
    row0a, row0b, row0c = [], [], []
    row0a.extend(st.columns([1, 1, 1.5, 1]))
    row0b.extend(st.columns(4))
    row0c.extend(st.columns(4))
    for i in range(4):
        row0a[i].write(data.columns[i])
        row0b[i].write(data.columns[i + 4])
        row0c[i].write(data.columns[i + 8])

    st.header('Dependent Variable')
    st.subheader('DEATH_EVENT')
    st.write('The label here considered is DEATH_EVENT which depicts the death of a patient during the follow-up period'
             ' of the treatment.')

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

    feature_details = {'anaemia': 'Anaemia is a condition in which the blood doesn\'t have enough healthy red blood cells.',
                       'diabetes': 'Having diabetes means you are more likely to develop heart disease.',
                       'high_blood_pressure': 'High blood pressure is a condition in which the force of the blood against the artery walls is too high. High blood pressure can also cause ischemic heart disease.',
                       'sex': 'Sex of the patient.',
                       'smoking': 'It describe the smoking habit of the patient.',
                       'age': 'Age of the patient. The average age of patient is ' + str(round(data['age'].mean(), 2)),
                       'creatinine_phosphokinase': 'Determines the level of CPK enzyme in the blood. Higher the level, more chance of heart disease.',
                       'ejection_fraction': 'Percentage of fluid ejected from heart chamber with each contraction. A borderline ejection fraction can range between 41% and 50%.',
                       'platelets': 'A normal platelet count ranges from 150,000 to 450,000 platelets per microliter of blood.',
                       'serum_creatinine': 'The measure of serum creatinine may also be used to estimate how quickly the kidneys filter blood. 0.5 to 1.2 (mg/dL) is a normal range. Higher than that results in kidney impairment. ',
                       'serum_sodium': 'Indicates the level of serum sodium in the blood. A normal blood sodium level is between 135 and 145 milliequivalents per liter (mEq/L). If your sodium blood levels are too high or too low, it may mean that you have a problem with your kidneys, dehydration, or another medical condition.',
                       'time': 'Determines the follow-up period of each patient. Most patients who deceased were in the initial follow-up period.'
                       }

    # count plot
    st.header('Counts and Count Difference')
    row2col1, row2col2 = st.columns([1, 2.5])
    with row2col1:
        st.subheader('Select a feature')
        option1 = st.radio('', discrete_features + label)
    with row2col2:
        if option1 == 'DEATH_EVENT':
            st.subheader(option1 + ' - Dependent Variable')
            fig = plt.figure(figsize=(6, 4))
            plt.xticks(fontsize=12)
            sns.countplot(x=option1, data=labelled_data)
        else:
            st.subheader(option1)
            fig, ax = plt.subplots(1, 2, figsize=(6, 4))
            plt.xticks(fontsize=12)
            sns.countplot(ax=ax[0], x=option1, data=labelled_data).set_title('A) Count Difference')
            sns.countplot(ax=ax[1], x=option1, hue='DEATH_EVENT', data=labelled_data).set_title('B) Count wrt DEATH_EVENT')
        st.pyplot(fig)

    st.write(feature_details[option1])

    # correlation matrix
    st.header('Correlation Matrix')
    selected_features = st.multiselect('Features', data.columns[:-1])
    row3col1, row3col2 = st.columns([6, 1])
    correlate = 0
    with row3col1:
        select_all_features = st.checkbox('Select All Features')
    with row3col2:
        if st.button('Correlate'):
            correlate = 1
    if select_all_features:
        selected_features = list(data.columns[:-1])
    if selected_features and (select_all_features or correlate):
        correlation = data[selected_features].corr()
        correlate = 0
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(correlation, cmap="Blues", vmax=0.6, annot=True, annot_kws={"size": 11}, fmt='.2f')
        st.pyplot(fig)

    # kde plot
    st.header('Probability Density')
    row4col1, row4col2 = st.columns([1, 2])
    with row4col1:
        st.subheader('Select a feature')
        option2 = st.radio('', continuous_features)
        is_hist = st.checkbox('Histogram')
    with row4col2:
        st.subheader(option2)
        fig = plt.figure(figsize=(8, 6))
        plt.xticks(fontsize=12)
        if is_hist:
            sns.histplot(x=option2, data=data, bins=20)
        else:
            sns.kdeplot(x=option2, hue='DEATH_EVENT', data=labelled_data, fill=True)
        st.pyplot(fig)

    st.write(feature_details[option2])

    st.header('Feature Selection')

    importance = mutual_info_classif(data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT'])
    feat_importance = pd.Series(importance, data.columns[:-1])
    feat_importance.plot(kind='barh', color='teal')
    plt.show()

    best_features = SelectKBest(chi2, k=10)
    features_ranking = best_features.fit(data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT'])
    # features_ranking_df = pd.Series(features_ranking, data.columns[:-1])
    ranking_dictionary = {}
    for i in range(len(features_ranking.scores_)):
        ranking_dictionary[data.columns[i]] = round(features_ranking.scores_[i], 3)
    asc_sort = sorted(ranking_dictionary.items(), key=lambda kv: (kv[1], kv[0]))
    plot_data = {'features': [i for i, j in asc_sort], 'scores': list(np.arange(1, 13))}
    print(plot_data)

    # st.bar_chart(pd.Series(plot_data))


    def predict(features, algo, sample):
        pass


    # st.header('Model Building')
    # model_input_features = st.multiselect('Features', data.drop(['DEATH_EVENT'], axis=1).columns)
    # algos = ['Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier', 'Gradient Boosting', 'SVM']
    # row5col1, row5col2 = st.columns(2)
    # with row5col1:
    #     sample_method = st.radio('Select Sample Method', ['No Sampling', 'Over Sampling', 'SMOTE'])
    # with row5col2:
    #     ml_algorithm = st.selectbox('Select Algorithm', algos)
    # if st.button('Predict'):
    #     if model_input_features:
    #         predict(model_input_features, ml_algorithm, sample_method)

elif page == 'Predict':
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
