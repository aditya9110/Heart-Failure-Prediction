import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from collections import Counter

mpl.rcParams['font.size'] = 10
mpl.rcParams['text.color'] = 'black'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import chi2, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE

row0col1, row0col2 = st.columns([1, 5])
with row0col1:
    st.image('1249-heart-beat-outline.gif')
with row0col2:
    st.title('Heart Failure Prediction')

file = st.file_uploader('Insert Data')

if file:
    data = pd.read_csv(file.name)
    dependent_var = st.selectbox('Select the Dependent Variable', data.columns, index=len(data.columns)-1)

    st.header('Independent Variables')
    independent_var = data.drop(dependent_var, axis=1)
    row0a, row0b, row0c = [], [], []
    row0a.extend(st.columns([1, 1, 1.5, 1]))
    row0b.extend(st.columns(4))
    row0c.extend(st.columns(4))
    for i in range(4):
        row0a[i].write(independent_var.columns[i])
        row0b[i].write(independent_var.columns[i + 4])
        row0c[i].write(independent_var.columns[i + 8])

    st.header('Dependent Variable')
    st.subheader(dependent_var)
    st.write('The label here considered is DEATH_EVENT which depicts the death of a patient during the follow-up period'
             ' of the treatment.')

    # discrete vs continuous features
    discrete_features, continuous_features = [], []
    for feature in data.columns:
        if feature == dependent_var:
            continue
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

    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d}'.format(v=val)

        return my_format

    # pie plot
    st.header('Counts and Count Difference')
    row2col1, row2col2 = st.columns([1, 2.5])
    with row2col1:
        st.subheader('Select a feature')
        option1 = st.radio('', discrete_features + [dependent_var])
    with row2col2:
        if option1 == dependent_var:
            st.subheader(option1 + ' - Dependent Variable')
            fig = plt.figure(figsize=(2, 2))
            fig.set_size_inches(2, 2)
            plt.pie(Counter(labelled_data[option1]).values(), labels=Counter(labelled_data[option1]).keys(),
                    startangle=90, autopct=autopct_format(Counter(labelled_data[option1]).values()), radius=1.5,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1})
        else:
            st.subheader(option1)
            # st.write(feature_details[option1])
            fig = plt.figure(figsize=(2, 2))
            fig.set_size_inches(2, 2)
            # fig.patch.set_facecolor('#0E1117')
            plt.pie(Counter(labelled_data[option1]).values(), labels=Counter(labelled_data[option1]).keys(),
                    startangle=90, autopct=autopct_format(Counter(labelled_data[option1]).values()), radius=1.5,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1})
        st.pyplot(fig)

    if option1 != dependent_var:
        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
        transformed_data = labelled_data.groupby([option1, 'DEATH_EVENT']).size().reset_index(name='count')
        for i, u in enumerate(labelled_data[option1].unique()):
            ax[i].title.set_text(option1 + ' - ' + str(u))
            ax[i].pie(transformed_data[transformed_data[option1] == u]['count'],
                      labels=transformed_data[transformed_data[option1] == u]['DEATH_EVENT'], startangle=90,
                      autopct=autopct_format(transformed_data[transformed_data[option1] == u]['count']))
        st.pyplot(fig)

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
        selected_features = list(data.drop(dependent_var, axis=1).columns)
    if selected_features and (select_all_features or correlate):
        correlation = data[selected_features].corr()
        correlate = 0
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(correlation, cmap="Blues", vmax=0.6, annot=True, annot_kws={"size": 11}, fmt='.2f')
        st.pyplot(fig)

    # kde plot and histogram
    st.header('Probability Density')
    row4col1, row4col2 = st.columns([1, 2])
    with row4col1:
        st.subheader('Select a feature')
        option2 = st.radio('', continuous_features)

        visualize_type = st.radio('', ['KDE Plot', 'Histogram', 'Box Plot'])
    with row4col2:
        st.subheader(option2)
        fig = plt.figure(figsize=(8, 6))
        plt.xticks(fontsize=12)
        if visualize_type == 'Box Plot':
            sns.boxplot(x=option2, data=data)
        elif visualize_type == 'Histogram':
            sns.histplot(x=option2, data=data, bins=20)
        else:
            sns.kdeplot(x=option2, hue='DEATH_EVENT', data=labelled_data, fill=True)
        st.pyplot(fig)

    # st.write(feature_details[option2])

    # feature selection
    st.header('Feature Importance')

    fig = plt.figure(figsize=(6, 4))
    importance = mutual_info_classif(data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT'])
    feat_importance = pd.Series(importance, data.columns[:-1])
    feat_importance.plot(kind='barh', color='teal')
    st.pyplot(fig)

    best_features = SelectKBest(chi2, k=10)
    features_ranking = best_features.fit(data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT'])
    # features_ranking_df = pd.Series(features_ranking, data.columns[:-1])
    ranking_dictionary = {}
    for i in range(len(features_ranking.scores_)):
        ranking_dictionary[data.columns[i]] = round(features_ranking.scores_[i], 3)
    asc_sort = sorted(ranking_dictionary.items(), key=lambda kv: (kv[1], kv[0]))
    plot_data = {'features': [i for i, j in asc_sort], 'scores': list(np.arange(1, 13))}

    predict_button = st.button('Predict')
    predict_button = True
    if predict_button:
        X = data[['ejection_fraction', 'serum_creatinine', 'time']]
        y = data[dependent_var]

        X_train, X_testi, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

        lof = LocalOutlierFactor()
        outlier_rows = lof.fit_predict(X_train)

        mask = outlier_rows != -1
        X_train, y_train = X_train[mask], y_train[mask]

        oversample = RandomOverSampler(sampling_strategy='minority')
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_testi)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write('Accuracy: ' + str(accuracy_score(y_test, y_pred)*100))
        st.write(classification_report(y_test, y_pred))
        fig, ax = plt.subplots(figsize=(2, 2))
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='small')

        plt.xlabel('Predictions', fontsize=6)
        plt.ylabel('Actuals', fontsize=6)
        plt.title('Confusion Matrix', fontsize=6)
        st.pyplot(fig)

        predicted_data = pd.DataFrame(X_testi, columns=['ejection_fraction', 'serum_creatinine', 'time'])
        predicted_data['Risk Probability'] = 0

        for i in range(predicted_data.shape[0]):
            prediction_values = np.array(X_test[i]).reshape(1, -1)
            predicted_data['Risk Probability'].iloc[i] = model.predict_proba(prediction_values)[:, 1]

        sorting_state = st.selectbox('Sort By', ['Index', 'Risk - Ascending', 'Risk - Descending'])
        if sorting_state == 'Index':
            st.dataframe(predicted_data.sort_index())
        elif sorting_state == 'Risk - Ascending':
            st.dataframe(predicted_data.sort_values(by=['Risk Probability']))
        elif sorting_state == 'Risk - Descending':
            st.dataframe(predicted_data.sort_values(by=['Risk Probability'], ascending=False))

else:
    st.subheader('Please upload a file!')

