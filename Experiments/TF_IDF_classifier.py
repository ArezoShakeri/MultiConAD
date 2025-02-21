import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_language', required=True)
parser.add_argument('--task', required=True)
parser.add_argument('--translated', required=True) # use "yes", if you want to the analysis using the English translated data 

args_slurm = parser.parse_args()

path_to_data_folder = "path_to_data_folder"
train_en = pd.read_json(path_to_data_folder + "/train_en_AD_Dem_info.jsonl", lines=True)
test_en = pd.read_json(path_to_data_folder + "/test_en_AD_Dem_info.jsonl", lines=True)
train_spa = pd.read_json(path_to_data_folder + "/translated_train_df_spa.jsonl", lines=True)
train_gr = pd.read_json(path_to_data_folder+"/translated_train_gr.jsonl", lines=True)
train_cha = pd.read_json(path_to_data_folder + "/translated_train_cha.jsonl", lines=True)
test_spa=pd.read_json(path_to_data_folder + "/translated_test_df_spa.jsonl", lines=True)
test_gr= pd.read_json(path_to_data_folder + "/translated_test_gr.jsonl", lines=True)
test_cha= pd.read_json(path_to_data_folder + "/translated_test_cha.jsonl", lines=True)

# Multi-lingual training and testing
train_dfs = [train_en, train_gr, train_cha, train_spa]
test_dfs = {
    'en': test_en,
    'gr': test_gr,
    'cha': test_cha,
    'spa': test_spa
}

# Mono-lingual training and testing
train_dfs = [train_spa]
test_dfs = {
    'spa': test_spa
}




# Add a column for translated text for English dataset

if args_slurm.translated== "yes":
    train_en['translated'] = train_en['Text_interviewer_participant']
    test_en['translated'] = test_en['Text_interviewer_participant']

def classify_language_dataset_TFIDF(train_dfs, test_dfs, test_language, random_state=42,task=None,translated=None):
    train_combined = pd.concat(train_dfs, ignore_index=True)
    if any(df.equals(train_en) for df in train_dfs):
        train_combined['Diagnosis'] = train_combined['Diagnosis'].replace('AD', 'Dementia')

    if task == "binary":
        train_combined = train_combined[train_combined['Diagnosis'] != 'MCI']

    if translated == "yes":
         X_train = train_combined['translated']
    else:
        X_train = train_combined['Text_interviewer_participant']
    y_train = train_combined['Diagnosis']
    
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    test_df = test_dfs[test_language]
    if any(df.equals(train_en) for df in train_dfs):
         test_df['Diagnosis'] = test_df['Diagnosis'].replace('AD', 'Dementia')
    if task == "binary":
        test_df = test_df[test_df['Diagnosis'] != 'MCI']
    
    if translated == "yes":
        X_test = test_df['translated']
    else:
        X_test = test_df['Text_interviewer_participant']
    
    y_test = test_df['Diagnosis']
    
    X_test_tfidf = tfidf.transform(X_test)
    
    classifiers = {
        'Decision Tree': (DecisionTreeClassifier(random_state=random_state), {'max_depth': [10, 20, 30]}),
        'Random Forest': (RandomForestClassifier(random_state=random_state), {'n_estimators': [50, 100, 200]}),
        'Naive Bayes': (MultinomialNB(), {'alpha': [0.5, 1.0, 1.5]}),
        'SVM': (SVC(random_state=random_state), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
        'Logistic Regression': (LogisticRegression(random_state=random_state), {'C': [0.1, 1, 10]})
    }
    
    for name, (clf, params) in classifiers.items():
        grid_search = GridSearchCV(clf, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train_tfidf, y_train)
        
        y_pred = grid_search.predict(X_test_tfidf)
        
        print(f"Classifier: {name}")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Test Set Language: {test_language}")
        print(classification_report(y_test, y_pred))
        print("\n")
    print("test dataset: ", test_language)
    for df in train_dfs:
        df_name = [name for name, value in globals().items() if value is df][0]
        print(f"DataFrame in training set: {df_name}")
    print(task)
    print("TF-IDF")
    print("Translation status: ",translated)

classify_language_dataset_TFIDF(train_dfs, test_dfs, args_slurm.test_language, task=args_slurm.task,translated=args_slurm.translated)
