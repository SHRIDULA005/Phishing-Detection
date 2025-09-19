import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv('phishing_data.csv')  
df['Result'] = np.where(df['Result'] == -1, 0, df['Result'])
target = df['Result']
features = df.drop(columns=['Result'])

def binary_classification_accuracy(actual, pred):
    print(f'Confusion matrix: \n{confusion_matrix(actual, pred)}')
    print(f'Accuracy score: \n{accuracy_score(actual, pred)}')
    print(f'Classification report: \n{classification_report(actual, pred)}')

folds = KFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(folds.split(features, target)):
    model = XGBClassifier()
    model.fit(np.array(features)[train_idx, :], np.array(target)[train_idx])
    predictions = model.predict(np.array(features)[val_idx, :])

    print(f'\n==== FOLD {fold + 1} ====')
    binary_classification_accuracy(np.array(target)[val_idx], predictions)
