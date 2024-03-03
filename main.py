import pandas as pd
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action='ignore', category=UserWarning)


def normalise(X):
    scaler = MinMaxScaler()
    model = scaler.fit(X)
    scaled_data = model.transform(X)

    return scaled_data
# Loading Train data
tfidfColumns = np.load('tfidf-data/train-tfidf.npy')
embColumns = np.load('embedding-data/train-embeddings.npy')
data1 = pd.read_csv("raw-data/train.csv")


# Dropping the requirements and role feature and adding the TFIDF matrix
tfidf_df = pd.DataFrame(tfidfColumns)
all_data = pd.concat([data1, tfidf_df], axis=1)
data = all_data.drop('requirements_and_role', axis=1)

# Separating salary_bin and mean_salary from data, whilst adding them to the end of data
salary_bin = data.pop('salary_bin')
mean_salary = data.pop('mean_salary')
data = data.join(salary_bin)
data = data.join(mean_salary)
# data = data.head(8000)
salary_bin = salary_bin.head(8000)
mean_salary = mean_salary.head(8000)

# Processing for embedding data
emb_df = pd.DataFrame(embColumns)
embData = pd.concat([data1, emb_df], axis=1)
embData = embData.drop('requirements_and_role', axis=1)
embData = embData.drop('salary_bin', axis=1)
embData = embData.drop('mean_salary', axis=1)
embData = embData.drop('gender_code', axis=1)
emb_data = embData.head(8000)
emb_data = emb_data.iloc[:, 1:]
emb_unlabelled_data = embData.tail(5902)
emb_unlabelled_data = emb_unlabelled_data.drop('job_id', axis=1)


# Processing for combined TFIDF and Embeddings data
combinedData = pd.concat([embData, data], axis=1)
combinedData = combinedData.drop('salary_bin', axis=1)
combinedData = combinedData.drop('mean_salary', axis=1)
combinedData = combinedData.drop('gender_code', axis=1)
combinedData = combinedData.drop('job_id', axis=1)
combinedHead = combinedData.head(8000)
combinedTail = combinedData.tail(5902)


# Loading and processing embedding Valid data
emb_valid = np.load('embedding-data/valid-embeddings.npy')
emb_valid_data = pd.read_csv('raw-data/valid.csv')
emb_valid_data = pd.concat([emb_valid_data, pd.DataFrame(emb_valid)], axis=1)
emb_valid_data = emb_valid_data.drop('requirements_and_role', axis=1)
emb_valid_salary_bin = emb_valid_data.pop('salary_bin')
emb_valid_mean_salary = emb_valid_data.pop('mean_salary')
emb_valid_data = emb_valid_data.iloc[:, 2:]

# Loading embeddings Test data
emb_test = np.load('embedding-data/test-embeddings.npy')
raw_test = pd.read_csv('raw-data/test.csv')
emb_test_data = pd.concat([raw_test, pd.DataFrame(emb_test)], axis=1)
emb_test_data = emb_test_data.drop('requirements_and_role', axis=1)
emb_test_data = emb_test_data.iloc[:, 2:]

# Getting and processing unlabelled data
data2 = all_data.drop('requirements_and_role', axis=1)
data2 = data2.drop('salary_bin', axis=1)
data2 = data2.drop('gender_code', axis=1)
data2 = data2.drop('mean_salary', axis=1)
unlabelled_data = data2.tail(5902)

# Loading and processing TFIDF Valid data
tfidf_valid = np.load('tfidf-data/valid-tfidf.npy')
valid_data = pd.read_csv('raw-data/valid.csv')
valid_data = pd.concat([valid_data, pd.DataFrame(tfidf_valid)], axis=1)
valid_data = valid_data.drop('requirements_and_role', axis=1)
valid_salary_bin = valid_data.pop('salary_bin')
valid_mean_salary = valid_data.pop('mean_salary')

# Loading TFIDF Test data
tfidf_test = np.load('tfidf-data/test-tfidf.npy')
data3 = pd.read_csv('raw-data/test.csv')
test_data = pd.concat([data3, pd.DataFrame(tfidf_test)], axis=1)
test_data = test_data.drop('requirements_and_role', axis=1)

#
lgr = LogisticRegression()
mnb = MultinomialNB()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()

X1 = data.iloc[:, 2:-2]
test_data = test_data.drop('job_id', axis=1)
test_data = test_data.drop('gender_code', axis=1)

X2 = valid_data.iloc[:, 2:]


# normalisedEmb = np.abs(embData)
#lgr.fit(emb_data, salary_bin)

# combinedTest = pd.concat([emb_test_data, test_data], axis=1)
# combinedValid = pd.concat([emb_valid_data, X2], axis=1)
lgr.fit(emb_data, salary_bin)
print(classification_report(valid_salary_bin, lgr.predict(emb_valid_data), zero_division=0))
# normalisedEmbValid = np.abs(emb_valid_data)
# print(knn.score(normalisedEmbValid, emb_valid_salary_bin))

# lgr.fit(X1, salary_bin)

# Predicting Valid dataset with labelled data
print(lgr.score(emb_valid_data, valid_salary_bin))

models = [DummyClassifier(strategy='most_frequent'),
          GaussianNB(),
          BernoulliNB(),
          MultinomialNB(),
          DecisionTreeClassifier(),
          KNeighborsClassifier(), # When the n_neighbors parameter is not set in the KNeighborsClassifier() function, its default value is set to 5
          LogisticRegression(max_iter=1000)]
titles = ['Zero-R',
          'GNB',
          'BNB',
          'MNB',
          'Decision Tree',
          'KNN',
          'Logistic Regression']

for title, model in zip(titles, models):
    if title != 'MNB':
        train_data = emb_data
        val_data = emb_valid_data
    else:
        train_data = normalise(emb_data)
        val_data = normalise(emb_valid_data)
    model.fit(train_data, salary_bin)
    start = time.time()
    acc = np.mean(cross_val_score(model, val_data, valid_salary_bin, cv=10))
    end = time.time()
    t = end - start
    print(title, 'accuracy:', acc)

# combinedTest['salary_bin'] = lgr.predict(combinedTest)
# #
# combinedTest.to_csv('predictions13.csv')

unlabelled_data = unlabelled_data.iloc[:, 1:]

prev_length = 0

X_train, X_test, y_train, y_test = train_test_split(emb_data, salary_bin, test_size=0.2, random_state=42)

# Set the number of folds (K) and create the cross-validation object
k_folds = 10
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

best_fold = None
best_score = 0

# Perform cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    # Get the training and validation data for the current fold
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Create and train your model
    lgr.fit(X_train_fold, y_train_fold)

    # Make predictions on the validation set
    y_val_pred = lgr.predict(X_val_fold)

    # Evaluate the model's performance using a suitable metric
    score = accuracy_score(y_val_fold, y_val_pred)

    # Print the score for the current fold
    print(f"Fold {fold + 1} Accuracy: {score}")

    # Check if the current fold has the best score
    if score > best_score:
        best_score = score
        best_fold = fold
        best_X_train = X_train_fold
        best_y_train = y_train_fold



# Print the best fold and its corresponding score
print(f"\nBest Fold: {best_fold + 1} Accuracy: {best_score}")
lgr.fit(best_X_train, best_y_train)

print("LGR accuracy using the best fold: ",lgr.score(emb_valid_data, valid_salary_bin))


# Self-Training approach
while len(emb_unlabelled_data) > 0:
    # Fit classifier to labeled data
    lgr.fit(emb_data, salary_bin)

    # Predict labels for unlabeled data
    predicted_labels = lgr.predict(emb_unlabelled_data)

    # Find instances with high confidence
    confidence_scores = lgr.predict_proba(emb_unlabelled_data).max(axis=1)
    high_confidence_indices = np.where(confidence_scores > 0.65)[0]
    high_confidence_data = emb_unlabelled_data.iloc[high_confidence_indices]
    high_confidence_labels = predicted_labels[high_confidence_indices]

    # Add high confidence data to labeled data
    emb_data = pd.concat([emb_data, high_confidence_data])
    salary_bin = pd.concat([salary_bin, pd.Series(high_confidence_labels)])

    # Remove high confidence data from unlabeled data
    emb_unlabelled_data = emb_unlabelled_data.drop(index=high_confidence_data.index)

    print(f'Training set size: {len(emb_data)}')
    print(f'Unlabeled set size: {len(emb_unlabelled_data)}')

    if len(emb_data) == prev_length:
        break

    prev_length = len(emb_data)


lgr.fit(emb_data, salary_bin)
print(classification_report(valid_salary_bin, lgr.predict(emb_valid_data), zero_division=0))
print("LGR Score after self-training: ",lgr.score(emb_valid_data, emb_valid_salary_bin))

emb_test_data['salary_bin'] = lgr.predict(emb_test_data)
# #
emb_test_data.to_csv('predictions.csv')


