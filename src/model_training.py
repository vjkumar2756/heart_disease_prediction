from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def train_models(X_train, y_train):
    models = {}

    # Logistic Regression
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train, y_train)
    models["LogisticRegression"] = model_lr

    # SVM
    model_svm = SVC(probability=True)
    model_svm.fit(X_train, y_train)
    models["SVM"] = model_svm

    # Decision Tree
    model_dt = DecisionTreeClassifier()
    model_dt.fit(X_train, y_train)
    models["DecisionTree"] = model_dt

    # Naive Bayes
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    models["NaiveBayes"] = model_nb

    # KNN (fixed: classifier, not regressor)
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train, y_train)
    models["KNN"] = model_knn

    return models