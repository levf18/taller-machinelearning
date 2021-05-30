import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score


url = 'diabetes.csv'
data = pd.read_csv(url)


data['Age'].mean()
data['Age'].replace(np.nan, 33, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Age = pd.cut(data.Age, rangos, labels=nombres)

rangos_preg = [0, 8, 15, 18, 25, 40, 60, 100]
nombres_preg = ['1', '2', '3', '4', '5', '6', '7']
data.Pregnancies = pd.cut(data.Pregnancies, rangos_preg, labels=nombres_preg)

rangos_skin = [0, 8, 15, 18, 25, 40, 60, 100]
nombres_skin = ['1', '2', '3', '4', '5', '6', '7']
data.SkinThickness = pd.cut(data.SkinThickness, rangos_skin, labels=nombres_skin)
data.dropna(axis=0, how='any', inplace=True)

data_train = data[:550]
data_test = data[550:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome)  # 0 No # 1 Si

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)  # 0 No # 1 Si


def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix,);
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print(classification_report(y_test, pred_y))


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

# Regresión Logística

# Selecciono un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter=7600)
# Entreno un modelo
logreg.fit(x_train, y_train)
# Accuracy de Entremaniento
print('*'* 50)
print('Regresión Logística')
print(f'Accuracy de Entrenamiento: {logreg.score(x_train, y_train)}')
# Accuracy de Test
print(f'Accuracy de Test: {logreg.score(x_test, y_test)}')
y_pred = logreg.predict(x_test_out)
print(f'Accuracy de Validación: {accuracy_score(y_pred, y_test_out)}')
print('Matriz de confusión')
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
print('*'* 50)

probs = logreg.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test_out, probs)
plot_roc_curve(fpr, tpr)

print(precision_score(y_test_out, y_pred))
# Regresión Logística con validación cruzada

logreg = LogisticRegression(solver='lbfgs', max_iter=7600)
kfold = KFold(n_splits=10)
cvscores = []
for train, test in kfold.split(x_train, y_train):
    logreg.fit(x_train[train], y_train[train])
    scores = logreg.score(x_train[test], y_train[test])
    cvscores.append(scores)
print(np.mean(cvscores))

print('Regresión Logística validación Cruzada')
print(f'Accuracy de Entrenamiento: {logreg.score(x_train, y_train)}')
# Accuracy de Test
print(f'Accuracy de Test: {logreg.score(x_test, y_test)}')
print('*'* 50)

# Maquinas de Soporte Vectorial

svc = SVC(gamma='auto')
svc.fit(x_train, y_train)
print('Maquina de soporte Vectorial')
print(f'Accuracy de Entrenamiento: {svc.score(x_train, y_train)}')
print(f'Accuracy de Test: {svc.score(x_test, y_test)}')
print('*'* 50)

# Vecinos mas cercanos clasificador

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

print('Clasificador Vecinos mas cercanos')
print(f'Accuracy de Entrenamiento: {knn.score(x_train, y_train)}')
print(f'Accuracy de Test: {knn.score(x_test, y_test)}')

