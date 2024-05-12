import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Wczytywanie danych
df = pd.read_csv('./preprocessing/youtubeChannelsPreprocessed.csv')
# balans klas
class_counts = df['category'].value_counts()
print(class_counts)
# 'rank', 'youtuber', 'subscribers', 'video_views', 'category', 'title',
#        'uploads', 'country', 'abbreviation', 'channel_type',
#        'video_views_rank', 'country_rank', 'video_views_for_the_last_30_days',
#        'lowest_monthly_earnings', 'highest_monthly_earnings',
#        'lowest_yearly_earnings', 'highest_yearly_earnings',
#        'subscribers_for_last_30_days', 'gross_tertiary_education_enrollment',
#        'population', 'unemployment_rate', 'urban_population', 'created_at'
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])
transformed_etiquettes = dict(zip(le.classes_, le.transform(le.classes_)))
print(transformed_etiquettes)
df['channel_type'] = le.fit_transform(df['channel_type'])

# Wybór cech i klasy docelowej
X = df[['uploads', 'video_views', 'video_views_for_the_last_30_days', 'subscribers', 'subscribers_for_last_30_days',
        'channel_type']]
y = df['category']
X_scaled = StandardScaler().fit_transform(X)

# Podział na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Klasyfikacja za pomocą KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Przewidywanie dla zestawu testowego
y_pred = knn.predict(X_test)

# Ewaluacja modelu
accuracy = accuracy_score(y_test, y_pred)
accuracy_knn3 = accuracy
print(f"Accuracy for n=3: {accuracy * 100:.2f}%")
confusion_matrix_knn3 = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))  # Zwiększ rozmiar wykresu
sns.heatmap(confusion_matrix_knn3, annot=True, fmt='d', xticklabels=transformed_etiquettes.keys(),
            yticklabels=transformed_etiquettes.keys())
plt.xticks(rotation=90)  # Obróć etykiety osi x o 90 stopni
plt.yticks(rotation=0)  # Obróć etykiety osi y o 0 stopni (opcjonalne)
plt.tight_layout()  # Dopasuj wykres do obszaru wykresu
plt.savefig('./plots_and_figures/knn3_confusion_matrix.png')
plt.clf()

# Dodanie wykresu learning curve
train_sizes, train_scores, test_scores = learning_curve(knn, X_train, y_train, cv=5, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Wielkość zbioru treningowego')
plt.ylabel('Dokładność')
plt.title('Krzywa uczenia dla KNN (n=3)')
plt.legend(loc='best')
plt.grid()
plt.savefig('./plots_and_figures/knn3_learning_curve.png')
plt.clf()
# Powtarzamy dla n=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_knn5 = accuracy
print(f"Accuracy for n=5: {accuracy * 100:.2f}%")
confusion_matrix_knn5 = confusion_matrix(y_test, y_pred)
# make plot with this matrix and save it to knn_confusion_matrix.png
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix_knn5, annot=True, fmt='d', xticklabels=transformed_etiquettes.keys(),
            yticklabels=transformed_etiquettes.keys())
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./plots_and_figures/knn5_confusion_matrix.png')
plt.clf()
# Dodanie wykresu learning curve
train_sizes, train_scores, test_scores = learning_curve(knn, X_train, y_train, cv=5, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Wielkość zbioru treningowego')
plt.ylabel('Dokładność')
plt.title('Krzywa uczenia dla KNN (n=5)')
plt.legend(loc='best')
plt.grid()
plt.savefig('./plots_and_figures/knn5_learning_curve.png')
plt.clf()
# Powtarzamy dla n=7
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_knn7 = accuracy
print(f"Accuracy for n=7: {accuracy * 100:.2f}%")

confusion_matrix_knn7 = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix_knn7, annot=True, fmt='d', xticklabels=transformed_etiquettes.keys(),
            yticklabels=transformed_etiquettes.keys())
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./plots_and_figures/knn7_confusion_matrix.png')
plt.clf()
# Dodanie wykresu learning curve
train_sizes, train_scores, test_scores = learning_curve(knn, X_train, y_train, cv=5, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Wielkość zbioru treningowego')
plt.ylabel('Dokładność')
plt.title('Krzywa uczenia dla KNN (n=7)')
plt.legend(loc='best')
plt.grid()
plt.savefig('./plots_and_figures/knn7_learning_curve.png')
plt.clf()
# Powtarzamy dla n=9
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_knn9 = accuracy
print(f"Accuracy for n=9: {accuracy * 100:.2f}%")

confusion_matrix_knn9 = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix_knn9, annot=True, fmt='d', xticklabels=transformed_etiquettes.keys(),
            yticklabels=transformed_etiquettes.keys())
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./plots_and_figures/knn9_confusion_matrix.png')
plt.clf()
# Dodanie wykresu learning curve
train_sizes, train_scores, test_scores = learning_curve(knn, X_train, y_train, cv=5, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Wielkość zbioru treningowego')
plt.ylabel('Dokładność')
plt.title('Krzywa uczenia dla KNN (n=3)')
plt.legend(loc='best')
plt.grid()
plt.savefig('./plots_and_figures/knn9_learning_curve.png')
plt.clf()
# Przykład użycia Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
confusion_matrix_random_forest = confusion_matrix(y_test, y_pred_rf)
# make plot with this matrix and save it to random_tree_confusion_matrix.png
# use transformed_etiquettes to set labels
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix_random_forest, annot=True, fmt='d', xticklabels=transformed_etiquettes.keys(),
            yticklabels=transformed_etiquettes.keys())
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./plots_and_figures/random_forest_confusion_matrix.png')
plt.clf()
# Dodanie wykresu learning curve
train_sizes, train_scores, test_scores = learning_curve(knn, X_train, y_train, cv=5, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Wielkość zbioru treningowego')
plt.ylabel('Dokładność')
plt.title('Krzywa uczenia dla drzewa decyzyjnego (Random Forest)')
plt.legend(loc='best')
plt.grid()
plt.savefig('./plots_and_figures/random_forest_learning_curve.png')
plt.clf()

# drzewo decyzyjne
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")

confusion_matrix_decision_tree = confusion_matrix(y_test, y_pred_dt)
# make plot with this matrix and save it to decision_tree_confusion_matrix.png
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix_decision_tree, annot=True, fmt='d', xticklabels=transformed_etiquettes.keys(),
            yticklabels=transformed_etiquettes.keys())
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./plots_and_figures/decision_tree_confusion_matrix.png')
plt.clf()
# Dodanie wykresu learning curve
train_sizes, train_scores, test_scores = learning_curve(knn, X_train, y_train, cv=5, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Wielkość zbioru treningowego')
plt.ylabel('Dokładność')
plt.title('Krzywa uczenia dla drzewa decyzyjnego (Decision Tree)')
plt.legend(loc='best')
plt.grid()
plt.savefig('./plots_and_figures/decision_tree_learning_curve.png')
plt.clf()

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb * 100:.2f}%")

confusion_matrix_naive_bayes = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix_naive_bayes, annot=True, fmt='d', xticklabels=transformed_etiquettes.keys(),
            yticklabels=transformed_etiquettes.keys())
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./plots_and_figures/naive_bayes_confusion_matrix.png')
plt.clf()
# Dodanie wykresu learning curve
train_sizes, train_scores, test_scores = learning_curve(knn, X_train, y_train, cv=5, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Wielkość zbioru treningowego')
plt.ylabel('Dokładność')
plt.title('Krzywa uczenia dla Naive Bayes')
plt.legend(loc='best')
plt.grid()
plt.savefig('./plots_and_figures/naive_bayes_learning_curve.png')
plt.clf()
# Budowa modelu sieci neuronowej
# wczytanie danych od nowa
df = pd.read_csv('./preprocessing/youtubeChannelsPreprocessed.csv')
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])
# channel_type będzie daną wejściową, więc też musimy ją zakodować, ale OneHotEncoderem
ohe = OneHotEncoder()
df['channel_type'] = ohe.fit_transform(df[['channel_type']]).toarray()

X = df[['uploads', 'video_views', 'video_views_for_the_last_30_days', 'subscribers', 'subscribers_for_last_30_days',
        'channel_type']]
y = df['category']
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation="softmax"),
    Dense(len(le.classes_), activation='softmax')  # Liczba wyjść równa liczbie klas
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trening modelu
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

# Przewidywanie dla zestawu testowego
y_pred = model.predict(X_test)

# Ewaluacja modelu
accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))
print(f"Neural Network Accuracy: {accuracy * 100:.2f}%")

confusion_matrix_nn = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix_nn, annot=True, fmt='d', xticklabels=transformed_etiquettes.keys(),
            yticklabels=transformed_etiquettes.keys())
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./plots_and_figures/neural_network_confusion_matrix.png')

# Dodanie wykresu learning curve
train_sizes, train_scores, test_scores = learning_curve(knn, X_train, y_train, cv=5, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Wielkość zbioru treningowego')
plt.ylabel('Dokładność')
plt.title('Krzywa uczenia dla sieci neuronowej')
plt.legend(loc='best')
plt.grid()
plt.savefig('./plots_and_figures/neural_network_learning_curve.png')
plt.clf()
