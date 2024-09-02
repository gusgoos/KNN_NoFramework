import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Función para calcular la distancia Euclidiana basada en la fórmula sqrt(sum((x1 - x2)^2))
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Función para predecir las etiquetas de clase usando el algoritmo de K-vecinos más cercanos
def knn_predict(X_train, y_train, X_test, k=3):
    # Lista para almacenar las etiquetas de clase predichas
    predictions = []
    for x in X_test:
        # Calcular distancias entre x y todos los ejemplos de entrenamiento
        distances = [euclidean_distance(x, x_train) for x_train in X_train]
        # Obtener los índices de los k-vecinos más cercanos
        k_indices = np.argsort(distances)[:k]
        # Obtener las etiquetas de los k-vecinos más cercanos
        k_nearest_labels = [y_train[i] for i in k_indices]
        # Encontrar la etiqueta más común entre los vecinos
        label_count = {}
        for label in k_nearest_labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        # Devolver la etiqueta con mayor cantidad
        most_common_label = max(label_count, key=label_count.get)
        predictions.append(most_common_label)
    return np.array(predictions)

# Cargar el conjunto de datos de Wine
wine = load_wine()
X = wine.data
y = wine.target

# Dividir los datos en conjuntos de entrenamiento y prueba (e.g., 75% entrenamiento, 25% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # Semilla aleatoria random_state=42 para reproducibilidad, se puede cambiar a otro valor

# Estandarizar las características
# Esto es importante para algoritmos basados en distancias como KNN para que todas las características tengan la misma escala
# Se ajusta el escalador solo en los datos de entrenamiento y se transforman los datos de prueba con el mismo escalador
# Esto evita el sesgo de información de los datos de prueba en el entrenamiento del modelo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Predecir con KNN
k = 5  # Se puede ajustar k para obtener mejores resultados
y_pred = knn_predict(X_train, y_train, X_test, k=k)

# Evaluar el modelo en datos estandarizados
print("Evaluación del modelo con datos estandarizados:")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
# Precisión, Recall, y F1 Score
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Precisión: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")