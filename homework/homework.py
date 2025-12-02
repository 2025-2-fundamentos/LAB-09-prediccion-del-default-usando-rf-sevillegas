# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


#Load data
def load_data(data):
    import pandas as pd

    data = pd.read_csv(f"./files/input/{data}.csv.zip", compression="zip") 

    return data

train_data = load_data("train_data")
test_data = load_data("test_data")


#Clean data
def clean_data(data):
    data = data.copy()
    data.rename(columns = {'default payment next month':'default'}, inplace = True)
    data.dropna(inplace=True)

    data["EDUCATION"] = data["EDUCATION"].apply(lambda x: "others" if x not in [1,2,3, 4] else str(x))
    data.drop(columns=["ID"], inplace=True)

    return data

cleaned_train_data = clean_data(train_data)
cleaned_test_data = clean_data(test_data)

x_train = cleaned_train_data.drop(columns=["default"])
y_train = cleaned_train_data["default"] 
x_test = cleaned_test_data.drop(columns=["default"])
y_test = cleaned_test_data["default"]



#Pipeline
def make_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer
    cat = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    
    to_str = FunctionTransformer(
        func=lambda X: X.astype(str),
        validate=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("to_str", to_str, cat),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ],
        remainder="passthrough",
    )


    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    return pipeline


#Optimización de hiperparametros con validación cruzada
def opt_params(pipeline):
    from sklearn.model_selection import GridSearchCV


    params = {
        'classifier__n_estimators': [200],
        'classifier__max_depth': [10, None],
        'classifier__min_samples_split': [2, 5],
    }

    grid_search = GridSearchCV(
        pipeline, 
        params, 
        cv=10, 
        scoring='balanced_accuracy', 
        n_jobs=-1,
        verbose=1
        )
    grid_search.fit(x_train, y_train)

    return grid_search


pipeline = make_pipeline()
estimator = opt_params(pipeline)

#Guardar modelo

import os
import gzip
import pickle

if not os.path.exists("./files/models/"):
    os.makedirs("./files/models/")


model_name = "./files/models/model.pkl.gz"
with gzip.open(model_name, 'wb') as f:
    pickle.dump(estimator, f)


#Calculo de métricas
def calc_metrics(estimator):

    x_train = cleaned_train_data.drop(columns=["default"])
    y_train = cleaned_train_data["default"]
    x_test = cleaned_test_data.drop(columns=["default"])
    y_test = cleaned_test_data["default"]

    from sklearn.metrics import (
        precision_score,
        balanced_accuracy_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    import json
    import os

    out_dir = "./files/output/"
    out_file = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # create the json file if it doesn't exist
    if not os.path.exists(out_file):
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("")


    

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:

        # ---- TRAIN METRICS ----
        y_pred_train = estimator.predict(x_train)
        metrics_train = {
            "type": "metrics",
            "dataset": "train",
            "precision": float(precision_score(y_train, y_pred_train)),
            "balanced_accuracy": float(balanced_accuracy_score(y_train, y_pred_train)),
            "recall": float(recall_score(y_train, y_pred_train)),
            "f1_score": float(f1_score(y_train, y_pred_train)),
        }
        file.write(json.dumps(metrics_train) + "\n")

        cm = confusion_matrix(y_train, y_pred_train)
        cm_train = {
            "type": "cm_matrix",
            "dataset": "train",
            "true_0": {
                "predicted_0": int(cm[0][0]),
                "predicted_1": int(cm[0][1]),
            },
            "true_1": {
                "predicted_0": int(cm[1][0]),
                "predicted_1": int(cm[1][1]),
            },
        }
        file.write(json.dumps(cm_train) + "\n")

        y_pred_test = estimator.predict(x_test)
        metrics_test = {
            "type": "metrics",
            "dataset": "test",
            "precision": float(precision_score(y_test, y_pred_test)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_test)),
            "recall": float(recall_score(y_test, y_pred_test)),
            "f1_score": float(f1_score(y_test, y_pred_test)),
        }
        file.write(json.dumps(metrics_test) + "\n")

        cm = confusion_matrix(y_test, y_pred_test)
        cm_test = {
            "type": "cm_matrix",
            "dataset": "test",
            "true_0": {
                "predicted_0": int(cm[0][0]),
                "predicted_1": int(cm[0][1]),
            },
            "true_1": {
                "predicted_0": int(cm[1][0]),
                "predicted_1": int(cm[1][1]),
            },
        }
        file.write(json.dumps(cm_test) + "\n")

calc_metrics(estimator)