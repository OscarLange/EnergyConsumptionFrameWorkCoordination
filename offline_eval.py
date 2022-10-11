from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os 
import pydot
from sklearn.tree import export_graphviz
import joblib
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

df = pd.read_csv('./stats/initial_stats.csv')
df = df.sort_values(by=['Freq', 'Util', 'WorkMode'])

def enumerate_work(work):
    if(work == "noop_test.csv"):
        return 0
    if(work == "add_test.csv"):
        return 1
    if(work == "sub_test.csv"):
        return 2
    if(work == "mul_test.csv"):
        return 3
    if(work == "div_test.csv"):
        return 4
    if(work == "addf_test.csv"):
        return 5
    if(work == "subf_test.csv"):
        return 6
    if(work == "mulf_test.csv"):
        return 7
    if(work == "divf_test.csv"):
        return 8
    if(work == "linkedlist_test.csv"):
        return 9

def clean_up(df):
    df = df.drop("Power", axis = 1)
    df = df.drop("Current", axis = 1)
    df = df.drop("Shunt", axis = 1)
    df = df.drop("Voltage", axis = 1)
    df["WorkMode"] = df["WorkMode"].apply(lambda x: enumerate_work(x))
    print(df)
    return df

def random_forrest():
    training_frame = df

    labels = np.array(training_frame["Power"])
    training_frame = clean_up(training_frame)
    print(training_frame)

    feature_list = list(training_frame.columns)
    features = np.array(training_frame)

    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(features, labels)

    predictions = rf.predict(features)

    errors = mean_squared_error(labels, predictions)
    print(errors)

    tree = rf.estimators_[50]
    export_graphviz(tree, out_file="tree.dot", feature_names=feature_list, rounded=True, precision= 1)
    (graph, ) = pydot.graph_from_dot_file(dir_path + 'tree.dot')

    graph.write_png(dir_path + "pictures/tree.png")
    joblib.dump(rf, "./random_forest.joblib")

def numpy_polyfit():
    global df
    print(df.groupby(['Freq','Util'])['Power'].mean())
    df["indicator"] = df["Freq"] + df["Util"]

    df_160 = df[df["Freq"] == 160]
    df_240 = df[df["Freq"] == 240]

    X1 = np.sort(df_160["indicator"].unique())
    X2 = np.sort(df_240["indicator"].unique())
    y1 =  df_160.groupby('indicator')['Power'].mean().to_numpy()
    y2 =  df_240.groupby('indicator')['Power'].mean().to_numpy()

    coefs1, residual1, _, _, _ = np.polyfit(X1, y1, 2, full=True)
    print(residual1/len(X1))
    print(coefs1)

    coefs2, residual2, _, _, _ = np.polyfit(X2, y2, 3, full=True)
    print(residual2/len(X1))
    print(coefs2)

    poly1d1 = np.poly1d(coefs1)
    poly1d2 = np.poly1d(coefs2)

    #prediction = np.polyval(poly1d, X_test)


    plt.scatter(X1, y1, color = 'blue', label="Avg 160")
    plt.plot(X1, poly1d1(X1), color = 'blue', label="Polynomial 160")

    plt.scatter(X2, y2, color = 'red', label="Avg 240")
    plt.plot(X2, poly1d2(X2), color = 'red', label="Polynomial 240")

    plt.xlabel('Work Percentage x Freq')
    plt.ylabel('Power Consumption')
    plt.legend()

    plt.show()

random_forrest()
numpy_polyfit()