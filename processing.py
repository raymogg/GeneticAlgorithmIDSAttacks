import os
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree
import graphviz
from labels import data_headers, attack_dict
from sklearn.preprocessing import OneHotEncoder

#Setup global variables from config
ATTACK = "teardrop"

#Create a label for attack types that are being focused on
def label_attack(row):
   if row['attack'] == ATTACK:
      return 1
   else:
      return 0

# --- DATA PREPARATION --- #

#Open the dataset
train_df = pd.read_csv("Datasets/NSL-KDD/KDDTrain+.txt", header=None, names=data_headers)
test_df = pd.read_csv("Datasets/NSL-KDD/KDDTest+.txt", header=None, names=data_headers)
train_df = train_df.astype({"protocol_type": str, "service": str, "flag": str})
test_df = test_df.astype({"protocol_type": str, "service": str, "flag": str})
train_df['type_label'] = "train"
test_df['type_label'] = "test"

#Concat the dataframes
concat_df = pd.concat([train_df , test_df])

#Onehot encode data
encoded_df = pd.get_dummies(concat_df, columns=["protocol_type", "service", "flag"])

# Resplit the data and drop the labels
train_df = encoded_df[encoded_df['type_label'] == 'train']
test_df = encoded_df[encoded_df['type_label'] == 'test']
train_df = train_df.drop('type_label', axis=1)
test_df = test_df.drop('type_label', axis=1)

#Label attacks
attack = train_df.apply (label_attack, axis=1)
train_df["attack_label"] = attack
test_attack = test_df.apply (label_attack, axis=1)
test_df["attack_label"] = test_attack

# --- TRAIN --- #
#Train decision tree for this attack type
clf = tree.DecisionTreeClassifier()

# #Split data and drop label data from x
x = train_df.drop(["unknown", "attack", "attack_label"], axis=1)

y = train_df['attack_label']
clf = clf.fit(x, y)

# --- TEST --- #
x_test = test_df.drop(["unknown", "attack", "attack_label"], axis=1)

y_test = test_df['attack_label']
score = clf.score(x_test, y_test)

#View features
features = []
for name, importance in zip(x.columns, clf.feature_importances_):
   features.append({"name": name, "imp": importance})

# --- OUTPUT --- #
print("Attack: " + str(ATTACK))
print("Score: " + str(score))
for feature in features:
   if (feature["imp"] != 0):
      print(str(feature["name"]) + " " + str(feature["imp"]))

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=x_test.columns, class_names=["Benign", "Attack"])  
graph = graphviz.Source(dot_data)  
graph.render()