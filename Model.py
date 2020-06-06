import os
import pandas as pd
from sklearn import tree
from sklearn import tree
import graphviz
from labels import data_headers, attack_dict
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_validate
import foolbox

#Setup global variables from config
# ATTACK = "teardrop"
#DEBUG = False
class Model:

   def __init__(self, debug, attack, save_model):
      self.DEBUG = debug
      self.ATTACK = attack
      self.SAVE = save_model


   #Create a label for attack types that are being focused on
   def label_attack(self, row):
      if row['attack'] == self.ATTACK:
         return 1
      else:
         return 0

   # --- DATA PREPARATION --- #

   def generate_model(self):

      #Open the dataset
      train_df = pd.read_csv("Datasets/NSL-KDD/KDDTrain+.txt", header=None, names=data_headers)
      test_df = pd.read_csv("Datasets/NSL-KDD/KDDTest+.txt", header=None, names=data_headers)
      train_df = train_df.astype({"protocol_type": str, "service": str, "flag": str})
      test_df = test_df.astype({"protocol_type": str, "service": str, "flag": str})
      train_df['type_label'] = "train"
      test_df['type_label'] = "test"

      #Concat the dataframes
      concat_df = pd.concat([train_df , test_df])
      if (self.DEBUG):
         print(concat_df["service"].unique())

      #TODO remove later --> this is just for decision tree testing for teardrop
      encoded_df = concat_df
      #Onehot encode data
      #encoded_df = pd.get_dummies(concat_df, columns=["protocol_type", "service", "flag"])

      # Resplit the data and drop the labels
      train_df = encoded_df[encoded_df['type_label'] == 'train']
      test_df = encoded_df[encoded_df['type_label'] == 'test']
      train_df = train_df.drop('type_label', axis=1)
      test_df = test_df.drop('type_label', axis=1)

      #Label attacks
      attack = train_df.apply (self.label_attack, axis=1)
      train_df["attack_label"] = attack
      test_attack = test_df.apply (self.label_attack, axis=1)
      test_df["attack_label"] = test_attack

      #TESTING -- DROP PROTOCOL TYPE SERVICE AND FLAG
      train_df = train_df.drop('protocol_type', axis=1)
      train_df = train_df.drop('service', axis=1)
      train_df = train_df.drop('flag', axis=1)
      test_df = test_df.drop('protocol_type', axis=1)
      test_df = test_df.drop('service', axis=1)
      test_df = test_df.drop('flag', axis=1)

      # --- TRAIN --- #
      #Train decision tree for this attack type
      clf = tree.DecisionTreeClassifier(max_depth=5)

      # #Split data and drop label data from x
      x = train_df.drop(["unknown", "attack", "attack_label"], axis=1)

      y = train_df['attack_label']
      clf = clf.fit(x, y)

      x_test = test_df.drop(["unknown", "attack", "attack_label"], axis=1)

      y_test = test_df['attack_label']

      score = clf.score(x_test, y_test)
      y_pred = clf.predict(x_test)
      #Score based of y test and y pred
      conf_matrix = confusion_matrix(y_test, y_pred)
      prec_score = precision_score(y_test, y_pred)
      acc_score = accuracy_score(y_test, y_pred)
      recall = recall_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred)

      if self.DEBUG:
         print(conf_matrix)
         print(prec_score)
         print(acc_score)
         print(recall)
         print(f1)

      if self.SAVE:
         #Generate tree
         dot_data = tree.export_graphviz(clf, out_file=None, feature_names=x_test.columns, class_names=["Benign", "Attack"])  
         graph = graphviz.Source(dot_data)  
         graph.render()
      
      # if (TEST_MODEL):
      #    # --- TEST --- #
      #    x_test = test_df.drop(["unknown", "attack", "attack_label"], axis=1)

      #    y_test = test_df['attack_label']

      #    score = clf.score(x_test, y_test)
      #    y_pred = clf.predict(x_test)
      #    #Score based of y test and y pred
      #    confusion_matrix = confusion_matrix(y_test, y_pred)
      #    precision_score = precision_score(y_test, y_pred)
      #    accuracy_score = accuracy_score(y_test, y_pred)
      #    recall_score = recall_score(y_test, y_pred)
      #    f1_score = f1_score(y_test, y_pred)

      #    print(confusion_matrix)
      #    print(precision_score)
      #    print(accuracy_score)
      #    print(recall_score)
      #    print(f1_score)

      #    #Test - Cross Val Score
      #    # scoring = ['accuracy', 'precision_macro','recall_macro','f1', 'roc_auc']
      #    # scores = cross_validate(clf, x, y, cv=5, scoring=scoring)
      #    #print(scores)


      #    #View features
      #    features = []
      #    for name, importance in zip(x.columns, clf.feature_importances_):
      #       features.append({"name": name, "imp": importance})

      #    for metric in scoring:
      #       total = 0.0
      #       entries = 0.0
      #       for score in scores["test_"+metric]:
      #          total += score
      #          entries += 1
            
      #       metric_sum = total / entries
      #       print("test_"+metric)
      #       print(metric_sum)

      #    # -- ATTACK -- #
      #    print("Generating Attack")
      #    # GENERATES ATTACKS USING GENETIC ALGORITHMS


      #    # --- OUTPUT --- #
      #    # print("Attack: " + str(ATTACK))
      #    #Precision, recall, f-measure and confusion matrix
      #    #print("Score: " + str(score))
      #    for feature in features:
      #       if (feature["imp"] != 0):
      #          print(str(feature["name"]) + " " + str(feature["imp"]))

      #    # dot_data = tree.export_graphviz(clf, out_file=None, feature_names=x_test.columns, class_names=["Benign", "Attack"])  
      #    # graph = graphviz.Source(dot_data)  
      #    # graph.render()
      return clf

#Run the model if called as a script
# if __name__ == "__main__":
#    #Default to debug and teardrop attack
#     model(True)