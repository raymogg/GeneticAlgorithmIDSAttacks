import os
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import cross_val_score


#Define some constants for ease of use
NSL_DIR = "Datasets/NSL-KDD/MachineLearningCVE"
NSL_FLAGS = [' Destination Port', ' Flow Duration', ' Total Fwd Packets',
       ' Total Backward Packets', 'Total Length of Fwd Packets',
       ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
       ' Fwd Packet Length Std', 'Bwd Packet Length Max',
       ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
       ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
       ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
       'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
       ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
       ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
       ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
       ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
       ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
       ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
       ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
       ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
       ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
       ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
       ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
       ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
       'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
       ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
       ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
       ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']

dtypes = {' Destination Port': np.int64, ' Flow Duration': np.int64, ' Total Fwd Packets':np.int64,
       ' Total Backward Packets': np.int64, 'Total Length of Fwd Packets': np.int64,
       ' Total Length of Bwd Packets':np.int64, ' Fwd Packet Length Max': np.int64,
       ' Fwd Packet Length Min': np.int64, ' Fwd Packet Length Mean': np.float,
       ' Fwd Packet Length Std': np.float, 'Bwd Packet Length Max': np.int64,
       ' Bwd Packet Length Min': np.int64, ' Bwd Packet Length Mean': np.float,
       ' Bwd Packet Length Std': np.float, 'Flow Bytes/s': np.object, ' Flow Packets/s': np.object,
       ' Flow IAT Mean': np.float, ' Flow IAT Std': np.float, ' Flow IAT Max': np.int64, ' Flow IAT Min': np.int64,
       'Fwd IAT Total': np.int64, ' Fwd IAT Mean': np.float, ' Fwd IAT Std': np.float, ' Fwd IAT Max': np.int64,
       ' Fwd IAT Min': np.int64, 'Bwd IAT Total': np.int64, ' Bwd IAT Mean': np.float, ' Bwd IAT Std': np.float,
       ' Bwd IAT Max': np.int64, ' Bwd IAT Min': np.int64, 'Fwd PSH Flags': np.int64, ' Bwd PSH Flags': np.int64,
       ' Fwd URG Flags': np.int64, ' Bwd URG Flags': np.int64, ' Fwd Header Length': np.int64,
       ' Bwd Header Length': np.int64, 'Fwd Packets/s': np.float, ' Bwd Packets/s': np.float,
       ' Min Packet Length': np.int64, ' Max Packet Length': np.int64, ' Packet Length Mean': np.float,
       ' Packet Length Std': np.float, ' Packet Length Variance': np.float, 'FIN Flag Count': np.int64,
       ' SYN Flag Count': np.int64, ' RST Flag Count': np.int64, ' PSH Flag Count': np.int64,
       ' ACK Flag Count': np.int64, ' URG Flag Count': np.int64, ' CWE Flag Count': np.int64,
       ' ECE Flag Count': np.int64, ' Down/Up Ratio': np.int64, ' Average Packet Size': np.float,
       ' Avg Fwd Segment Size': np.float, ' Avg Bwd Segment Size': np.float,
       ' Fwd Header Length.1': np.int64, 'Fwd Avg Bytes/Bulk': np.float, ' Fwd Avg Packets/Bulk': np.float,
       ' Fwd Avg Bulk Rate': np.float, ' Bwd Avg Bytes/Bulk': np.float, ' Bwd Avg Packets/Bulk': np.float,
       'Bwd Avg Bulk Rate': np.float, 'Subflow Fwd Packets': np.int64, ' Subflow Fwd Bytes': np.int64,
       ' Subflow Bwd Packets': np.int64, ' Subflow Bwd Bytes': np.int64, 'Init_Win_bytes_forward': np.int64,
       ' Init_Win_bytes_backward': np.int64, ' act_data_pkt_fwd': np.int64,
       ' min_seg_size_forward': np.int64, 'Active Mean': np.float, ' Active Std': np.float, ' Active Max': np.float,
       ' Active Min': np.float, 'Idle Mean': np.float, ' Idle Std': np.float, ' Idle Max': np.float, ' Idle Min': np.float}

def label_dos_attack(row):
   if row[' Label'] == "DoS slowloris" or row[' Label'] == "DoS Slowhttptest" or row[' Label'] == "DoS Hulk" or row[' Label'] == "DoS GoldenEye":
      return 1
   else:
       return 0

def label_attack(row):
   if row[' Label'] != "BENIGN":
      return 1
   else:
      return 0

def _assert_all_finite(X):
   """Like assert_all_finite, but only for ndarray."""
   X = np.asanyarray(X)
   # First try an O(n) time, O(1) space solution for the common case that
   # everything is finite; fall back to O(n) space np.isfinite to prevent
   # false positives from overflow in sum method.
   if np.isfinite(X.sum()):
      print("A")
   if (np.isfinite(X).all()):
      print("B")
   if(not X.dtype.char in np.typecodes['AllFloat']):
      print("C")

#Open all datasets in the NSL data
nsl = os.listdir(NSL_DIR)

dfs = []

for file_name in nsl:
    #Load in each csv
    df = pd.read_csv(NSL_DIR + "/" + file_name, index_col=None, header=0, dtype=dtypes)
    print(NSL_DIR + "/" + file_name)
    dfs.append(df)
   
#dfs.append(pd.read_csv("Datasets/NSL-KDD/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv", dtype=dtypes))

#Construct a frame of all data
nsl_frame = pd.concat(dfs, axis=0, ignore_index=True)
#print(nsl_frame.head)

#Get all entries that aren't benign
#attacks = nsl_frame.loc[nsl_frame[' Label'] != "BENIGN"]
#print(attacks.head)
#List all attack types
#print(attacks[' Label'].unique())

#Train DOS decision tree
#Add a new binary column called dos_label --> 1 if it is a DoS attack, 0 else (doesn't pick up other attacks)
attack = nsl_frame.apply (label_attack, axis=1)
nsl_frame["attack_label"] = attack
print(nsl_frame.columns)
nsl_frame = nsl_frame.dropna()

# mask1 = nsl_frame['Flow Bytes/s'].apply(lambda x: isinstance(x, (str)))
# print(mask1.head)
# mask2 = nsl_frame[' Flow Packets/s'].apply(lambda x: isinstance(x, (str)))
# mask3 = nsl_frame[' Flow IAT Mean'].apply(lambda x: isinstance(x, (str)))
# nsl_frame = nsl_frame[mask1]
# nsl_frame = nsl_frame[mask2]
# nsl_frame = nsl_frame[mask3]
# print(nsl_frame.head)

#Train a decision tree on our data
nsl_frame = nsl_frame.dropna()
nsl_frame = nsl_frame.reset_index()
clf = tree.DecisionTreeClassifier()
x = nsl_frame[NSL_FLAGS]
y = nsl_frame['attack_label']
# _assert_all_finite(x)
# _assert_all_finite(y)
X = x.as_matrix().astype(np.float)
Y = y.as_matrix().astype(np.float)
X[~np.isfinite(X)] = 0
Y[~np.isfinite(Y)] = 0
clf = clf.fit(X, Y)
scoring = cross_val_score(clf, X, Y, cv=10)
print(scoring)
#tree.plot_tree(clf) 


