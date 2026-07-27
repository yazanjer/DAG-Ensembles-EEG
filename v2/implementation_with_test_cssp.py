# ============================================================
# EEG CSP + CSSP ENSEMBLE WITH SIMULATED ANNEALING
# BCIC IV Dataset 1
# ============================================================

import os
import copy
import math
import pickle
import random
import itertools
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import networkx as nx

from collections import Counter
from pathlib import Path
import mne
from mne.decoding import CSP
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# GLOBAL CONFIG
# ------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

mne.set_log_level("WARNING")

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_RESULTS_DIR = PROJECT_ROOT / "results_sa_same_svm_kernal"
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILES_TO_PROCESS = [
    PROJECT_ROOT / 'dataset' / 'BCICIV_calib_ds1a.mat',
    PROJECT_ROOT / 'dataset' / 'BCICIV_calib_ds1b.mat',
    PROJECT_ROOT / 'dataset' / 'BCICIV_calib_ds1f.mat',
    PROJECT_ROOT / 'dataset' / 'BCICIV_calib_ds1g.mat',
]

TIME_WINDOWS_ANALYSIS = [(0.5,3.5),(1.0,4.0),(1.0,3.0)]
FREQUENCY_BANDS = [(8,30),(8,13),(14,30)]
CSP_COMPONENTS = [4,6,7,8]

# ------------------------------------------------------------
# DATA LOADER (FIXED STRUCT INDEXING)
# ------------------------------------------------------------

def load_bciciv_data(path):

    data = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)

    cnt = data['cnt'].T
    fs = int(data['nfo'].fs)

    pos = data['mrk'].pos
    y = data['mrk'].y

    duration = int(4*fs)

    epochs, labels = [], []

    for p,l in zip(pos,y):
        if p+duration <= cnt.shape[1]:
            epochs.append(cnt[:,p:p+duration])
            labels.append(l)

    X = np.stack(epochs)
    y = np.array(labels)

    y = np.where(y==np.unique(y)[0],0,1)

    return X,y,fs

# ------------------------------------------------------------
# SAFE CROP (REMOVE MNE WARNING)
# ------------------------------------------------------------

def safe_crop(ep,tmin,tmax):
    return ep.crop(tmin=tmin,
                   tmax=min(tmax,ep.tmax),
                   verbose=False)

# ------------------------------------------------------------
# CSSP (Temporal Augmentation)
# ------------------------------------------------------------

def cssp_transform(X, delays=2):

    if delays==0:
        return X

    trials,chans,samples = X.shape
    out=[]

    for d in range(delays+1):
        out.append(X[:,:,d:samples-delays+d])

    return np.concatenate(out,axis=1)

# ------------------------------------------------------------
# FEATURE EXTRACTION (MNE BASED)
# ------------------------------------------------------------

def extract_features(X_train,X_val,X_test,y_train,fs):

    ch_names=[f"Ch{i}" for i in range(X_train.shape[1])]
    info=mne.create_info(ch_names,fs,'eeg')

    ep_tr=mne.EpochsArray(X_train,info,verbose=False)
    ep_va=mne.EpochsArray(X_val,info,verbose=False)
    ep_te=mne.EpochsArray(X_test,info,verbose=False)

    bank={}

    for band in FREQUENCY_BANDS:
        for tw in TIME_WINDOWS_ANALYSIS:
            for n_comp in CSP_COMPONENTS:

                tr=safe_crop(ep_tr.copy().filter(*band),*tw)
                va=safe_crop(ep_va.copy().filter(*band),*tw)
                te=safe_crop(ep_te.copy().filter(*band),*tw)

                Xtr,Xva,Xte=tr.get_data(),va.get_data(),te.get_data()

                # ---- CSP
                csp=CSP(n_components=n_comp,log=True,norm_trace=False)

                Ftr=csp.fit_transform(Xtr,y_train)
                Fva=csp.transform(Xva)
                Fte=csp.transform(Xte)

                bank[("CSP",band,tw,n_comp)]=(Ftr,Fva,Fte)

                # ---- CSSP
                Xtr2=cssp_transform(Xtr)
                Xva2=cssp_transform(Xva)
                Xte2=cssp_transform(Xte)

                csp2=CSP(n_components=n_comp,log=True,norm_trace=False)

                bank[("CSSP",band,tw,n_comp)]=(
                    csp2.fit_transform(Xtr2,y_train),
                    csp2.transform(Xva2),
                    csp2.transform(Xte2)
                )

    print("Feature configs:",len(bank))
    return bank

# ------------------------------------------------------------
# BASE CLASSIFIER
# ------------------------------------------------------------

class BaseClassifier:

    def __init__(self,key):
        self.key=key
        self.model=SVC(kernel="rbf",
                       C=1,
                       gamma="scale",
                       probability=True)


    def fit(self,feats,y):
        self.model.fit(feats[self.key][0],y)

    def predict_proba(self,feats,split=1):
        return self.model.predict_proba(feats[self.key][split])

# ------------------------------------------------------------
# ENSEMBLE NODE
# ------------------------------------------------------------

class EnsembleNode:

    def __init__(self,parents,mode="SV"):
        self.parents=parents
        self.mode=mode

    def predict_proba(self,feats,split):

        probs=[p.predict_proba(feats,split) for p in self.parents]
        stack=np.array(probs)

        if self.mode=="SV":
            return stack.mean(axis=0)

        if self.mode=="HV":
            return np.max(stack,axis=0)

        if self.mode=="MIN":
            return np.min(stack,axis=0)

    def predict(self,feats,split):
        return np.argmax(self.predict_proba(feats,split),axis=1)

# ------------------------------------------------------------
# DAG
# ------------------------------------------------------------

class EnsembleDAG:
    def __init__(self,root):
        self.root=root

    def accuracy(self,feats,y):
        pred=self.root.predict(feats,1)
        return accuracy_score(y,pred)

# ------------------------------------------------------------
# SIMULATED ANNEALING
# ------------------------------------------------------------

class SAOptimizer:

    def __init__(self,pool,feats,y_val,name):
        self.pool=pool
        self.feats=feats
        self.y=y_val
        self.name=name

        self.history={"acc":[],"temp":[],"best":[]}

        self.current=self.random_dag()
        self.best=self.current

    def random_dag(self):
        parents=random.sample(self.pool,4)
        node=EnsembleNode(parents,"SV")
        return EnsembleDAG(node)

    def perturb(self,dag):
        new=copy.deepcopy(dag)
        idx=random.randint(0,len(new.root.parents)-1)
        new.root.parents[idx]=random.choice(self.pool)
        return new

    def run(self,iters=200,temp=5,cooling=0.97):

        curr_acc=self.current.accuracy(self.feats,self.y)
        best_acc=curr_acc

        for i in range(iters):

            new=self.perturb(self.current)
            new_acc=new.accuracy(self.feats,self.y)

            if new_acc>curr_acc or random.random()<math.exp((new_acc-curr_acc)/temp):
                self.current=new
                curr_acc=new_acc

            if curr_acc>best_acc:
                best_acc=curr_acc
                self.best=copy.deepcopy(self.current)

                with open(f"{BASE_RESULTS_DIR}/best_model_{self.name}.pkl","wb") as f:
                    pickle.dump(self.best,f)

            self.history["acc"].append(curr_acc)
            self.history["best"].append(best_acc)
            self.history["temp"].append(temp)

            temp*=cooling

        return self.best

# ------------------------------------------------------------
# VISUALIZE DAG
# ------------------------------------------------------------

def draw_dag(dag, filename):

    G = nx.DiGraph()

    # ---------------------------
    # ROOT NODE (ENSEMBLE)
    # ---------------------------
    root_label = f"Ensemble\n({dag.root.mode})"

    G.add_node(
        "ROOT",
        label=root_label,
        color="lightblue",
        shape="s"
    )

    # ---------------------------
    # CHILD CLASSIFIERS
    # ---------------------------
    for i, parent in enumerate(dag.root.parents):

        node_id = f"C{i}"

        clf_label = (
            f"{parent.key[0]}\n"
            f"{parent.key[1]}\n"
            f"{parent.key[2]}\n"
            f"{parent.key[3]} CSP"
        )

        G.add_node(
            node_id,
            label=clf_label,
            color="lightgreen",
            shape="o"
        )

        G.add_edge("ROOT", node_id)

    # ---------------------------
    # LAYOUT
    # ---------------------------
    pos = nx.spring_layout(G, seed=42)

    labels = nx.get_node_attributes(G, "label")
    colors = [G.nodes[n]["color"] for n in G.nodes]

    plt.figure(figsize=(10,6))

    nx.draw(
        G,
        pos,
        labels=labels,
        node_color=colors,
        node_size=3000,
        font_size=9,
        font_weight="bold",
        arrows=True
    )

    plt.title("Best Ensemble Structure")

    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    print(f"   -> Tree saved: {filename}")

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------

for file_path in FILES_TO_PROCESS:

    name=os.path.basename(file_path).split("_")[-1].replace(".mat","")
    print("\nProcessing",name)

    X,y,fs=load_bciciv_data(file_path)

    X_tmp,X_test,y_tmp,y_test=train_test_split(
        X,y,test_size=0.15,stratify=y,random_state=SEED)

    X_train,X_val,y_train,y_val=train_test_split(
        X_tmp,y_tmp,test_size=0.176,stratify=y_tmp,random_state=SEED)

    feats=extract_features(X_train,X_val,X_test,y_train,fs)

    pool=[BaseClassifier(k) for k in feats.keys()]

    for p in pool:
        p.fit(feats,y_train)

    optimizer=SAOptimizer(pool,feats,y_val,name)
    best_model=optimizer.run()

    # ---------------- TEST ----------------
    y_pred=best_model.root.predict(feats,2)
    acc=accuracy_score(y_test,y_pred)

    print("TEST ACC:",acc)

    # SAVE REPORT
    report=classification_report(y_test,y_pred,output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        f"{BASE_RESULTS_DIR}/results_{name}.csv")

    # CONFUSION MATRIX (BIG FONT)
    cm=confusion_matrix(y_test,y_pred)
    disp=ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues",values_format='d')
    for t in disp.text_.ravel():
        t.set_fontsize(16)

    plt.title(f"{name} Confusion Matrix\nAcc={acc:.2%}")
    plt.savefig(f"{BASE_RESULTS_DIR}/confusion_{name}.png")
    plt.close()

    # DRAW DAG
    draw_dag(best_model,
             f"{BASE_RESULTS_DIR}/tree_{name}.png")

    # OPT HISTORY
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(optimizer.history["acc"],alpha=.5)
    plt.plot(optimizer.history["best"],color="red")
    plt.title("Optimization Accuracy")

    plt.subplot(122)
    plt.plot(optimizer.history["temp"])
    plt.title("Temperature")

    plt.savefig(f"{BASE_RESULTS_DIR}/optimization_{name}.png")
    plt.close()

print("\nDONE.")