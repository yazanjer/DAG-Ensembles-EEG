# ============================================================
# EEG CSP + CSSP ENSEMBLE WITH SIMULATED ANNEALING
# FULL PROFESSIONAL IMPLEMENTATION (FINAL)
# ============================================================

import os
import copy
import math
import pickle
import random
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from pathlib import Path
import mne
from mne.decoding import CSP

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
mne.set_log_level("WARNING")

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_RESULTS_DIR = PROJECT_ROOT / "results_sa_different_svms_kernels"
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

SVM_CONFIGS = [

    # ---------- ORIGINAL ----------
    {"kernel": "rbf", "C": 1,   "gamma": "scale"},
    {"kernel": "rbf", "C": 10,  "gamma": "scale"},
    {"kernel": "rbf", "C": 1,   "gamma": 0.1},
    {"kernel": "rbf", "C": 0.1, "gamma": "scale"},

    # ---------- NEW (ADDED) ----------

    # 1️⃣ Strong regularization (good for noisy CSP features)
    {"kernel": "rbf", "C": 0.01, "gamma": "scale"},

    # 2️⃣ High-complexity boundary
    {"kernel": "rbf", "C": 100, "gamma": "scale"},

    # 3️⃣ Very smooth kernel (large radius)
    {"kernel": "rbf", "C": 1, "gamma": 0.01},

    # 4️⃣ Linear SVM (VERY important baseline for CSP)
    {"kernel": "linear", "C": 1},

]

# ------------------------------------------------------------
# DATA LOADER (FIXED BCICIV)
# ------------------------------------------------------------

def load_bciciv_data(path):

    data = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)

    cnt = data['cnt'].T
    fs = int(data['nfo'].fs)
    pos = data['mrk'].pos
    y = data['mrk'].y

    duration = int(4 * fs)

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
# SAFE CROP
# ------------------------------------------------------------

def safe_crop(ep,tmin,tmax):
    return ep.crop(tmin=tmin,
                   tmax=min(tmax,ep.tmax),
                   verbose=False)

# ------------------------------------------------------------
# CSSP
# ------------------------------------------------------------

def cssp_transform(X,delays=2):

    if delays==0:
        return X

    T,C,S = X.shape
    out=[]
    for d in range(delays+1):
        out.append(X[:,:,d:S-delays+d])

    return np.concatenate(out,axis=1)

# ------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_features(X_train, X_val, X_test, y_train, fs):

    ch = [f"Ch{i}" for i in range(X_train.shape[1])]
    info = mne.create_info(ch, fs, 'eeg')

    ep_tr = mne.EpochsArray(X_train, info, verbose=False)
    ep_va = mne.EpochsArray(X_val, info, verbose=False)
    ep_te = mne.EpochsArray(X_test, info, verbose=False)

    bank = {}

    for band in FREQUENCY_BANDS:
        for tw in TIME_WINDOWS_ANALYSIS:
            for n_comp in CSP_COMPONENTS:

                # ---------------- FILTER + CROP ----------------
                tr = safe_crop(ep_tr.copy().filter(*band), *tw)
                va = safe_crop(ep_va.copy().filter(*band), *tw)
                te = safe_crop(ep_te.copy().filter(*band), *tw)

                Xtr, Xva, Xte = tr.get_data(), va.get_data(), te.get_data()

                # ==================================================
                # CSP FEATURES
                # ==================================================
                csp = CSP(n_components=n_comp, log=True, norm_trace=False)

                Ftr = csp.fit_transform(Xtr, y_train)
                Fva = csp.transform(Xva)
                Fte = csp.transform(Xte)

                # ----- STANDARD SCALING -----
                scaler = StandardScaler()
                Ftr = scaler.fit_transform(Ftr)
                Fva = scaler.transform(Fva)
                Fte = scaler.transform(Fte)

                bank[("CSP", band, tw, n_comp)] = (Ftr, Fva, Fte)

                # ==================================================
                # CSSP FEATURES
                # ==================================================
                Xtr2 = cssp_transform(Xtr)
                Xva2 = cssp_transform(Xva)
                Xte2 = cssp_transform(Xte)

                csp2 = CSP(n_components=n_comp, log=True, norm_trace=False)

                Ftr2 = csp2.fit_transform(Xtr2, y_train)
                Fva2 = csp2.transform(Xva2)
                Fte2 = csp2.transform(Xte2)

                # ----- STANDARD SCALING -----
                scaler2 = StandardScaler()
                Ftr2 = scaler2.fit_transform(Ftr2)
                Fva2 = scaler2.transform(Fva2)
                Fte2 = scaler2.transform(Fte2)

                bank[("CSSP", band, tw, n_comp)] = (Ftr2, Fva2, Fte2)

    print("Feature configs:", len(bank))
    return bank
# ------------------------------------------------------------
# BASE CLASSIFIER
# ------------------------------------------------------------

class BaseClassifier:

    def __init__(self,key,svm_params):

        self.key=key
        self.params=svm_params
        self.base_id=str(svm_params)

        self.model=SVC(probability=True,
                       random_state=SEED,
                       **svm_params)

        self.id=f"{key}_{self.base_id}"

    def fit(self,feats,y):
        self.model.fit(feats[self.key][0],y)

    def predict_proba(self,feats,split):
        return self.model.predict_proba(feats[self.key][split])


# ------------------------------------------------------------
# ENSEMBLE NODE
# ------------------------------------------------------------

class EnsembleNode:

    def __init__(self,parents,mode="SV"):
        self.parents=parents
        self.mode=mode
        self.meta_model=None

    def fit_meta(self,feats,y_val):

        if self.mode!="STACK":
            return

        meta_X=np.hstack([
            p.predict_proba(feats,1)
            for p in self.parents
        ])

        self.meta_model=LogisticRegression(
            max_iter=1000,
            random_state=SEED
        )
        self.meta_model.fit(meta_X,y_val)

    def predict_proba(self,feats,split):

        probs=np.array([
            p.predict_proba(feats,split)
            for p in self.parents
        ])

        if self.mode=="SV":
            return probs.mean(axis=0)

        if self.mode=="HV":
            preds=np.argmax(probs,axis=2)
            maj=np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,arr=preds)
            out=np.zeros((len(maj),probs.shape[-1]))
            out[np.arange(len(maj)),maj]=1
            return out

        if self.mode=="MIN":
            return np.min(probs,axis=0)

        if self.mode=="STACK":
            meta_X=np.hstack([
                p.predict_proba(feats,split)
                for p in self.parents
            ])
            return self.meta_model.predict_proba(meta_X)

        return probs.mean(axis=0)

    def predict(self,feats,split):
        return np.argmax(self.predict_proba(feats,split),axis=1)


class EnsembleDAG:

    def __init__(self,root):
        self.root=root

    def fit_meta(self,feats,y_val):
        self.root.fit_meta(feats,y_val)

    def accuracy(self,feats,y):
        pred=self.root.predict(feats,1)
        return accuracy_score(y,pred)


# ------------------------------------------------------------
# SIMULATED ANNEALING
# ------------------------------------------------------------

class SAOptimizer:

    def __init__(self,pool,feats,y_val,name):

        self.feats=feats
        self.y=y_val
        self.name=name

        self.history={"acc":[],"best":[],"temp":[]}

        # group by SAME base model
        self.groups={}
        for clf in pool:
            self.groups.setdefault(clf.base_id,[]).append(clf)

        self.current=self.random_dag()
        self.best=copy.deepcopy(self.current)

    def random_dag(self):

        base_id=random.choice(list(self.groups.keys()))
        
        parents=random.sample(self.groups[base_id],4)
        # mode=random.choice(["SV","HV","MIN","STACK"])
        mode=random.choice(["SV","HV","MIN"])

        dag=EnsembleDAG(EnsembleNode(parents,mode))
        dag.fit_meta(self.feats,self.y)
        return dag

    def perturb(self,dag):

        new=copy.deepcopy(dag)
        base_id=new.root.parents[0].base_id
        candidates=self.groups[base_id]

        idx=random.randrange(len(new.root.parents))
        new.root.parents[idx]=random.choice(candidates)

        if random.random()<0.3:
            # new.root.mode=random.choice(["SV","HV","MIN","STACK"])
            new.root.mode=random.choice(["SV","HV","MIN"])

        new.fit_meta(self.feats,self.y)
        return new

    def run(self,iters=500,temp=8,cooling=0.97):

        curr=self.current.accuracy(self.feats,self.y)
        best=curr

        for _ in range(iters):

            new=self.perturb(self.current)
            new_acc=new.accuracy(self.feats,self.y)

            if new_acc>curr or random.random()<math.exp((new_acc-curr)/temp):
                self.current=new
                curr=new_acc

            if curr>best:
                best=curr
                self.best=copy.deepcopy(self.current)

                pickle.dump(
                    self.best,
                    open(f"{BASE_RESULTS_DIR}/best_model_{self.name}.pkl","wb")
                )

            self.history["acc"].append(curr)
            self.history["best"].append(best)
            self.history["temp"].append(temp)

            temp*=cooling

        return self.best


# ------------------------------------------------------------
# DRAW DAG (ROOT SHOWS ENSEMBLE TYPE)
# ------------------------------------------------------------

import textwrap

def draw_dag(dag, filename):

    G = nx.DiGraph()

    # ---------- ROOT ----------
    G.add_node(
        "ROOT",
        label=f"Ensemble\nMode: {dag.root.mode}"
    )

    # ---------- CHILD MODELS ----------
    for i, p in enumerate(dag.root.parents):

        node = f"M{i}"

        # Wrap long labels into multiple lines
        wrapped_label = "\n".join(
            textwrap.wrap(str(p.id), width=35)
        )

        G.add_node(node, label=wrapped_label)
        G.add_edge("ROOT", node)

    # ---------- LAYOUT ----------
    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.2,        # more spacing between nodes
        iterations=100
    )

    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=6500,
        node_color="lightblue",
        edgecolors="black"
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowsize=20,
        width=2
    )

    # ---------- CUSTOM LABEL DRAWING ----------
    labels = nx.get_node_attributes(G, "label")

    for node, (x, y) in pos.items():
        ax.text(
            x, y,
            labels[node],
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.4",
                fc="white",
                ec="black",
                alpha=0.9
            )
        )

    plt.title("Ensemble Structure", fontsize=14)

    # Prevent clipping
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# MAIN PIPELINE
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

    pool=[BaseClassifier(k,cfg)
          for k in feats.keys()
          for cfg in SVM_CONFIGS]

    for clf in pool:
        clf.fit(feats,y_train)

    optimizer=SAOptimizer(pool,feats,y_val,name)
    best_model=optimizer.run()

    # ---------------- TEST ----------------
    y_pred=best_model.root.predict(feats,2)
    acc=accuracy_score(y_test,y_pred)
    print("TEST ACC:",acc)

    # REPORT
    report=classification_report(y_test,y_pred,output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        f"{BASE_RESULTS_DIR}/results_{name}.csv")

    # CONFUSION MATRIX
    cm=confusion_matrix(y_test,y_pred)
    disp=ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues",values_format='d')

    for t in disp.text_.ravel():
        t.set_fontsize(18)

    plt.title(f"{name} Confusion Matrix\nAcc={acc:.2%}")
    plt.savefig(f"{BASE_RESULTS_DIR}/confusion_{name}.png")
    plt.close()

    draw_dag(best_model,
             f"{BASE_RESULTS_DIR}/tree_{name}.png")

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(optimizer.history["acc"],alpha=.5)
    plt.plot(optimizer.history["best"],color="red")
    plt.title("Optimization Accuracy")

    plt.subplot(122)
    plt.plot(optimizer.history["temp"])
    plt.title("Temperature")

    plt.savefig(f"{BASE_RESULTS_DIR}/optimization_{name}.png")
    plt.show()
    plt.close()

print("\nDONE.")