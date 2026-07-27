import numpy as np
import scipy.io
import copy
import math
import random
import itertools
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
import networkx as nx 
from enum import Enum
from scipy.signal import butter, lfilter
from collections import Counter
from pathlib import Path
# ==============================================================================
# 0. COMPATIBILITY PATCH
# ==============================================================================
import sklearn.utils.validation
_original_check_X_y = sklearn.utils.validation.check_X_y

def _patched_check_X_y(X, y, **kwargs):
    if 'force_writeable' in kwargs:
        del kwargs['force_writeable']
    return _original_check_X_y(X, y, **kwargs)

sklearn.utils.validation.check_X_y = _patched_check_X_y

# ==============================================================================
# 1. IMPORTS & SETUP
# ==============================================================================
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, mean_squared_error
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
print(f"Random Seed Frozen: {SEED}")
WINDOW_SIZE = [(0, 4),(1,3)]
# WINDOW_START = 0
# WINDOW_END = 4
COMPONENTS = [[6, 7, 8], [4, 5, 6, 7, 8]]  # experm1, experm2
# NOTE: results directories are derived from PROJECT_ROOT at runtime (see the
# __main__ block). No absolute/personal paths remain (Reviewer 1 #6).
# ==============================================================================
# 2. CONFIGURATION & ENUMS
# ==============================================================================

class FrequencyBand(Enum):
    ALPHA = (8, 13)
    BETA = (14, 30)
    FULL_MU = (8, 30)
    LOW_MU = (7, 15)

class FeatureType(Enum):
    CSP = "CSP"
    CTP = "CTP"

class AlgorithmType(Enum):
    LDA = "LDA"
    SVM = "SVM"

class OperatorType(Enum):
    MV = "Majority Voting"
    HV = "Hard Voting"
    SV = "Soft Voting"
    MIN = "Min Probability"
    ST = "Stacking"

# ==============================================================================
# 3. VISUALIZATION UTILS
# ==============================================================================

def plot_split_distribution(y_train, y_val, y_test, class_names, dataset_name,BASE_RESULTS_DIR):
    """
    Plots the distribution of classes across Train, Val, and Test sets side-by-side.
    """
    # Count classes for each set
    sets = {'Train': y_train, 'Val': y_val, 'Test': y_test}
    
    # Prepare data for plotting
    labels = class_names # e.g., ['left', 'right']
    n_classes = len(labels)
    
    # Extract counts: row=set, col=class
    counts = {s_name: [np.sum(y_data == 0), np.sum(y_data == 1)] for s_name, y_data in sets.items()}
    
    x = np.arange(len(labels))  # label locations
    width = 0.25  # width of bars

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot bars
    rects1 = ax.bar(x - width, counts['Train'], width, label='Train', color='#1f77b4')
    rects2 = ax.bar(x, counts['Val'], width, label='Val', color='#ff7f0e')
    rects3 = ax.bar(x + width, counts['Test'], width, label='Test', color='#2ca02c')

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of Trials')
    ax.set_title(f'Label Distribution by Split: {dataset_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Label bars with counts
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    filename = os.path.join(BASE_RESULTS_DIR, f"split_dist_{dataset_name}.png")
    plt.savefig(filename)
    plt.close()
    print(f"   -> Split Distribution plot saved: {filename}")

def visualize_model_structure(dag, filename="model_structure.png"):
    G = nx.DiGraph()
    
    def build_graph(node, parent_id=None):
        node_id = str(id(node))
        
        if isinstance(node, EnsembleOperatorNode):
            # Operator Node (Blue Square)
            name = node.op_type.value
            label = "MV" if "Majority" in name else "HV" if "Hard" in name else \
                    "SV" if "Soft" in name else "MIN" if "Min" in name else "ST"
            G.add_node(node_id, label=label, shape='s', color='lightblue')
            
            for child in node.parents:
                child_id = build_graph(child, node_id)
                G.add_edge(node_id, child_id)
                
        elif isinstance(node, BaseClassifierNode):
            # Base Classifier Node (Green Circle)
            # UPDATED: Added algorithm name (node.algo_type.name)
            label = f"{node.band.name}\n{node.feat_type.name}\n{node.n_comp}c\n{node.algo_type.name}"
            G.add_node(node_id, label=label, shape='o', color='lightgreen')
            
        return node_id

    # Build the graph
    build_graph(dag.root)
    
    # Hierarchical Layout Logic
    pos = {}
    def hierarch_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
        pos[root] = (xcenter, vert_loc)
        children = list(G.successors(root))
        if not children: return
        dx = width / len(children) 
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            hierarch_pos(G, child, width = dx, vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, xcenter = nextx)
            
    try: 
        hierarch_pos(G, str(id(dag.root)))
    except: 
        pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 6))
    
    # Get attributes
    node_labels = nx.get_node_attributes(G, 'label')
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    
    # UPDATED DRAWING PARAMETERS
    nx.draw(G, pos, 
            labels=node_labels, 
            with_labels=True, 
            node_size=1200,      # Smaller nodes (was 2500)
            font_size=5,         # Smaller font (was 8)
            node_color=node_colors, 
            font_weight='bold', 
            edge_color='gray', 
            arrows=True)
            
    plt.title("Best Ensemble Structure")
    plt.savefig(filename)
    plt.close()
    print(f"   -> Tree structure saved: {filename}")

# ==============================================================================
# 4. DATA PROCESSING
# ==============================================================================

def create_epochs(cnt, mrk_pos, mrk_y, fs, window_start=0.0, window_end=4.0):
    start_offset = int(window_start * fs)
    end_offset = int(window_end * fs)
    epoch_length = end_offset - start_offset
    num_trials = len(mrk_pos)
    num_channels = cnt.shape[1]
    X = np.zeros((num_trials, num_channels, epoch_length))
    valid_trials = []
    for i in range(num_trials):
        current_pos = mrk_pos[i]
        t_start = current_pos + start_offset
        t_end = current_pos + end_offset
        if t_end <= cnt.shape[0]:
            epoch = cnt[t_start:t_end, :].T 
            X[i] = epoch
            valid_trials.append(i)
    return X[valid_trials], mrk_y[valid_trials]

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=2)

class DataProcessor:
    def __init__(self, fs, component_options):
        self.fs = fs
        self.component_options = component_options
        self.transformers = {} 

    def process_and_split(self, X_raw, y_raw):
        # 3-Way Split: 70% Train, 15% Val, 15% Test
        print(f"--> Splitting Data (Stratified)")
        
        # 1. Extract Test Set (15%)
        X_temp, X_test_raw, y_temp, y_test = train_test_split(
            X_raw, y_raw, test_size=0.15, stratify=y_raw, random_state=SEED
        )
        
        # 2. Split remainder (85%) into Train/Val
        # We want Val to be 15% of TOTAL. 15/85 = 0.1765
        val_size_adjusted = 0.15 / 0.85
        X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=SEED
        )
        
        print(f"    Train: {len(y_tr)} trials | Val: {len(y_val)} trials | Test: {len(y_test)} trials")
        
        X_train_dict = {}
        X_val_dict = {}
        X_test_dict = {}
        
        print(f"--> Extracting Features & Fitting CSPs...")
        for band in FrequencyBand:
            low, high = band.value
            X_tr_filt = butter_bandpass_filter(X_tr_raw, low, high, self.fs)
            X_val_filt = butter_bandpass_filter(X_val_raw, low, high, self.fs)
            X_test_filt = butter_bandpass_filter(X_test_raw, low, high, self.fs)
            
            for feat in FeatureType:
                for n_comp in self.component_options:
                    key = (band, feat, n_comp)

                    if feat == FeatureType.CTP:
                        feat_tr = np.log(np.var(X_tr_filt, axis=2) + 1e-10)
                        feat_val = np.log(np.var(X_val_filt, axis=2) + 1e-10)
                        feat_test = np.log(np.var(X_test_filt, axis=2) + 1e-10)
                        self.transformers[key] = "CTP" 
                    else: 
                        csp = CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)
                        feat_tr = csp.fit_transform(X_tr_filt, y_tr)
                        feat_val = csp.transform(X_val_filt)
                        feat_test = csp.transform(X_test_filt)
                        self.transformers[key] = csp

                    X_train_dict[key] = feat_tr
                    X_val_dict[key] = feat_val
                    X_test_dict[key] = feat_test
                
        return X_train_dict, y_tr, X_val_dict, y_val, X_test_dict, y_test

    def prepare_test_data(self, X_raw):
        # Used if loading model later
        X_dict = {}
        for band in FrequencyBand:
            low, high = band.value
            X_filt = butter_bandpass_filter(X_raw, low, high, self.fs)
            for feat in FeatureType:
                for n_comp in self.component_options:
                    key = (band, feat, n_comp)
                    transformer = self.transformers.get(key)
                    if transformer is None: continue
                    if feat == FeatureType.CTP:
                        X_dict[key] = np.log(np.var(X_filt, axis=2) + 1e-10)
                    else:
                        X_dict[key] = transformer.transform(X_filt)
        return X_dict

# ==============================================================================
# 5. MODEL ARCHITECTURE
# ==============================================================================

class BaseClassifierNode:
    def __init__(self, band, feat_type, n_comp, algo_type, params):
        self.band = band
        self.feat_type = feat_type
        self.n_comp = n_comp
        self.algo_type = algo_type
        self.params = params
        self.id = f"{band.name}_{feat_type.name}_{n_comp}c"
        self.model = self._build_pipeline()

    def _build_pipeline(self):
        steps = []
        if self.algo_type == AlgorithmType.LDA:
            clf = LinearDiscriminantAnalysis(**self.params)
        elif self.algo_type == AlgorithmType.SVM:
            clf = SVC(probability=True, **self.params)
        steps.append(('classifier', clf))
        return Pipeline(steps)

    def _get_data(self, X_dict):
        return X_dict[(self.band, self.feat_type, self.n_comp)]

    def fit(self, X_dict, y):
        self.model.fit(self._get_data(X_dict), y)

    def predict_proba(self, X_dict):
        return self.model.predict_proba(self._get_data(X_dict))
        
    def predict(self, X_dict):
        return self.model.predict(self._get_data(X_dict))

class ClassifierPool:
    def __init__(self, component_options):
        self.pool = [] 
        self.component_options = component_options

    def generate_pool(self):
        self.pool = []
        
        # --- EXPANDED HYPERPARAMETERS (AS REQUESTED) ---
        svm_params = [
            {'kernel': 'linear', 'C': 0.1}, {'kernel': 'linear', 'C': 1.0}, {'kernel': 'linear', 'C': 10.0},
            {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}, {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 100.0, 'gamma': 'auto'}, {'kernel': 'rbf', 'C': 1.0, 'gamma': 0.01},
            {'kernel': 'rbf', 'C': 1.0, 'gamma': 0.1}, {'kernel': 'poly', 'degree': 2, 'C': 1.0},
            {'kernel': 'poly', 'degree': 3, 'C': 1.0}
        ]
        lda_params = [
            {'solver': 'svd'}, {'solver': 'lsqr', 'shrinkage': 'auto'}, 
            {'solver': 'lsqr', 'shrinkage': 0.1}, {'solver': 'lsqr', 'shrinkage': 0.5},
            {'solver': 'eigen', 'shrinkage': 'auto'}
        ]
        
        # component_options = [6, 7, 8]
        combinations = itertools.product(FrequencyBand, FeatureType, component_options)
        
        for band, feat, n_comp in combinations:
            for p in lda_params:
                self.pool.append(BaseClassifierNode(band, feat, n_comp, AlgorithmType.LDA, p))
            for p in svm_params:
                self.pool.append(BaseClassifierNode(band, feat, n_comp, AlgorithmType.SVM, p))

    def pre_train_all(self, X_train_dict, y_train):
        print(f"--> Pre-training {len(self.pool)} base classifiers...")
        for node in self.pool:
            try: node.fit(X_train_dict, y_train)
            except: pass

    def get_random_distinct(self, k=2, exclude=[]):
        candidates = [c for c in self.pool if c not in exclude]
        if len(candidates) < k: raise ValueError("Pool too small")
        return random.sample(candidates, k)

class EnsembleOperatorNode:
    def __init__(self, op_type, parents):
        self.op_type = op_type
        self.parents = parents
        self.meta_clf = None

    def fit(self, X_dict, y):
        if self.op_type == OperatorType.ST:
            probs = [p.predict_proba(X_dict) for p in self.parents]
            self.meta_clf = LogisticRegression(random_state=SEED)
            self.meta_clf.fit(np.hstack(probs), y)

    def predict_proba(self, X_dict):
        probs_list = [p.predict_proba(X_dict) for p in self.parents]
        prob_stack = np.array(probs_list) 
        N_classifiers, N_samples, N_classes = prob_stack.shape

        if self.op_type == OperatorType.MV:
            final_probs = np.zeros((N_samples, N_classes))
            hard_preds = np.argmax(prob_stack, axis=2)
            for i in range(N_samples):
                votes = hard_preds[:, i]
                counts = Counter(votes)
                most, count = counts.most_common(1)[0]
                if count >= 2: final_probs[i, most] = 1.0
                else: 
                    sum_probs = np.sum(prob_stack[:, i, :], axis=0)
                    final_probs[i, :] = sum_probs / (sum_probs.sum() + 1e-10)
            return final_probs
        elif self.op_type == OperatorType.HV: 
            return np.max(prob_stack, axis=0)
        elif self.op_type == OperatorType.SV:
            summed = np.sum(prob_stack, axis=0)
            out = np.zeros_like(summed)
            return np.divide(summed, summed.sum(axis=1, keepdims=True), out=out, where=summed.sum(axis=1, keepdims=True)!=0)
        elif self.op_type == OperatorType.MIN:
            mins = np.min(prob_stack, axis=0)
            out = np.zeros_like(mins)
            return np.divide(mins, mins.sum(axis=1, keepdims=True), out=out, where=mins.sum(axis=1, keepdims=True)!=0)
        elif self.op_type == OperatorType.ST:
            return self.meta_clf.predict_proba(np.hstack(probs_list))
        return np.mean(prob_stack, axis=0)

    def predict(self, X_dict):
        return np.argmax(self.predict_proba(X_dict), axis=1)

class EnsembleDAG:
    def __init__(self, root):
        self.root = root
    def fit_meta_learners(self, X_dict, y):
        def _recursive_fit(node):
            if isinstance(node, EnsembleOperatorNode):
                for parent in node.parents: _recursive_fit(parent)
                if node.op_type == OperatorType.ST: node.fit(X_dict, y)
        _recursive_fit(self.root)
    def accuracy(self, X_dict, y):
        return accuracy_score(y, self.root.predict(X_dict))

# ==============================================================================
# 6. OPTIMIZER
# ==============================================================================

class SimulatedAnnealingOptimizer:
    def __init__(self, pool, X_val, y_val, dataset_name, processor):
        self.pool = pool
        self.X_val = X_val
        self.y_val = y_val
        self.dataset_name = dataset_name
        self.processor = processor 
        self.current_dag = self._create_constrained_dag()
        self.best_dag = self.current_dag
        self.best_acc = 0.0
        self.history = {'accuracy': [], 'temperature': [], 'best_accuracy': []}

    def _create_constrained_dag(self):
        bases = self.pool.get_random_distinct(k=4)
        op1 = EnsembleOperatorNode(random.choice(list(OperatorType)), [bases[0], bases[1]])
        op2 = EnsembleOperatorNode(random.choice(list(OperatorType)), [bases[2], bases[3]])
        root = EnsembleOperatorNode(random.choice(list(OperatorType)), [op1, op2])
        dag = EnsembleDAG(root)
        dag.fit_meta_learners(self.X_val, self.y_val)
        return dag

    def _perturb(self, dag):
        new_dag = copy.deepcopy(dag)
        mutation_options = ['swap_base', 'change_op', 'change_root', 'add_base', 'delete_base']
        mutation_type = random.choice(mutation_options)
        target_op = new_dag.root.parents[random.randint(0, 1)]
        
        if mutation_type == 'swap_base':
            child_idx = random.randint(0, len(target_op.parents) - 1)
            exclude = target_op.parents[:child_idx] + target_op.parents[child_idx+1:]
            target_op.parents[child_idx] = self.pool.get_random_distinct(k=1, exclude=exclude)[0]
        elif mutation_type == 'change_op': target_op.op_type = random.choice(list(OperatorType))
        elif mutation_type == 'change_root': new_dag.root.op_type = random.choice(list(OperatorType))
        elif mutation_type == 'add_base':
            try: target_op.parents.append(self.pool.get_random_distinct(k=1, exclude=target_op.parents)[0])
            except: pass
        elif mutation_type == 'delete_base':
            if len(target_op.parents) > 2: del target_op.parents[random.randint(0, len(target_op.parents) - 1)]
            else: 
                child_idx = random.randint(0, len(target_op.parents) - 1)
                exclude = target_op.parents[:child_idx] + target_op.parents[child_idx+1:]
                target_op.parents[child_idx] = self.pool.get_random_distinct(k=1, exclude=exclude)[0]
        return new_dag

    def run(self, iterations=100, temp=5.0, cooling_rate=0.95, nreheat=20):
        curr_acc = self.current_dag.accuracy(self.X_val, self.y_val)
        self.best_acc = curr_acc
        stagnant = 0
        
        print(f"    Init Accuracy: {curr_acc:.2%}")
        
        for i in range(iterations):
            new_dag = self._perturb(self.current_dag)
            new_dag.fit_meta_learners(self.X_val, self.y_val)
            new_acc = new_dag.accuracy(self.X_val, self.y_val)
            
            delta_e = curr_acc - new_acc
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
                self.current_dag = new_dag
                curr_acc = new_acc
                
                if curr_acc > self.best_acc:
                    self.best_acc = curr_acc
                    self.best_dag = copy.deepcopy(new_dag)
                    
                    filename = os.path.join(BASE_RESULTS_DIR, f"best_model_{self.dataset_name}.pkl")
                    save_bundle = {'model': self.best_dag, 'processor': self.processor}
                    # with open(filename, 'wb') as f:
                    #     pickle.dump(save_bundle, f)
                    
                    print(f"    Iter {i}: Best! {self.best_acc:.2%} (Saved {filename})")
                    stagnant = 0
            
            self.history['accuracy'].append(curr_acc)
            self.history['temperature'].append(temp)
            self.history['best_accuracy'].append(self.best_acc)
            
            temp *= cooling_rate
            stagnant += 1
            if stagnant > nreheat:
                temp *= 1.5
                stagnant = 0
                print(f"    --> Reheating! (Temp boosted to {temp:.2f})")
            
            if temp < 0.0001: break
        return self.best_dag

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    for component_options in COMPONENTS:
        for window_start, window_end in WINDOW_SIZE:

            PROJECT_ROOT = Path(__file__).resolve().parent

            # 2. Set the results directory relative to the project root
            if component_options == [6, 7, 8]:
                folder_name = f'results_experm1_{window_start}t{window_end}_SVMandLDA'
            else:
                folder_name = f'results_experm2_{window_start}t{window_end}_SVMandLDA'

            # The '/' operator automatically joins paths correctly for Windows/Mac/Linux
            BASE_RESULTS_DIR = PROJECT_ROOT / folder_name

            # Create the directory (pathlib has a built-in way to do this)
            BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            # 3. Set the dataset directory and files dynamically
            DATASET_DIR = PROJECT_ROOT / 'dataset'

            # Create the list of file paths
            files_to_process_paths = [
                DATASET_DIR / 'BCICIV_calib_ds1a.mat',
                DATASET_DIR / 'BCICIV_calib_ds1b.mat',
                DATASET_DIR / 'BCICIV_calib_ds1f.mat',
                DATASET_DIR / 'BCICIV_calib_ds1g.mat',
            ]
            
            for f_path in files_to_process_paths:
                print("\n" + "="*60)
                dataset_name = os.path.splitext(os.path.basename(f_path))[0].split('_')[-1]
                print(f"PROCESSING DATASET: {dataset_name} ({f_path})")
                print("="*60)

                try:
                    # 1. LOAD DATA
                    data = scipy.io.loadmat(f_path, struct_as_record=True)
                    raw_eeg = data['cnt'].astype(float)
                    mrk_pos = data['mrk']['pos'][0][0].flatten()
                    mrk_y = data['mrk']['y'][0][0].flatten()
                    
                    fs = int(data['nfo']['fs'][0][0].flatten()[0]) 
                    classes = [c[0] for c in data['nfo']['classes'][0][0].flatten()]
                    print(f"Classes: {classes}")

                    # 2. CREATE EPOCHS
                    X, y_raw = create_epochs(raw_eeg, mrk_pos, mrk_y, fs, window_start=window_start, window_end=window_end)
                    y_binary = np.where(y_raw == -1, 0, 1) # Map to 0/1

                    # 3. PROCESS (With 3-Way Split)
                    processor = DataProcessor(fs,component_options)
                    # Unpack the 3 splits
                    X_train_dict, y_train_opt, X_val_dict, y_val_opt, X_test_dict, y_test_opt = processor.process_and_split(X, y_binary)
                    
                    # --- NEW: PLOT SPLIT DISTRIBUTION ---
                    plot_split_distribution(y_train_opt, y_val_opt, y_test_opt, classes, dataset_name,BASE_RESULTS_DIR)

                    # 4. PRE-TRAIN POOL (Full Grid)
                    pool = ClassifierPool(component_options)
                    pool.generate_pool()
                    pool.pre_train_all(X_train_dict, y_train_opt)

                    # 5. OPTIMIZE (On Validation Set)
                    # --- UPDATED PARAMETERS HERE ---
                    print(f"\nStarting Optimization for {dataset_name}...")
                    optimizer = SimulatedAnnealingOptimizer(pool, X_val_dict, y_val_opt, dataset_name, processor)
                    best_model = optimizer.run(iterations=300, temp=5.0, cooling_rate=0.97, nreheat=20)

                    # ==================================================================
                    # 6. TESTING PHASE (On Held-Out Test Set)
                    # ==================================================================
                    print("\n" + "-"*40)
                    print(f"TESTING FINAL MODEL ON HELD-OUT DATA (15%)")
                    print("-" * 40)
                    
                    y_pred_test = best_model.root.predict(X_test_dict)
                    final_test_acc = accuracy_score(y_test_opt, y_pred_test)
                    final_test_mse = mean_squared_error(y_test_opt, y_pred_test)
                    
                    print(f">>> TEST ACCURACY: {final_test_acc:.2%}")
                    print(f">>> TEST MSE:      {final_test_mse:.4f}")

                    # --- Save Report to CSV ---
                    report = classification_report(y_test_opt, y_pred_test, target_names=[str(c) for c in classes], output_dict=True)
                    df_report = pd.DataFrame(report).transpose()
                    csv_filename = os.path.join(BASE_RESULTS_DIR, f"results_{dataset_name}.csv")
                    # df_report.to_csv(csv_filename)
                    print(f"   -> Report saved to: {csv_filename}")

                    # --- Plot Confusion Matrix ---
                    cm = confusion_matrix(y_test_opt, y_pred_test)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

                    fig, ax = plt.subplots(figsize=(6, 6))

                    disp.plot(
                        cmap=plt.cm.Blues,
                        ax=ax,
                        values_format='d',
                        text_kw={'fontsize': 14}   # <-- numbers inside cells
                    )

                    # Axis label sizes
                    ax.set_xlabel("Predicted label", fontsize=14)
                    ax.set_ylabel("True label", fontsize=14)

                    # Tick label sizes
                    ax.tick_params(axis='both', which='major', labelsize=12)

                    # Title size
                    plt.title(f"{dataset_name} Test Confusion Matrix\nAcc: {final_test_acc:.2%}", fontsize=15)

                    # Optional: thicker grid + borders (pairs well with larger text)
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)

                    plt.savefig(os.path.join(BASE_RESULTS_DIR, f"confusion_{dataset_name}.pdf"))
                    plt.close()

                    print(f"   -> Confusion Matrix saved: confusion_{dataset_name}.pdf")

                    # 7. SAVE OPTIMIZATION PLOT
                    # visualize_model_structure(best_model, os.path.join(BASE_RESULTS_DIR, f"tree_{dataset_name}.pdf"))
                    history_df = pd.DataFrame(optimizer.history)
                    history_df.to_csv(os.path.join(BASE_RESULTS_DIR, f"history_{dataset_name}.csv"), index=False)
                    plt.figure(figsize=(12, 5))

                    plt.subplot(1, 2, 1)
                    plt.plot(optimizer.history['accuracy'], label='Val Acc', alpha=0.6, linewidth=2.5)
                    plt.plot(optimizer.history['best_accuracy'], label='Best Val', color='red', linewidth=3)
                    plt.title(f'{dataset_name}: Optimization')
                    plt.legend()

                    plt.subplot(1, 2, 2)
                    plt.plot(optimizer.history['temperature'], color='orange', linewidth=2.5)
                    plt.title(f'{dataset_name}: Cooling')

                    plt.savefig(os.path.join(BASE_RESULTS_DIR, f"optimization_{dataset_name}.pdf"))
                    plt.close()

                except Exception as e:
                    print(f"ERROR processing {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()