import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from random import shuffle, seed
from scipy.optimize import minimize

# 경로를 시스템 경로에 추가
sys.path.insert(0, '../../fair_classification/')  # fair classification 코드가 있는 디렉토리를 추가
import utils as ut
import loss_funcs as lf  # loss funcs that can be optimized subject to various constraints

def get_one_hot_encoding(attr_vals):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_vals = encoder.fit_transform(np.array(attr_vals).reshape(-1, 1))
    return encoded_vals, {val: i for i, val in enumerate(encoder.categories_[0])}

def _get_train_test_split(n, train_frac, random_state):
    np.random.seed(random_state)
    indices = np.random.permutation(n)
    train_size = int(n * train_frac)
    return indices[:train_size], indices[train_size:]

def _apply_train_test_split(X, y, Z, tr_idx, te_idx):
    Xtr = X[tr_idx]
    Xte = X[te_idx]
    ytr = y[tr_idx]
    yte = y[te_idx]
    Ztr = Z[tr_idx]
    Zte = Z[te_idx]
    return Xtr, Xte, ytr, yte, Ztr, Zte

def load_bank_data(filepath='/content/fair-classification/disparate_impact/adult_data_demo/bank-full.csv', 
                   load_data_size=None, svm=False, random_state=42, intercept=False, train_frac=0.7):
    bank = pd.read_csv(filepath)
    print("Initial data loaded")

    bank.loc[bank['marital'] != 'married', 'marital'] = 0
    bank.loc[bank['marital'] == 'married', 'marital'] = 1
    print("Marital status converted")

    if svm:
        bank['y'] = bank['y'].map({"no": -1, "yes": 1})
    else:
        bank['y'] = bank['y'].map({"no": 0, "yes": 1})
    print("Target variable converted")

    attrs = bank.columns
    int_attrs = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    sensitive_attrs = ['marital']
    attrs_to_ignore = ['marital', 'y']
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    X = []
    y = []
    x_control = {}

    attrs_to_vals = {}
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    y = bank.values[:, -1]

    for i in range(len(bank)):
        line = bank.iloc[i].values
        for j in range(0, len(line)):
            attr_name = attrs[j]
            attr_val = line[j]
            if attr_name in sensitive_attrs:
                x_control[attr_name].append(attr_val)
            elif attr_name in attrs_to_ignore:
                pass
            else:
                attrs_to_vals[attr_name].append(attr_val)
    print("Attributes and control variables separated")

    def convert_attrs_to_ints(d):
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs:
                continue
            attr_vals = [str(val) for val in attr_vals]  # Convert all values to strings
            uniq_vals = sorted(list(set(attr_vals)))
            val_dict = {val: i for i, val in enumerate(uniq_vals)}
            d[attr_name] = [val_dict[val] for val in attr_vals]

    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)
    print("Attributes converted to integers")

    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country":
            X.append(attr_vals)
        else:
            attr_vals, index_dict = get_one_hot_encoding(attr_vals)
            if attr_vals.shape == (45211,):
                attr_vals = attr_vals.reshape(45211, 1)
            for inner_col in attr_vals.T:
                X.append(inner_col)
    print("One-hot encoding applied")

    X = np.array(X, dtype=float).T
    y = np.array(y, dtype=float)
    for k, v in x_control.items():
        x_control[k] = np.array(v, dtype=float)

    perm = list(range(0, len(y)))
    seed(random_state)
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]
    print("Data shuffled")

    if load_data_size is not None:
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    n = X.shape[0]
    Z = np.expand_dims(x_control['marital'], axis=-1)
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z, tr_idx, te_idx)
    print("Train-test split applied")

    mm = MinMaxScaler()
    Xtr = mm.fit_transform(Xtr)
    Xte = mm.transform(Xte)
    print("Feature scaling applied")

    if intercept:
        Xtr = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
        Xte = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
        print("Intercept added")

    return Xtr, Xte, ytr, yte, Ztr.flatten(), Zte.flatten()

def test_bank_data():
    """ Load the bank data """
    filepath = '/content/fair-classification/disparate_impact/adult_data_demo/bank-full.csv'
    Xtr, Xte, ytr, yte, Ztr, Zte = load_bank_data(filepath)
    x_control = {'marital': Ztr}

    ut.compute_p_rule(x_control["marital"], ytr)  # compute the p-rule in the original data

    """ Split the data into train and test """
    Xtr = ut.add_intercept(Xtr)  # add intercept to X before applying the linear classifier
    Xte = ut.add_intercept(Xte)  # add intercept to X before applying the linear classifier

    x_train, y_train, x_control_train = Xtr, ytr, x_control
    x_test, y_test, x_control_test = Xte, yte, {'marital': Zte}

    apply_fairness_constraints = None
    apply_accuracy_constraint = None
    sep_constraint = None

    loss_function = lf.logistic_loss
    sensitive_attrs = ["marital"]
    sensitive_attrs_to_cov_thresh = {}
    gamma = None

    def train_test_classifier():
        w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
        train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
        distances_boundary_test = (np.dot(x_test, w)).tolist()
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
        cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
        p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])
        return w, p_rule, test_score

    """ Classify the data while optimizing for accuracy """
    print()
    print("== Unconstrained (original) classifier ==")
    # all constraint flags are set to 0 since we want to train an unconstrained (original) classifier
    apply_fairness_constraints = 0
    apply_accuracy_constraint = 0
    sep_constraint = 0
    w_uncons, p_uncons, acc_uncons = train_test_classifier()

    """ Now classify such that we optimize for accuracy while achieving perfect fairness """
    apply_fairness_constraints = 1  # set this flag to one since we want to optimize accuracy subject to fairness constraints
    apply_accuracy_constraint = 0
    sep_constraint = 0
    sensitive_attrs_to_cov_thresh = {"marital": 0}
    print()
    print("== Classifier with fairness constraint ==")
    w_f_cons, p_f_cons, acc_f_cons = train_test_classifier()

    """ Classify such that we optimize for fairness subject to a certain loss in accuracy """
    apply_fairness_constraints = 0  # flag for fairness constraint is set back to 0 since we want to apply the accuracy constraint now
    apply_accuracy_constraint = 1  # now, we want to optimize fairness subject to accuracy constraints
    sep_constraint = 0
    gamma = 0.5  # gamma controls how much loss in accuracy we are willing to incur to achieve fairness -- increase gamma to allow more loss in accuracy
    print("== Classifier with accuracy constraint ==")
    w_a_cons, p_a_cons, acc_a_cons = train_test_classifier()

    """ 
    Classify such that we optimize for fairness subject to a certain loss in accuracy 
    In addition, make sure that no points classified as positive by the unconstrained (original) classifier are misclassified.
    """
    apply_fairness_constraints = 0  # flag for fairness constraint is set back to 0 since we want to apply the accuracy constraint now
    apply_accuracy_constraint = 1  # now, we want to optimize accuracy subject to fairness constraints
    sep_constraint = 1  # set the separate constraint flag to one, since in addition to accuracy constraints, we also want no misclassifications for certain points (details in demo README.md)
    gamma = 1000.0
    print("== Classifier with accuracy constraint (no +ve misclassification) ==")
    w_a_cons_fine, p_a_cons_fine, acc_a_cons_fine = train_test_classifier()

    return

def main():
    test_bank_data()

if __name__ == '__main__':
    main()
