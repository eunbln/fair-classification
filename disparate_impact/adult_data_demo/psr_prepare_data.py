import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from random import shuffle, seed

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

def load_bank_data(filepath, load_data_size=None, svm=False, random_state=42, intercept=False, train_frac=0.7):
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

    return Xtr, Xte, ytr, yte, Ztr, Zte

# Call the function and print the results
filepath = '/content/fair-classification/disparate_impact/adult_data_demo/bank-full.csv'
Xtr, Xte, ytr, yte, Ztr, Zte = load_bank_data(filepath)
print("Train data shape:", Xtr.shape)
print("Test data shape:", Xte.shape)
print("Train labels shape:", ytr.shape)
print("Test labels shape:", yte.shape)
print("Train sensitive data shape:", Ztr.shape)
print("Test sensitive data shape:", Zte.shape)
