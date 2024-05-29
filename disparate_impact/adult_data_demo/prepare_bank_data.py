import os
import sys
import urllib.request
sys.path.insert(0, '../../fair_classification/')  # 공정성 분류 코드를 포함하는 디렉토리를 시스템 경로에 추가합니다.
import utils as ut  # 공정성 분류를 위한 유틸리티 함수들이 정의된 모듈
import numpy as np
from random import seed, shuffle  # 난수 생성 및 데이터 셔플링을 위한 모듈
import pandas as pd

SEED = 1122334455  # 랜덤 시드를 설정하여 재현 가능한 결과를 얻습니다.
seed(SEED)
np.random.seed(SEED)

def get_one_hot_encoding(vals):
    """
    주어진 값들을 원 핫 인코딩합니다.
    """
    unique_vals = list(set(vals))
    index_dict = {val: i for i, val in enumerate(unique_vals)}
    one_hot_encoded = np.zeros((len(vals), len(unique_vals)))

    for i, val in enumerate(vals):
        one_hot_encoded[i, index_dict[val]] = 1

    return one_hot_encoded, index_dict

def load_bank_data(load_data_size=None):
    # CSV 파일을 로드할 때 구분자를 세미콜론으로 설정합니다.
    try:
        bank = pd.read_csv('./bank-full.csv', delimiter=';')
        print("CSV 파일 로드 성공")
        print("열 이름:", bank.columns)
    except FileNotFoundError:
        print("CSV 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return None, None, None

    # 'marital' 열이 존재하는지 확인합니다.
    if 'marital' not in bank.columns:
        print("'marital' 열을 찾을 수 없습니다.")
        return None, None, None

    bank['marital'] = bank['marital'].apply(lambda x: 1 if x == 'married' else 0)

    """
    은행 데이터를 로드하고 전처리한 후 훈련 및 테스트 데이터로 분할합니다.
    """
    # 속성과 민감 속성을 정의합니다.
    attrs = bank.columns
    int_attrs = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    sensitive_attrs = ['marital']
    attrs_to_ignore = ['marital', 'education']
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    X = []  # 특성 데이터
    y = []  # 타겟 데이터
    x_control = {}  # 민감 속성 데이터

    attrs_to_vals = {}  # 속성 값들을 저장할 딕셔너리
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []  # 민감 속성의 값들을 저장할 리스트 초기화
        elif k in attrs_to_ignore:
            pass  # 무시할 속성은 넘어갑니다.
        else:
            attrs_to_vals[k] = []  # 속성 값들을 저장할 리스트 초기화

    for _, row in bank.iterrows():
        class_label = row['y']
        if class_label in ["no.", "no"]:
            class_label = -1  # 클래스 레이블을 -1로 설정합니다.
        elif class_label in ["yes.", "yes"]:
            class_label = +1  # 클래스 레이블을 +1로 설정합니다.
        else:
            raise Exception("Invalid class label value")

        y.append(class_label)  # 클래스 레이블을 추가합니다.

        for attr_name in attrs:
            attr_val = row[attr_name]
            if attr_name in sensitive_attrs:
                x_control[attr_name].append(attr_val)
            elif attr_name in attrs_to_ignore:
                pass  # 무시할 속성은 넘어갑니다.
            else:
                attrs_to_vals[attr_name].append(attr_val)  # 속성 값들을 추가합니다.

    def convert_attrs_to_ints(d):
        """
        문자열로 된 속성 값을 정수로 변환합니다.
        """
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs:
                continue  # 이미 정수인 경우 건너뜁니다.
            uniq_vals = sorted(list(set(attr_vals)))  # 고유값을 가져옵니다.

            # 각 고유값에 정수 코드를 할당합니다.
            val_dict = {val: i for i, val in enumerate(uniq_vals)}

            # 속성 값을 해당 정수 코드로 변환합니다.
            d[attr_name] = [val_dict[val] for val in attr_vals]

    # 문자열 속성 값을 정수로 변환합니다.
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)

    X_data = []

    # 정수 값이 바이너리 값이 아니면 원 핫 인코딩을 수행합니다.
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country":
            X_data.append(np.array(attr_vals).reshape(-1, 1))  # 정수 속성이거나 'native_country'인 경우 그대로 추가합니다.
        else:
            attr_vals, index_dict = get_one_hot_encoding(attr_vals)
            X_data.append(attr_vals)  # 원-핫 인코딩된 속성 값을 추가합니다.

    # X_data의 모든 배열을 수평으로 결합합니다.
    X = np.hstack(X_data)

    # numpy 배열로 변환하여 반환합니다.
    y = np.array(y, dtype=float)
    for k, v in x_control.items():
        x_control[k] = np.array(v, dtype=float)

    # 데이터를 무작위로 섞습니다.
    perm = list(range(len(y)))
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # 데이터를 일부만 샘플링합니다.
    if load_data_size is not None:
        print(f"Loading only {load_data_size} examples from the data")
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control

# 데이터를 로드하고 결과를 출력하는 코드 추가
if __name__ == "__main__":
    X, y, x_control = load_bank_data()  # 데이터 로드 (필요에 따라 사이즈 조정 가능)
    if X is not None and y is not None and x_control is not None:
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Sensitive attribute marital distribution:", np.bincount(x_control['marital'].astype(int)))
