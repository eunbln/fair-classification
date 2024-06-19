import os, sys
import urllib.request  # Python 3에서는 urllib2 대신 urllib.request를 사용합니다.
sys.path.insert(0, '../../fair_classification/')  # 공정 분류 코드가 이 디렉토리에 있습니다.
import utils as ut
import numpy as np
from random import seed, shuffle

# 랜덤 시드 설정: 무작위 순열을 다시 생성할 수 있도록 합니다.
SEED = 1122334455
seed(SEED)
np.random.seed(SEED)

"""
    성인 데이터셋은 http://archive.ics.uci.edu/ml/datasets/Adult 에서 얻을 수 있습니다.
    코드는 현재 디렉토리에서 데이터 파일(adult.data, adult.test)을 찾고,
    발견되지 않으면 UCI 아카이브에서 다운로드합니다.
	https://archive.ics.uci.edu/dataset/222/bank+marketing
"""

def check_data_file(fname):
    files = os.listdir(".")  # 현재 디렉토리 목록 가져오기
    print("Looking for file '%s' in the current directory..." % fname)

    # 파일이 현재 디렉토리에 없으면 UCI 아카이브에서 다운로드합니다.
    if fname not in files:
        print("'%s' not found! Downloading from UCI Archive..." % fname)
        addr = "https://archive.ics.uci.edu/dataset/222/bank+marketing%s" % fname
        response = urllib.request.urlopen(addr)
        data = response.read()
        # 바이너리 모드로 파일을 열어 데이터를 씁니다.
        with open(fname, "wb") as fileOut:
            fileOut.write(data)
        print("'%s' downloaded and saved locally.." % fname)
    else:
        print("File found in current directory..")
    
    print()
    return

def load_adult_data(load_data_size=None):
    """
        load_data_size가 None으로 설정되면 전체 데이터를 로드하고 반환합니다.
        load_data_size가 숫자이면, 예를 들어 10000, 무작위로 선택된 10,000개의 예를 반환합니다.
    """

    # 모든 속성 리스트
    attrs = ['age', 'job', 'marital', 'education', 'default', 'balance', 
             'housing', 'loan', 'contact', 'day', 'month', 'duration', 
			 'campaign', 'pdays', 'previous', 'poutcome', 'y']

    # 정수 값을 가지는 속성 리스트
    int_attrs = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

    # 공정성 제약 조건에 사용될 민감한 속성
    sensitive_attrs = ['housing']

    # 분류에서 무시할 속성: 민감한 속성 및 외부에서 계산된 속성
    attrs_to_ignore = ['housing', 'education']

    # 분류에 사용할 속성
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    # 성인 데이터는 훈련용과 테스트용 두 개의 다른 파일로 제공됩니다.
    data_files = ["bank_marketing.data", "bank_marketing.test"]

    X = []  # 입력 데이터
    y = []  # 출력 데이터 (레이블)
    x_control = {}  # 민감한 속성 데이터

    # 각 속성에 대한 값을 저장할 딕셔너리 초기화
    attrs_to_vals = {}
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    # 데이터 파일 확인 및 다운로드
    for f in data_files:
        check_data_file(f)

        # 파일에서 데이터를 읽고 전처리
        for line in open(f):
            line = line.strip()
            if line == "": continue  # 빈 줄 건너뛰기
            line = line.split(", ")
            if len(line) != 15 or "?" in line:  # 속성이 누락된 줄 건너뛰기
                continue

            # 클래스 레이블 처리
            class_label = line[-1]
            if class_label in ["no.", "no"]:
                class_label = -1
            elif class_label in ["yes.", "yes"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")
            y.append(class_label)

            # 각 속성 값 처리 및 저장
            for i in range(0, len(line) - 1):
                attr_name = attrs[i]
                attr_val = line[i]
                # 매우 희소한 특징의 차원 줄이기
                #if attr_name == "native_country":
                #    if attr_val != "United-States":
                #        attr_val = "Non-United-Stated"
                #elif attr_name == "education":
                #    if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                #        attr_val = "prim-middle-school"
                #    elif attr_val in ["9th", "10th", "11th", "12th"]:
                #        attr_val = "high-school"

                # 민감한 속성 처리
                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)

    def convert_attrs_to_ints(d):
        """
            문자열 속성을 이산화하여 정수로 변환합니다.
        """
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs: continue
            uniq_vals = sorted(list(set(attr_vals)))  # 고유 값 얻기

            # 고유 값에 대한 정수 코드 계산
            val_dict = {}
            for i in range(0, len(uniq_vals)):
                val_dict[uniq_vals[i]] = i

            # 값을 정수 인코딩으로 대체
            for i in range(0, len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    # 이산 값을 정수 표현으로 변환
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)

    # 정수 값이 이진수가 아닌 경우, 원-핫 인코딩을 적용해야 합니다.
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country":  # native_country는 이진수로 인코딩되었습니다.
            X.append(attr_vals)
        else:
            attr_vals, index_dict = ut.get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:
                X.append(inner_col)

    # numpy 배열로 변환하여 쉽게 처리
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype=float)
    for k, v in x_control.items(): 
        x_control[k] = np.array(v, dtype=float)

    # 데이터를 섞기
    perm = list(range(0, len(y)))  # 각 fold를 만들기 전에 데이터를 섞습니다.
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # 데이터의 일부만 로드해야 하는지 확인
    if load_data_size is not None:
        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control
