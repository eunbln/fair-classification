import numpy as np

def squared_loss(w, X, y, return_arr=False):
    y_pred = np.dot(X, w)
    loss = (y - y_pred) ** 2
    if return_arr:
        return loss
    else:
        return np.mean(loss)

def logistic_loss(w, X, y, return_arr=False):
    z = np.dot(X, w)
    yz = y * z
    loss = np.logaddexp(0, -yz)
    if return_arr:
        return loss
    else:
        return np.mean(loss)

if __name__ == "__main__":
    # 예제 데이터 생성
    X = np.array([[0.5, 1.5], [1, 2], [1.5, 0.5], [1, 1]])
    y = np.array([1, 0, 1, 1])
    w = np.array([0.2, 0.4])
    
    # 손실 함수 계산
    sq_loss = squared_loss(w, X, y)
    log_loss = logistic_loss(w, X, y)
    
    # 결과 출력
    print("Squared Loss:", sq_loss)
    print("Logistic Loss:", log_loss)
