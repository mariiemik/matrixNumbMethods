import numpy as np

def gram_schmidt_qr(A):
    """QR-разложение методом Грамма-Шмидта"""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i].T, A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R

def householder_qr(A):
    """QR-разложение методом Хаусхолдера"""
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(n):
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x)
        v = x - e1
        v = v / np.linalg.norm(v)
        H = np.eye(m)
        H[k:, k:] -= 2.0 * np.outer(v, v)
        R = H @ R
        Q = Q @ H.T
    return Q, R

def givens_qr(A):
    """QR-разложение методом вращений Гивенса"""
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for i in range(n):
        for j in range(m-1, i, -1):
            a, b = R[j-1, i], R[j, i]
            r = np.hypot(a, b)
            c, s = a / r, -b / r
            G = np.eye(m)
            G[[j-1, j], [j-1, j]] = c
            G[j-1, j], G[j, j-1] = s, -s
            R = G @ R
            Q = Q @ G.T
    return Q, R

def test_qr_methods():
    """Тестирование QR-разложения разными методами"""
    A = np.random.rand(4, 4)  # Случайная матрица 4x4
    print("Исходная матрица A:")
    print(A)

    print("\nМетод Грамма-Шмидта:")
    Q_gs, R_gs = gram_schmidt_qr(A)
    print("Q:")
    print(Q_gs)
    print("R:")
    print(R_gs)
    print("Проверка: Q @ R:")
    print(Q_gs @ R_gs)

    print("\nМетод Хаусхолдера:")
    Q_hh, R_hh = householder_qr(A)
    print("Q:")
    print(Q_hh)
    print("R:")
    print(R_hh)
    print("Проверка: Q @ R:")
    print(Q_hh @ R_hh)

    print("\nМетод вращений Гивенса:")
    Q_gv, R_gv = givens_qr(A)
    print("Q:")
    print(Q_gv)
    print("R:")
    print(R_gv)
    print("Проверка: Q @ R:")
    print(Q_gv @ R_gv)

if __name__ == "__main__":
    test_qr_methods()
