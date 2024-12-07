import math
import numpy as np

eps = 1e-6
print("Машинный эпсилон:", eps)


# Ctrl + Alt + L - форматирование файла


def LUmatrix(matrixLU: np.array) -> (np.array, int):
    n = len(matrixLU)
    P = np.eye(n)
    perest = 0
    for j in range(n):  # идем по столбцам
        pivot_val = abs(matrixLU[j][j])  # опорный элемент
        pivot_row = j
        for i in range(j + 1, n):  # идем по строчкам и ищем макс по модулю элемент
            if pivot_val < abs(matrixLU[i][j]):
                pivot_val = abs(matrixLU[i][j])
                pivot_row = i
        print("1: ", pivot_row, " ", pivot_val)
        # pivot_row = np.argmax(matrixLU[j:, j])
        # pivot_val = abs(matrixLU[pivot_row][j])
        # print("2: ", pivot_row, " ", pivot_val)
        # print()
        if pivot_val == 0:
            return "ERROR"

        # обмениваем строки местами
        if pivot_row != j:
            matrixLU[[pivot_row, j]] = matrixLU[[j, pivot_row]]
            P[[pivot_row, j]] = P[[j, pivot_row]]
            perest += 1
        # for i in range(n):
        for i in range(j + 1, n):
            summ = 0
            """if j >= i:
                for k in range(i):
                    summ += matrixLU[i][k] * matrixLU[k][j]
                #matrixLU[i][j] = matrixLU[i][j] - summ
                matrixLU[i][j] = matrix[i][j] - summ"""
            # if j < i:
            # for k in range(j):
            #    summ += matrixLU[i][k] * matrixLU[k][j]
            # matrixLU[i][j] = matrix[i][j] - summ
            # matrixLU[i][j] = matrixLU[i][j] - summ
            matrixLU[i][j] /= matrixLU[j][j]
            # вычитаем из все строки строку выше * число
            if matrixLU[i][j] != 0:
                # b[i] -= matrixLU[i][j] * b[j]
                matrixLU[i, j + 1:] -= matrixLU[i][j] * matrixLU[j, j + 1:]
                """for k in range(j + 1, n):
                    matrixLU[i][k] -= matrixLU[j][k] * matrixLU[i][j]"""
                # matrix[i][j]
    return P, perest


def LUmatrix_mod(matrixLU: np.array) -> (int, np.array, int, np.array):
    n = len(matrixLU)
    per_s = np.eye(n)
    per_c = np.eye(n)
    perest_s = 0
    perest_c = 0
    rank = n
    for j in range(n):  # идем по столбцам
        pivot_val = abs(matrixLU[j][j])  # опорный элемент
        pivot_row = j
        for i in range(j + 1, n):  # идем по строчкам и ищем макс по модулю элемент
            if pivot_val < abs(matrixLU[i][j]):
                pivot_val = abs(matrixLU[i][j])
                pivot_row = i
        if pivot_val <= eps:
            # если закончились ненулевые столбцы, выходим
            rank -= 1
            if rank == j:
                return rank, per_s, perest_s, per_c
            # обмениваем местами столбцы, нулевой идет в конец
            matrixLU[:, [rank, j]] = matrixLU[:, [j, rank]]
            per_c[:, [rank, j]] = per_c[:, [j, rank]]
            perest_c += 1
            # снова ищем макс эл
            pivot_val = abs(matrixLU[j][j])  # опорный элемент
            pivot_row = j
            for i in range(j + 1, n):  # идем по строчкам и ищем макс по модулю элемент
                if pivot_val < abs(matrixLU[i][j]):
                    pivot_val = abs(matrixLU[i][j])
                    pivot_row = i

        # обмениваем строки местами
        if pivot_row != j:
            matrixLU[[pivot_row, j]] = matrixLU[[j, pivot_row]]
            per_s[[pivot_row, j]] = per_s[[j, pivot_row]]
            perest_s += 1
        # for i in range(j+1, n):
        for i in range(j + 1, n):  # дальше не смотрим там нулевые столбцы
            summ = 0
            matrixLU[i][j] /= matrixLU[j][j]
            # вычитаем из все строки строку выше * число
            # if matrixLU[i][j]!= 0:
            matrixLU[i, j + 1:] -= matrixLU[i][j] * matrixLU[j, j + 1:]
    print("FUNC:")
    print(per_c)
    print(matrixLU)
    return rank, per_s, perest_s, per_c

def LUP(matrix):
    n = len(matrix)
    matrixLU = np.copy(matrix)
    # Pb = np.copy(b)
    rank, per_s, perest_str, per_c = LUmatrix_mod(matrixLU)

    # PA = np.dot(per_s, matrix)
    # print("P: ", PA)
    # print("PC:", per_c)
    U = np.triu(matrixLU)
    L = np.tril(matrixLU, -1)
    np.fill_diagonal(L, 1)

    return rank, per_s, perest_str, per_c, L, U

""" for i in range(len(matrix)):
        for j in range(len(matrix)):
            summ = 0
            if i > j:  # заполняем нижнюю матрицу
                for k in range(j):
                    summ += matrixLU[i][k] * matrixLU[k][j]
                matrixLU[i][j] = 1 / matrixLU[j][j] * (matrix[i][j] - summ)
            else:
                for k in range(i):
                    summ += matrixLU[i][k] * matrixLU[k][j]
                matrixLU[i][j] = matrix[i][j] - summ"""


def slau(left: np.array, right: np.array, b: np.array) -> np.array:
    n = len(left)
    x_v = np.zeros(n, dtype=float)
    y_v = np.zeros(n, dtype=float)
    # find y
    for i in range(n):
        sum_ly = 0
        for k in range(i):
            sum_ly += left[i][k] * y_v[k]
        y_v[i] = b[i] - sum_ly

    # находим х
    for i in range(n - 1, -1, -1):
        sum_ux = 0
        for k in range(i + 1, n):
            sum_ux += right[i][k] * x_v[k]
        x_v[i] = 1 / right[i][i] * (y_v[i] - sum_ux)

    return x_v


def slauQR(Q: np.array, R: np.array, b: np.array) -> np.array:
    """QRx = b
        => Rx = transp(Q) * b"""
    b_new = np.dot(np.transpose(Q), b)  # transp(Q) * b

    n = len(Q)
    x_v = np.zeros(n, dtype=float)
    # находим х
    for i in range(n - 1, -1, -1):
        sum_ux = 0
        for k in range(i + 1, n):
            sum_ux += R[i][k] * x_v[k]
        x_v[i] = 1 / R[i][i] * (b_new[i] - sum_ux)

    return x_v


def norm(a: np.array) -> int:
    # берем бесконечную норму
    # максимум суммы строк
    return np.max([np.sum(np.abs(row)) for row in a])


def sovmest(lower: np.array, rank: int, b: np.array):  # передаем сюда b с перестановками
    # нужно преобразовать вектор b, вычесть нужные строчки
    for j in range(rank):
        for i in range(j + 1, len(b)):
            b[i] -= b[j] * lower[i][j]
    print("modified b: ", b)

    non_zero_elements = b[np.abs(b) > eps]
    return len(non_zero_elements) == rank


def chast_sol(upper: np.array, b: np.array, rank: int):
    # идем с конца к верху
    n = len(b)
    sol = np.zeros(n, dtype=float)

    # sol[rank-1+1:] = 0
    # sol[rank-1] = b[rank-1]/upper[rank-1][rank-1]
    for i in range(rank - 1, -1, -1):
        sum = 0
        for k in range(i + 1, rank):
            sum += upper[i][k] * sol[k]
        sol[i] = (b[i] - sum) / upper[i][i]

    #        non_zero_elements = upper[i][upper[i] != 0]
    #       if i == rank - 1:
    return sol


def Jacobi(A: np.array, b: np.array):
    B = np.array((-A + np.diag(np.diag(A))) / np.diag(A)[:, np.newaxis])
    c = b / np.diag(A)[:, np.newaxis]

    x_i = c.copy()
    x_i_plus_one = np.dot(B, x_i) + c

    B_norm = np.linalg.norm(B)
    kol = 0
    while B_norm / (1 - B_norm) * np.linalg.norm(x_i_plus_one - x_i) > eps:
        x_i = x_i_plus_one.copy()
        x_i_plus_one = np.dot(B, x_i) + c
        kol+=1
    print("Количество итераций = ", kol)
    return x_i_plus_one

    # x = c.copy()
    # for i in range(n):
    #     x[i] = sum(x * B[i, :]) + c[i]

def Zeidel(A: np.array, b: np.array):
    B = np.array((-A + np.diag(np.diag(A))) / np.diag(A)[:, np.newaxis])
    c = b / np.diag(A)[:, np.newaxis]

    x_i_plus_one = c.copy()

    B_norm = np.linalg.norm(B)
    print("norm = ", B_norm)
    kol = 0
    while True:
        x_i = x_i_plus_one.copy()
        for i in range(len(A)):
            x_i_plus_one[i] = sum( np.dot(B[i, :], x_i_plus_one)) + c[i]
        kol += 1
        q = B_norm / (1 - B_norm) if B_norm < 1 else 1e2
        if q * np.linalg.norm(x_i_plus_one - x_i) < eps or kol > 100000:
            print("Количество итераций = ", kol)
            return x_i_plus_one

    """ while B_norm / (1 - B_norm) * norm(x_i_plus_one - x_i) > eps:
        x_i = x_i_plus_one.copy()
       # x_i_plus_one = np.dot(B, x_i) + c
        print("x_i = ", x_i_plus_one)
        for i in range(len(A)):
            print(x_i_plus_one[i],'\n', B[i, :])
            print()
            x_i_plus_one[i] = sum(x_i_plus_one[i] * B[i, :]) + c[i]
"""
    #return x_i_plus_one



if __name__ == '__main__':

    point = 2
    """ 2 - ищем решение или частное решение с помощью LU 
        3 - QR разложение, метод отражений 
        4 - Метод Якоби """

    if point == 2:
        matrix = np.array([[2, 3, -1, 1],
                                              [8, 12, -9, 8],
                                              [4, 6, 3, -2],
                                              [2, 3, 9, -7]
                                              ], dtype=float)
        n = 4
        matrix = np.array(np.random.randint(low=1, high=10, size=(n, n)), dtype=float)
        rank,per_s, perest_str, per_c, L, U  = LUP(matrix)
        # matrix = np.array([[2, 3, -1, 1],
        #                    [8, 12, -9, 8],
        #                    [4, 6, 3, -2],
        #                    [2, 3, 9, -7]
        #                    ], dtype=float)
        # # b = np.array(np.random.randint(low=1, high=10, size=(n, 1)),dtype=float)
        b = np.array([1, 3, 3, 3])
        # # b = np.array( np.random.randint(1,10,size=n), dtype= float)
        # print("original matrix: \n", matrix)
        #
        # matrixLU = np.copy(matrix)
        # # Pb = np.copy(b)
        # rank, per_s, perest_str, per_c = LUmatrix_mod(matrixLU)
        #
        PA = np.dot(per_s, matrix)
        # print("P: ", PA)
        # print("PC:", per_c)
        # U = np.triu(matrixLU)
        # L = np.tril(matrixLU, -1)
        # np.fill_diagonal(L, 1)
        # LU = np.dot(L, U)
        # print("L: ", L)
        # print("U: ", U)

        if rank == n:
            # ищем определитель
            # det L = 1, надо найти определитель U
            diagonal_elements = np.diag(U)
            determinant = np.prod(diagonal_elements) * (1 if perest_str % 2 else -1)
            print("determinant = ", determinant)

            # Ax=b
            Pb = np.dot(per_s, b)
            x_v = slau(L, U, Pb)
            ax = np.dot(matrix, x_v)
            print("x_v: ", x_v)
            print("ax:", ax)
            print("b: ", b)
            print("ax-b= ", np.subtract(ax, b))

            # обратная матрица
            inverted_matrix = np.dot(np.linalg.inv(U),
                                     np.linalg.inv(L))  # обратная матрица матрицы с поменяными строчками
            print("inv:", inverted_matrix)
            a_dot_inv_a = np.dot(PA, inverted_matrix)
            inv_a_dot_a = np.dot(inverted_matrix, PA)
            print(inv_a_dot_a)
            print(a_dot_inv_a)

            # число обусловленности
            ob = norm(PA)
            ob_inv = norm(inverted_matrix)
            obuslov = ob * ob_inv
            print("obuslov = ", obuslov)
        else:
            print("Система вырожденная")
            print("rank = ", rank)
            Pb = np.dot(per_s, b)
            #  после вызова функции Pb изменится
            if sovmest(L, rank, Pb):
                print("Система совместная")
                sol = np.dot(per_c, chast_sol(U, Pb, rank))
                print("solution: ", sol)
                print(np.dot(matrix, sol))
                print(b)
            else:
                print("Система несовместная")
    elif point == 3:
        """A = np.array([[1, 2, 3, 4], [4, 6, 2, 8], [5, 7, 3, 9], [6, 8, 1, 4]], dtype = np.float64)
        b = np.array([1, 2, 3, 4], dtype=float)"""
        n = 4
        # A = np.array([[3, 1, 1], [1, 5, 1], [1, 1, 7]], dtype=float)
        A = np.array(np.random.randint(low=1, high=10, size=(n, n)), dtype=float)

        b = np.array([5, 7, 9, 1], dtype=float)
        R = A.copy()  # верхнетреугольная матрица
        print("original matrix:\n", A)
        print("determinant = ", np.linalg.det(A))
        Q = np.eye(len(A),dtype=float) #: np.array

        for k in range(len(A) - 1):  # проходимся по все столбцам кроме последнего, чтобы обнулить
            s = R[:, k].copy()  # обнуляем k столбец
            s[0: k] = 0
            s[k] = s[k] - np.linalg.norm(s)
            n = s / np.linalg.norm(s)  # искомый вектор нормали

            H = np.eye(len(A), dtype=float) - np.dot(n[:, np.newaxis], n[np.newaxis, :]) * 2  # матрица отражения
            R = np.dot(H, R)
            print("Q", k, " : ", H)

            Q = np.dot(Q, H)  #  каждую Q надо запоминать на каждом шаге
            # print("R",k," : ", R)

        print("R = \n", R)
        # QR = A => Q = AR^(-1)
        # Q = np.dot(A, np.linalg.inv(R))
        print("Q = \n", Q)
        # Q = np.dot(A, np.linalg.inv(R))
        # print("Q = \n", Q)
        print("QR = \n", (np.dot(Q, R)))
        x = slauQR(Q, R, b)
        print("x = ", x)
        print("Ax = ", np.dot(A, x))
        print("b = ", b)
    elif point == 4:
        A = np.array([[10, 2, 3], [5, 60, 7], [8, 9, 110]], dtype=float)
        b = np.array([[1], [1], [1]], dtype=float)
        solveJ = Jacobi(A, b)
        print("Jacobi=", solveJ)
        print(np.linalg.solve(A,b) - solveJ)
        print()
        solveZ = Zeidel(A, b)
        print("Zeid=",solveZ)
        print(np.dot(A, solveZ) - b)


        print("\n\nположительно определенная:")
        A = np.array([[1, 2, 3],
                      [2, 4, 5],
                      [3, 5, 6]], dtype=float)
        A = A @ A
        b = np.array([[1], [1], [1]], dtype=float)
        solveJ = Jacobi(A, b)
        print("Jacobi=", solveJ)
        print(np.dot(A, solveJ) - b)
        print()
        solveZ = Zeidel(A, b)
        print("Zeid=", solveZ)
        print(np.dot(A, solveZ) - b)

