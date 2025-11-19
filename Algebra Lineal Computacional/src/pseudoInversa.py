import src.alc as alc
import numpy as np

# A * Apinv * A = A
def naive_condicion1(X, pX, tol):
    mul = alc.dot(X, alc.dot(pX, X))
    
    for fila in range(X.shape[0]):
        for col in range(X.shape[1]):
            if abs(mul[fila, col] - X[fila, col]) > tol:
                return False
    return True

# Apinv * A * Apinv = Apinv
def naive_condicion2(X, pX, tol):
    mul = alc.dot(pX, alc.dot(X, pX))
    
    for fila in range(X.shape[0]):
        for col in range(X.shape[1]):
            if abs(mul[fila, col] - pX[fila, col]) > tol:
                return False
    return True

# A * Apinv simetrica
def naive_condicion3(X, pX, tol):
    return alc.esSimetrica(alc.dot(X, pX), tol)

# Apinv * A simetrica
def naive_condicion4(X, pX, tol):
    return alc.esSimetrica(alc.dot(pX, X), tol)

# recibe dos matrices y devuelve True si verifican las condiciones de Moore-Penrose
def esPseudoInversaNaive(X, pX, tol=1e-8):
    return naive_condicion1(X,pX, tol) and naive_condicion2(X,pX, tol) and naive_condicion3(X,pX, tol) and naive_condicion4(X,pX, tol)



# version no naive =================================================================================================================

# recibe dos matrices y devuelve True si verifican las condiciones de Moore-Penrose
def esPseudoInversa(X, pX, tol=1e-8):
    # U, S, V = alc.svd_completa(X)
    # _, Spinv, _ = alc.svd_completa(pX)
    U, S, V = np.linalg.svd(X)
    _, Spinv, _ = np.linalg.svd(pX)

    S_matriz = np.zeros((len(S), len(S)))
    for i in range(len(S_matriz)):
        S_matriz[i][i] = S[i]
    
    Spinv_matriz = np.zeros((len(S), len(S)))
    for i in range(len(S_matriz)):
        if S_matriz[i][i] == 0:
            continue
        Spinv_matriz[i][i] = 1/S_matriz[i][i]
        

    sub_mul1 = alc.dot(U, alc.dot(S_matriz, Spinv_matriz)) # U * S * Spinv
    
    # condicion 1: A * Apinv * A = A
    mul = alc.dot(sub_mul1, alc.dot(S_matriz, alc.traspuesta(V)))

    for fila in range(X.shape[0]):
        for col in range(X.shape[1]):
            if abs(mul[fila, col] - X[fila, col]) > tol:
                return False
    
    # condicion 2: A * Apinv simetrica
    if not alc.esSimetrica(alc.dot(sub_mul1, alc.traspuesta(U)), tol):
        return False

    # condicion 3: Apinv * A * Apinv = Apinv
    sub_mul1 = alc.dot(V, alc.dot(Spinv_matriz, S_matriz))
    mul = alc.dot(sub_mul1, alc.dot(Spinv_matriz, alc.traspuesta(U)))

    for fila in range(X.shape[0]):
        for col in range(X.shape[1]):
            if abs(mul[fila, col] - X[fila, col]) > tol:
                return False
    
    # condicion 4: Apinv * A simetrica
    if not alc.esSimetrica(alc.dot(sub_mul1, alc.traspuesta(V)), tol):
        return False

    return True
