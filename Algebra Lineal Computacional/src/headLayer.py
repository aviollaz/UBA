import numpy as np

import src.alc as alc

## Cholesky

def fully_connected_lineal_cholesky(
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    Calculo de matriz W estimada usando el algoritmo por Cholesky

    Parameters
    ----------
    - X: np.ndarray
        Datos / Embeddings de las imágenes
    - Y: np.ndarray
        Clasificaciones esperadas

    Returns
    -------
    - W: np.ndarray
        Matríz W óptima
    """
    n,p = X.shape 
    m,p1 = Y.shape
    if p != p1:
        raise ValueError("Las matrices X e Y deben tener la misma cantidad de columnas.")

    rango = np.linalg.matrix_rank(X)
    if rango == p and n > p:
        X_t = alc.traspuesta(X)
        A = alc.dot(X_t,X)
        L = alc.calculaCholesky(A) #Diagonal inferior 
        print("* Cholesky factor L calculated")
        U = pinvEcuacionesNormales(X, L, Y)
        return alc.dot(Y, U)

    if rango == n and p > n:
        X_t = alc.traspuesta(X)
        A = alc.dot(X,X_t)
        L = alc.calculaCholesky(A)
        print("* Cholesky factor L calculated")
        V_t = pinvEcuacionesNormales(X, L, Y)
        V = alc.traspuesta(V_t)
        W = alc.dot(Y, V)
        return W
       
    if rango == n and rango == p:
        #Objetivo : resolver X^t * W^t = Y^t, para ello hacemos LU de X^t. LU * W^t = Y^t, para resolver esto hacemos dos sustituciones
        # L * S = Y^t  -> S
        # U * W^t = S  -> W^t
        X_t = alc.traspuesta(X)
        L, U = alc.calculaLU(X_t)              
        S   = alc.sustitucion_adelante_multiple(L, alc.traspuesta(Y)) # L * S = Y , con S como incognita
        W_t = alc.sustitucion_atras_multiple(U, S)    # U * W_t = S , con W_t como incognita
        W   = alc.traspuesta(W_t)
        return W
    
def pinvEcuacionesNormales(X, L, Y):
    Z = alc.sustitucion_adelante_multiple(L, alc.traspuesta(X))
    U = alc.sustitucion_atras_multiple(alc.traspuesta(L), Z)
    return U

## SVD
def pinvSVD(U, S, V, Y):   
    Spinv = np.zeros((len(S), len(S)))

    for i in range(len(S)):
        if S[i] != 0:
            Spinv[i, i] = 1.0/S[i]
    VsigmaUt = alc.dot(V, alc.dot(Spinv, alc.traspuesta(U)))

    Y_reducida = Y[:, :VsigmaUt.shape[0]]
    W = alc.dot(Y_reducida, VsigmaUt)

    return W

def fully_connected_lineal_svd(
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    Calculo de matriz W estimada usando el algoritmo por SVD

    Parameters
    ----------
    - X: np.ndarray
        Datos / Embeddings de las imágenes
    - Y: np.ndarray
        Clasificaciones esperadas

    Returns
    -------
    - W: np.ndarray
        Matríz W óptima
    """

    # U,S,V = alc.svd_reducida(X)
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    V = alc.traspuesta(Vt)

    W = pinvSVD(U,S,V,Y)

    print("* Finished SVD method")
    return W

## QR

def fully_connected_lineal_qr(
    X: np.ndarray,
    Y: np.ndarray,
    algorithm: str = 'hh'
) -> np.ndarray:
    """
    Calculo de matriz W estimada usando el algoritmo por QR

    Parameters
    ----------
    - X: np.ndarray
        Datos / Embeddings de las imágenes
    - Y: np.ndarray
        Clasificaciones esperadas
    - algorithm: str
        gs | hh según si QR se calcula con Gram Schmidt o House Holder

    Returns
    -------
    - W: np.ndarray
        Matríz W óptima
    """
    if algorithm not in ["gs", "hh"]:
        raise Exception("Invalid QR algorithm")
    
    if algorithm == 'gs':
        Q, R, _ = alc.QR_con_GS_reducido(alc.traspuesta(X))
    else:
        Q, R = alc.QR_con_HH(alc.traspuesta(X))

    if Q is None or R is None:
        raise Exception("Could not compute QR")

    Qt = alc.traspuesta(Q)
    Vt = alc.sustitucion_atras_multiple(R, Qt)
    V = alc.traspuesta(Vt)
    W = alc.dot(Y, V)
    
    print("* Finished QR method with", algorithm)
    return W
