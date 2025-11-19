import math
import numpy as np
#FUNCIONES AUXILIARES


def dot(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Escalar · escalar
    if u.ndim == 0 and v.ndim == 0:
        return (u * v).item()

    # Mat/vec · escalar  ||  escalar · mat/vec  -> multiplicación escalar
    if v.ndim == 0 and u.ndim in (1, 2):
        return u * v
    if u.ndim == 0 and v.ndim in (1, 2):
        return u * v

    # Vec · vec (producto interno)
    if u.ndim == 1 and v.ndim == 1:
        if u.shape[0] != v.shape[0]:
            raise ValueError(f"dot: tamaños incompatibles {u.shape} · {v.shape}")
        return float(np.sum(u * v))

    # Mat · vec
    if u.ndim == 2 and v.ndim == 1:
        if u.shape[1] != v.shape[0]:
            raise ValueError(f"dot: tamaños incompatibles {u.shape} · {v.shape}")
        return np.array([np.sum(u[i, :] * v) for i in range(u.shape[0])])

    # Vec · Mat (interpreto vec como fila)
    if u.ndim == 1 and v.ndim == 2:
        if u.shape[0] != v.shape[0]:
            raise ValueError(f"dot: tamaños incompatibles {u.shape} · {v.shape}")
        # resultado es (v.shape[1],)
        return np.array([np.sum(u * v[:, j]) for j in range(v.shape[1])])

    # Mat · Mat
    if u.ndim == 2 and v.ndim == 2:
        if u.shape[1] != v.shape[0]:
            raise ValueError(f"dot: tamaños incompatibles {u.shape} · {v.shape}")
        m, n, p = u.shape[0], u.shape[1], v.shape[1]
        out = np.empty((m, p))
        for i in range(m):
            for j in range(p):
                out[i, j] = np.sum(u[i, :] * v[:, j])
        return out

    raise TypeError(f"dot: no soporta ndim u={u.ndim}, v={v.ndim}")

def idN(n):
    I = np.zeros((n, n), dtype=float)
    for i in range(n):
        I[i, i] = 1.0
    return I

def e1(p):
    e = np.zeros(p, dtype=float)
    e[0] = 1.0
    return e

def outer(u, v):
    return u[:, None] * v[None, :]


def sustitucion_adelante_multiple(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Resuelve el sistema L·V = B, donde L es una matriz triangular inferior
    de tamaño (n x n) y B es una matriz de tamaño (n x p) o un vector de tamaño (n,).
    Retorna la matriz V de tamaño (n x p) que satisface la ecuación.
    """
    
    L = np.asarray(L)
    B = np.asarray(B)

    n = L.shape[0]
    if L.shape[1] != n:
        raise ValueError("L debe ser cuadrada (n×n)")

    # Normalizar B a (n, p)
    if B.ndim == 1:
        B = B.reshape(n, 1)
    elif B.shape[0] == n:
        pass  # ya está (n, p)
    elif B.shape[1] == n:
        B = traspuesta(B)  # venía (p, n) -> (n, p)
    else:
        raise ValueError(f"B debe ser (n,p) o (p,n) con n={n}, pero es {B.shape}")
    
    n,p = B.shape  

    V = np.zeros((n, p)) 

    for i in range(n):
        # producto L[i, :i] (1 x i) con V[:i, :] (i x p) -> (1 x p)
        suma = dot(L[i, :i], V[:i, :])
        V[i, :] = (B[i, :] - suma) / L[i, i]

    return V

def sustitucion_atras_multiple(U: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Resuelve el sistema U·V = B, donde U es una matriz triangular superior
    de tamaño (n x n) y B es una matriz de tamaño (n x p) o un vector (n,).
    Retorna la matriz V (n x p) tal que U V = B.
    """

    U = np.asarray(U)
    B = np.asarray(B)

    n = U.shape[0]
    if U.shape[1] != n:
        raise ValueError("U debe ser cuadrada (n×n)")

    # Normalizar B a (n, p)
    if B.ndim == 1:
        B = B.reshape(n, 1)
    elif B.shape[0] == n:
        pass  # ya está (n, p)
    elif B.shape[1] == n:
        B = traspuesta(B)  # venía (p, n) -> (n, p)
    else:
        raise ValueError(f"B debe ser (n,p) o (p,n) con n={n}, pero es {B.shape}")

    n, p = B.shape
    
    V = np.zeros((n, p))

    # Recorremos de abajo hacia arriba
    for i in range(n-1, -1, -1):

        # Producto U[i, i+1:] (1 x k) con V[i+1:, :] (k x p)
        suma = dot(U[i, i+1:], V[i+1:, :])
        
        V[i, :] = (B[i, :] - suma) / U[i, i]

    return V


def calcular_rango(X):
    _,U = calculaLU(X)
    rango = 0
    for i in range(len(U)):
        if abs(U[i][i]) < 1e-10:
            continue
        else:
            rango += 1
    return rango

#LABO 1 
def error(x,y):
    return (abs(x-y))
    
def error_relativo(x,y):
    return((abs(x-y))/abs(x))

def matricesIguales(A,B):
    if A.shape != B.shape:
        return False
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if abs(A[i, j] - B[i, j]) > 1e-5:
                return False
    return True 

def traspuesta(A):
    n, m = A.shape
    T = np.zeros((m, n), dtype=A.dtype)
    for i in range(n):
        T[:, i] = A[i, :]
    return T

def esSimetrica(A):
    matrizTranspuesta = traspuesta(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] - matrizTranspuesta[i, j] > 1e-5:
                return False
    return True

#LABO 2 

def rota(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


def escala(s):
    """
    Recibe una tira de números s y retorna una matriz cuadrada n×n,
    donde n es el tamaño de s.
    La matriz escala la componente i de un vector de R^n
    en un factor s[i].
    """
    size = len(s)
    matriz_escalado = np.zeros((size, size))
    for i in range(size):
        matriz_escalado[i, i] = s[i]
    return matriz_escalado


def rota_y_escala(theta, s):
    """
    Recibe un ángulo theta y una tira de números s (en R^2),
    y retorna una matriz 2x2 que rota el vector en un ángulo theta
    y luego lo escala en un factor s.
    """
    matriz_rotacion = rota(theta)
    matriz_escalado = escala(s)
    return dot(matriz_escalado, matriz_rotacion)


def afin(theta, s, b):
    """
    Recibe un ángulo theta, una tira de números s (en R^2),
    y un vector b en R^2.
    Retorna una matriz 3x3 que rota el vector en un ángulo theta,
    luego lo escala en un factor s y por último lo traslada en b.
    """
    matrizRotadaYEscalada = rota_y_escala(theta, s)
    return np.array([
        [matrizRotadaYEscalada[0, 0], matrizRotadaYEscalada[0, 1], b[0]],
        [matrizRotadaYEscalada[1, 0], matrizRotadaYEscalada[1, 1], b[1]],
        [0, 0, 1]
    ])


def trans_afin(v, theta, s, b):
    """
    Recibe un vector v (en R^2), un ángulo theta,
    una tira de números s (en R^2), y un vector b en R^2.
    Retorna el vector w resultante de aplicar la transformación afín a v.
    """
    matrizAfin = afin(theta, s, b)
    vectorHomogeneo = np.array([v[0], v[1], 1])
    vectorTransformado = matrizAfin @ vectorHomogeneo
    return vectorTransformado[0:2]

#LABO 3

def norma(x,p):
    result =0
    if p == float('inf') or p == 'inf':
        return max(abs(xi) for xi in x) 
    for i in range(len(x)):
        result += abs(x[i])**p # 2 operaciones 
    return (result ** (1/p)) # 2 operaciones 
def norma2(x):
    # Flops: n mult + (n-1) sumas = 2n-1   (+1 sqrt si la contás)
    s = 0.0
    for xi in x:
        s += xi * xi
    return s ** 0.5  # sqrt (contarla aparte si querés)
  
def normaliza(X,p):
    result=[]
    for x in X:
        n = norma(x,p)
        y = [xi / n for xi in x]             # nueva lista; no divide listas
        result.append(y)    
    return result


def normaMatMC(A,q,p,Np):
    maximo = -float('inf')
    vector = None
    n = A.shape[1]
    rng = np.random.default_rng(None)
    X = []
    for _ in range(Np):
        x = rng.normal(size=n)          # o rng.uniform(-1,1,size=n)
        X.append(x)
    Y = normaliza(X, p)
    for y in Y:
        Ay = A @ y
        normaAy = norma(Ay, q)
        if normaAy > maximo:
            maximo = normaAy
            vector = y
    return maximo, vector

def normaExacta(A, p=[1, float('inf')]):
    resultado = None

    m, n = A.shape  # m: filas, n: columnas
    if p == 1:
        # ||A||_1 = max suma por COLUMNA
        mayor1 = -float('inf')
        for j in range(n):
            sumaColumna = 0.0
            for i in range(m):
                sumaColumna += abs(A[i, j])
            if sumaColumna > mayor1:
                mayor1 = sumaColumna
        resultado = mayor1
    if p == float('inf') or p == 'inf':
        # ||A||_∞ = max suma por FILA
        mayorInf = -float('inf')
        for i in range(m):
            sumaFila = 0.0
            for j in range(n):
                sumaFila += abs(A[i, j])
            if sumaFila > mayorInf:
                mayorInf = sumaFila
        resultado = mayorInf

    return resultado

def condMC(A, p, Np=10000):
    valA, _  = normaMatMC(A, p, p, Np)
    valInv, _ = normaMatMC(inversa(A), p, p, Np)
    return valA * valInv


def condExacto(A, p):
    valA = normaExacta(A, p)
    valInv = normaExacta(inversa(A), p)
    return valA * valInv

#LABO 4

def lyb(L,b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum((L[i,j] * y[j]) for j in range(i))/ L[i,i]
    return y

def uxy(U,y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = (y[i] - sum((U[i,j] * x[j]) for j in range(i+1,n))) / U[i][i]
    return x

def calculaLU(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return None

    for i in range(m-1): # columnas
        p = Ac[i][i]
        if p == 0:
            return None
        for j in range(i+1, n): # filas
            c = Ac[j][i] / p
            Ac[j][i:n] = Ac[j][i:n] - c*Ac[i][i:n]
            Ac[j][i] = c
            cant_op = cant_op + (n+1-i) * 2
        
        

    L = Ac.copy()
    U = Ac.copy()
    ## armo las matrices L y U
    for i in range(m):
        for j in range(m):
            if i<j:
                L[i][j]= 0
            elif i == j :
                L[i][j] = 1
            else:
                U[i][j] = 0
                
    
    return L, U, cant_op

def resolve_triangular(L,b,inferior = True):
    """
    Resuelve una matríz triangular
    """
    n = L.shape[0]
    res = np.zeros(n)
    if inferior:
        for i in range(n):
            res[i] = (b[i] - sum((L[i,j] * res[j]) for j in range(i))) / L[i][i]
    else:
        for i in range(n-1,-1,-1):
            res[i] = (b[i] - sum((L[i,j] * res[j]) for j in range(i+1,n))) / L[i][i]
    return res

def inversa(A):
    n = A.shape[0]
    I = np.eye(n)
    L,U,num = calculaLU(A)
    Inv = np.zeros((n,n))
    for j in range(n):
        e = I[:,j]
        y = resolve_triangular(L,e,inferior=True)
        x = resolve_triangular(U,y,inferior=False)
        Inv[:,j] = x
    return Inv


def calculaLDV(A):
    L, U, _ = calculaLU(A)
    n = len(U)
    # Matrices D y V llenas de ceros
    D = np.zeros((n, n))
    V = np.zeros((n, n))

    for i in range(n):
        di = U[i][i]
        D[i][i] = di
        # V = D^{-1} * U  → dividir la fila i de U por U[i][i]
        for j in range(n):
            V[i][j] = U[i][j] / di

    return L,D,V

def esSimetrica(A, atol):
    n = A.shape[0]
    m= A.shape[1]
    if m!= n: return False
    for i in range(n):
        for j in range(i+1, n):
            if abs(A[i][j] - A[j][i]) > atol:
                return False
    return True

def esSDP(A,atol=1e-10):
    if not(esSimetrica(A,atol)):
        return False
    _,D,_ = calculaLDV(A)
    for i in range(A.shape[0]):
        if D[i][i] <= atol:
            return False
    return True

def calculaCholesky(A, tol = 1e-10):
    if not(esSDP(A,tol)):
        raise ValueError("La matriz no es SDP")
    n = A.shape[0]
    L, D, V = calculaLDV(A)
    R_chol = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (D[j][j] > tol):
                R_chol[i][j] = L[i][j] * np.sqrt(D[j][j])
    return R_chol

#LABO 5

def QR_con_GS_reducido(A, tol=1e-12, retorna_nops=False):
    p, n = A.shape 
    
    # Inicializar Q (p x n) y R (n x n)
    Q = np.zeros((p, n))
    R = np.zeros((n, n))
    cant_op = 0
    
    # Primer paso (j=0)
    a1 = A[:, 0]
    r11 = norma2(a1) 
    cant_op += 2*p - 1 # Usar 'p' para operaciones de vector de tamaño p
    R[0][0] = r11
    if r11 < tol:
        return None, None, 0
        
    Q[:, 0] = a1 / R[0][0]
    cant_op += p # Usar 'p' para operaciones de vector de tamaño p
    
    # Bucle principal (j desde 1 hasta n-1)
    for j in range(1, n):
        aj = A[:, j]
        qj_moño = aj
        for k in range(j):
            qk = Q[:, k]
            # Usar 'p' para las operaciones de producto punto (dot)
            rkj = dot(qk, qj_moño)
            cant_op += 2*p - 1 
            R[k][j] = rkj 
            
            # Usar 'p' para las operaciones de combinación lineal
            qj_moño = qj_moño - rkj * Q[:, k]
            cant_op += 2*p
            
        # Usar 'p' para norma2
        R[j][j] = norma2(qj_moño)
        cant_op += 2*p - 1
        if R[j][j] < tol:
            return None, None, 0
            
        # Usar 'p' para norma2 y división
        Q[:, j] = qj_moño / R[j][j]
        cant_op += p
        
    return Q, R, cant_op

def QR_con_HH(A, tol=1e-12):
    m, n = A.shape
    R = np.zeros((m, n))
    R[:, :] = A[:, :] 
    Q = idN(m)
    for k in range(n): # O(n)
        x = R[k:, k]
        sign_x0 = 1.0 if x[0] >= 0.0 else -1.0
        alpha = - sign_x0 * norma2(x)
        u = x - alpha * e1(m - k)
        nu = norma2(u)
        if nu > tol:
            u /= nu
            Hk = idN(m - k) - 2 * outer(u, u)
            H = idN(m) 
            H[k:, k:] = Hk #copiandolo queda ya O(m^2)
            R = np.dot(H, R) # esto cuesta O(m^2 * n)
            Q = np.dot(Q, H.T) # esto cuesta O(m^2 * n)
            R[k+1:, k] = 0.0
    #complejidad total de HH en caso de que m=n es de  O(n^4)
    return Q[:, :n], R[:n, :]


def calculaQR(A,metodo='RH',tol=1e-12):
    if metodo == 'GS':
        return QR_con_GS_reducido(A,tol=tol)
    elif metodo == 'RH':
        return QR_con_HH(A,tol=tol)
    else:
        raise ValueError("Método desconocido. Usar 'GS' o 'RH'.")

#LABO 6

def funcion_f(A,v):
    w_moño = dot(A, v) # cost ~ 2n^2 - n
    norma2_w_moño = norma2(w_moño) # cost ~ 2n 
    if norma2_w_moño <= 1e-15:  # evito división por cero
        w = 0
    else:
        w = w_moño / norma2_w_moño # cost ~ n
    return w # cost total ~ 2n^2 + 2n 
    
def metpot2k(A, tol=1e-15, K=1000):
    n = A.shape[1]
    cant_op = 0
    rng = np.random.default_rng(None)
    v = rng.normal(size=n)

    # f_A^2(v) inicial
    v_moño = funcion_f(A, v)          # cost ~ 2n^2 + 2n
    v_moño = funcion_f(A, v_moño)     # cost ~ 2n^2 + 2n
    e = dot(v_moño, v)                # cost ~ 2n
    k_iter = 0
    cant_op += 2 * (2*n**2 + 2*n) + 2*n
    while abs(e - 1) > tol and k_iter < K:
        v = v_moño
        v_moño = funcion_f(A, v)
        v_moño = funcion_f(A, v_moño)
        e = dot(v_moño, v)
        k_iter += 1
        cant_op += 2 * (2*n**2 + 2*n) + 2*n


    lambda_ = dot(v_moño, dot(A, v_moño))
    cant_op += 2*n**2 + n
    error = abs(e - 1)
    return v_moño, lambda_, cant_op  

def diagRH(A, tol=1e-15, K=1000):
    # v1, λ1 ← metpot2k(A,tol,K)
    v1, lambda1, *_ = metpot2k(A, tol, K)

    n = A.shape[1]

    # Normalizo v1 (por las dudas)
    nv = norma2(v1)
    if nv > 0:
        v1 = v1 / nv
    else:
        v1 = e1(n)

    # Hv1 ← I − 2 (e1−v1)(e1−v1)^t / ||e1−v1||^2   (con rama segura)
    e1_v1 = e1(n) - v1
    norma_e1_v1 = norma2(e1_v1)
    if norma_e1_v1 <= tol:
        Hv1 = idN(n)                                   # reflector trivial
    else:
        u = e1_v1 / norma_e1_v1
        Hv1 = idN(n) - 2.0 * outer(u, u)

    # ---- Rama base (n == 1) ----
    if n == 1:
        S = Hv1.copy()
        # D ← Hv1 A Hv1^T
        D = dot(Hv1, dot(A, traspuesta(Hv1)))
        return S, D

    # ---- Caso recursivo ----
    # B ← Hv1 A Hv1^T    (¡OJO con la traspuesta!)
    B = dot(Hv1, dot(A, traspuesta(Hv1)))

    # A_moño ← B[2:n, 2:n]
    A_moño = B[1:, 1:]

    # S_moño, D_moño ← diagRH(A_moño, tol, K)
    S_moño, D_moño = diagRH(A_moño, tol, K)

    # D ← diag(λ1, D_moño)
    D = np.zeros((n, n), dtype=float)
    D[0, 0] = lambda1
    D[1:, 1:] = D_moño

    # S ← Hv1 · diag(1, S_moño)
    S = idN(n)
    S[1:, 1:] = S_moño
    S = dot(Hv1, S)

    return S, D

#LABO 7 

def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """
    numbers = np.random.random((n,n))
    suma_columnas = np.sum(numbers, axis=0)
    A = numbers / suma_columnas
    return A

    

def transiciones_al_azar_uniformes(n,thres):
    """
    n la cantidad de filas (columnas) de la matriz de transición.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. 
    El elemento i,j es distinto de cero si el número generado al azar para i,j es menor o igual a thres. 
    Todos los elementos de la columna $j$ son iguales 
    (a 1 sobre el número de elementos distintos de cero en la columna).
    """
    numbers = np.random.random((n,n))
    for i in range(n):
        for j in range(n):
            if numbers[i,j] <= thres:
                numbers[i,j] = 1 
            if numbers[i,j] > thres:
                numbers[i,j] = 0
    suma_columnas = np.sum(numbers, axis=0)
    A = numbers / suma_columnas
    return A


def nucleo(A,tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial), usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """
    A_t = traspuesta(A)
    B = dot(A_t, A)
    S, D = diagRH(B, tol)

    k = 0
    n = D.shape[0]

    for i in range(n-1, -1, -1):
        if abs(D[i, i]) <= tol:
            k += 1
        else:
            break

    if k == 0:
        return np.empty((0, S.shape[0]))

    return S[:, n-k:]

def puntos_fijos(T,tol):
    """
    T una matriz cuadrada de transicion de Markov.
    tol la tolerancia para asumir que un vector es punto fijo.
    Retorna los puntos fijos de T (vectores v tales que T*v = v) como una matriz de n x k, con k el numero de puntos fijos encontrados.
    """
    n = T.shape[0]
    I = np.eye(n)
    A = T - I
    S = nucleo(A, tol)
    return S

def crea_rala(listado,m_filas,n_columnas,tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero. Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo menor a tol deben descartarse por default. 
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    tupla = (m_filas, n_columnas)
    diccionario = {}
    if listado == []:
        return diccionario, tupla
    lista_i = listado[0]
    lista_j = listado[1]
    lista_valores = listado[2]
    for index in range(len(lista_i)):
        i = lista_i[index]
        j = lista_j[index]
        valor = lista_valores[index]
        if abs(valor) >= tol:
            diccionario[(i,j)] = valor
    return diccionario, tupla

def multiplica_rala_vector(A,v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v. 
    Retorna un vector w resultado de multiplicar A con v
    """
    diccionario = A[0]
    M = np.zeros(len(v))
    for (i,j), valor in diccionario.items():
        M[i] += valor * v[j]
    return M


def es_markov(T,tol=1e-6):
    """
    T una matriz cuadrada.
    tol la tolerancia para asumir que una suma es igual a 1.
    Retorna True si T es una matriz de transición de Markov (entradas no negativas y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            if T[i,j]<0:
                return False
    for j in range(n):
        suma_columna = sum(T[:,j])
        if np.abs(suma_columna - 1) > tol:
            return False
    return True

def es_markov_uniforme(T,thres=1e-6):
    """
    T una matriz cuadrada.
    thres la tolerancia para asumir que una entrada es igual a cero.
    Retorna True si T es una matriz de transición de Markov uniforme (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    if not es_markov(T,thres):
        return False
    # cada columna debe tener entradas iguales entre si o iguales a cero
    m = T.shape[1]
    for j in range(m):
        non_zero = T[:,j][T[:,j] > thres]
        # all close
        if non_zero.size == 0:
            return True  
        close = all(np.abs(non_zero - non_zero[0]) < thres)
        if not close:
            return False
    return True


def esNucleo(A,S,tol=1e-5):
    """
    A una matriz m x n
    S una matriz n x k
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Retorna True si las columnas de S estan en el nucleo de A (es decir, A*S = 0. Esto no chequea si es todo el nucleo
    """
    for col in S.T:
        res = A @ col
        if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
            return False
    return True

#LABO 8

def svd_reducida(A,k="max",tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """
    m,n = A.shape
    a_traspuesta = traspuesta(A)
    if m>=n:
        S,D = diagRH(dot(a_traspuesta,A))
    else:
        S,D = diagRH(dot(A,a_traspuesta))
    m_D = D.shape[0]
    n_D = D.shape[1]
    if k == "max":
        k=0
        for i in range(min(n_D, m_D)):
            if D[i][i] > tol:
                k+=1
            else:
                break
    hatU = np.zeros((m,k))
    hatSig = np.zeros((k))
    hatV = np.zeros((n,k))
    for i in range(k):
            hatSig[i] = np.sqrt(D[i][i])
    if m>=n:
        for i in range(k):
            hatV[:,i] = S[:,i]
        B = dot(A,hatV)
        for i in range(B.shape[1]):
            B[:, i] /= hatSig[i]
        hatU = B
        hatU = np.asarray(hatU)
    else:
        for i in range(k):
            hatU[:,i] = S[:,i]
        B = dot(a_traspuesta,hatU)
        for i in range(B.shape[1]):
            B[:, i] /= hatSig[i]
        hatV = B
        hatV = np.asarray(hatV)
    
    return hatU,hatSig,hatV

def svd_completa(A,tol=1e-15):
    m,n = A.shape
    a_traspuesta = A.T

    if m>=n:
        S,D = diagRH(dot(a_traspuesta,A))
    else:
        S,D = diagRH(dot(A,a_traspuesta))

    s = min(m, n)
    # --- usamos un vector de sigmas para operar y al final llenamos hatSig diagonal ---
    sig = np.zeros(s)
    for i in range(s):
        sig[i] = np.sqrt(max(D[i][i], 0.0))

    if m>=n:
        hatU = np.zeros((m,m))
        hatV = S.copy()
        B = dot(A,hatV)
        counter = 0
        # usar sólo las primeras s columnas (hay s sigmas)
        for i in range(s):
            if(sig[i] > tol):
                B[:, i] /= sig[i]
                counter+=1
            else: 
                B[:, i] = np.zeros((B.shape[0]))
        G = np.zeros((B.shape[0], counter))
        c = 0
        for i in range(s):
            if norma2(B[:, i]) > tol:
                G[:, c] = B[:, i]
                c += 1
        if G.shape[1] > 0:
            G = QR_con_HH(G)[0]
        else:
            G = np.zeros((m, 0))

        nuevas = []  # inicializo siempre
        if G.shape[1] < m:
            diff = m - G.shape[1]
            for i in range(diff):
                counter = 0
                while (G.shape[1] + len(nuevas) < m) and (counter < m): 
                    e = np.zeros(m, dtype=float)
                    e[counter] = 1.0
                    w = e.copy()
                    proj = np.zeros_like(w)
                    for j in range(G.shape[1]):
                        alpha = dot(G[:,j],w)
                        proj += alpha * G[:, j]
                    for q in nuevas:
                        alpha = dot(q, w)
                        proj += alpha * q
                    w -= proj
                    nrm = norma2(w)
                    if (nrm > tol):
                        w = w / nrm
                        nuevas.append(w)
                        break
                    if counter >= m:   # >=
                        return -1
                    counter+=1
        
        for i in range(G.shape[1]):
            hatU[:,i] = G[:,i]
        for i in range(len(nuevas)):
            hatU[:,G.shape[1]+i] = nuevas[i]
        hatU = np.asarray(hatU)

    else:
        hatV = np.zeros((n,n))
        hatU = S.copy()
        B = dot(a_traspuesta,hatU)
        counter = 0
        # usar sólo las primeras s columnas (hay s sigmas)
        for i in range(s):
            if(sig[i] > tol):
                B[:, i] /= sig[i]
                counter+=1
            else: 
                B[:, i] = np.zeros((B.shape[0]))
        G = np.zeros((B.shape[0], counter))
        c = 0
        for i in range(s):
            if norma2(B[:, i]) > tol:
                G[:, c] = B[:, i]
                c += 1
        if G.shape[1] > 0:
            G = QR_con_HH(G)[0]
        else:
            G = np.zeros((n, 0))

        nuevas = []  # inicializo siempre
        if G.shape[1] < n:
            diff = n - G.shape[1]
            for i in range(diff):
                counter = 0
                while (G.shape[1] + len(nuevas) < n) and (counter < n): 
                    e = np.zeros(n, dtype=float)
                    e[counter] = 1.0
                    w = e.copy()
                    proj = np.zeros_like(w)
                    for j in range(G.shape[1]):
                        alpha = dot(G[:,j],w)
                        proj += alpha * G[:, j]
                    for q in nuevas:
                        alpha = dot(q, w)
                        proj += alpha * q
                    w -= proj
                    nrm = norma2(w)
                    if (nrm > tol):
                        w = w / nrm
                        nuevas.append(w)
                        break
                    if counter >= n:   # >=
                        return -1
                    counter+=1
        
        for i in range(G.shape[1]):
            hatV[:,i] = G[:,i]
        for i in range(len(nuevas)):
            hatV[:,G.shape[1]+i] = nuevas[i]
        hatV = np.asarray(hatV)
    
    # --- construir hatSig (m x n) al final con la diagonal sig ---
    hatSig = np.zeros((m, n))
    for i in range(s):
        hatSig[i, i] = sig[i]

    return hatU,hatSig,hatV

