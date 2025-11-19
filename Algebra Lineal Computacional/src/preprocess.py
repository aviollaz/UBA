from typing import Tuple
import numpy as np

def cargarDataset(carpeta: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve las matrices necesitadas para procesar las imagenes.

    Parameters
    ----------
    - carpeta :str
        La carpeta base donde están los embeddings para entrenar validar.

    Returns:
    -----------
    - Xt: np.ndarray
        Matriz embeddings train
    - Yt: np.ndarray
        Matriz clasificación train
    - Xv: np.ndarray
        Matríz embedding validación
    - Yv: np.ndarray
        Matríz clasificación validación
    """

    Xt, Yt = extraerMatrices(carpeta, segmento="train")
    Xv, Yv = extraerMatrices(carpeta, segmento='val')

    return Xt, Yt, Xv, Yv

def extraerMatrices(carpeta: str, segmento:str):
    """
    Extrae las matrices Xt, Xv del segmento indicado

    Parameters
    ------------
    - carpeta: str
        Carpeta root del dataset
    - segmento: str 
        Es 'val' o 'train'

    Returns
    ---------
    - X: np.ndarray
        Matriz embeddings 
    - Y: np.ndarray
        Matriz clasificación 
    """
    X_cats = np.load(f"{carpeta}/{segmento}/cats/efficientnet_b3_embeddings.npy")    
    X_dogs = np.load(f"{carpeta}/{segmento}/dogs/efficientnet_b3_embeddings.npy")

    X = np.concatenate((X_cats, X_dogs), axis=1)

    num_cats = X_cats.shape[1]
    num_dogs = X_dogs.shape[1]

    Y = [[1,0] for _ in range(num_cats)] + [[0,1] for _ in range(num_dogs)] 
    Y = np.array(Y)
    Y = Y.T

    # # randomizar orden de los datos
    idx = np.random.permutation(X.shape[1])
    X = X[:, idx]
    Y = Y[:, idx]

    return X,Y


if __name__ == '__main__':
    Xt, Yt, Xv, Yv = cargarDataset('dataset/cats_and_dogs')
