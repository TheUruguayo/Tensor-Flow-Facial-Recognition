import numpy as np
import matplotlib.pyplot as plt
import h5py
import PIL
import scipy
import scipy.misc

from PIL import Image
from scipy import ndimage

import cv2
import os
from numpy import asarray
from numpy import savetxt

def load_images_from_folder(folder, size, bandera):
    """
       Carga las imagenes las comprime, estandariza y aplica efectos

        """
    images = []
    size = size
    count = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))

        if img is not None:
            img = cv2.resize(img, size)
            img = img / 255  # esto centra y estandariza la database, aveces tambien restas el promedio
            # print(filename)
            images.append(img)
            if bandera:  # agregamos efectos para lograr mayor cantidad de entradas
                img2 = cv2.stylization(img, sigma_s=150, sigma_r=0.25)
                images.append(img2)
                img2 = cv2.flip(img, 1)
                images.append(img2)
        if count == 810:
            break
        count = count + 1

    return images


def sigmoid(z):
    """
    En este caso de clasificacion binaria vamos a utilizar sigmoid
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
     En el caso de gradient descent no es nescesario inizializar de manera aleatoria
    de esta manera  tenemos lo siguiente
     """
    w = np.zeros((dim, 1), dtype=int)
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    """
    Foward Propagation, de izquierda a derecha, estamos calculando una primera aproximacion
    Argumentos:
    w -- weights, numpay array de (num_px * num_px * 3, 1)
    b -- bias, escalar
    X -- nuestras entradas (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (contiene 0 si no es Ignacio, 1 si es Ignacio) tamaño(1, number of examples)

    Backwards, derecha a izquierda:
    cost -- el costo de la logist regression
    dw -- gradiente del costo con respecto w, las dimecniones son las misma que W
    db -- gradiente del costo con respecto b, las dimecniones son las misma que b

    """
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    h = np.dot(w.T, X) + b
    z = h
    ac = 1 / (1 + np.exp(-z))  # compute activation
    cost1 = np.dot((1 - Y), (np.log(1 - ac)).T)
    cost2 = np.dot(Y, (np.log(ac)).T)  # compute cost
    cost = -1 / m * (cost1 + cost2)

    # BACKWARD PROPAGATION (TO FIND GRAD)

    dz = (ac - Y)
    db = np.sum(dz) / m
    dw = np.dot(X, dz.T) * 1 / m  # esto funciona aca por el tipo de W,en otros caso el correcto es cambiado dz y X


    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    # assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    Funcion de optimizacion de w y b
    This function optimizes w and b by running a gradient descent algorithm


    Argumentos:
    w -- weights, numpay array de (num_px * num_px * 3, 1)
    b -- bias, escalar
    X -- nuestras entradas (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (contiene 0 si no es Ignacio, 1 si es Ignacio) tamaño(1, number of examples)
    num_iterations -- numero de iterecion en el loop de optimizacion
    learning_rate -- es la constante que va en la optimizacion
    print_cost -- si queremos ver o no el costo cada 100 repeticiones

    Retorna:
    params -- diccionario conteniendo w y b
    grads -- diccionario que contiene los pesos y bias del gradient descent con respecto a la funcion de costo
    costs -- lista de los costos computados
    """
    costs = []
    for i in range(num_iterations):
        # Cost and gradient calculation

        grads, cost = propagate(w, b, X, Y)
        a = learning_rate

        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - a * dw
        b = b - a * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    '''
   Utilizando todo lo caclulado usamos la activacion para determinar si la salida es "1" o "0"
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    h = np.dot(w.T, X) + b
    z = h
    A = 1 / (1 + np.exp(-z))
    for i in range(A.shape[1]): # se puede utilizar otra funcion con trheshold, que esta mejor optimizada
        # Convert probabilities A[0,i] to actual predictions p[0,i]

        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        elif A[0, i] > 0.5:
            Y_prediction[0, i] = 1
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
        Utilza todas las funciones atneriores
            """
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=True)

    w = parameters["w"]
    b = parameters["b"]
    print(b.shape)
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d