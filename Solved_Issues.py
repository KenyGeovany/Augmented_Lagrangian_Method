import numpy as np

# ------------------------------------------------------------------------------------------------------------
# Transformación lineal por pixel
M = np.ones((2, 2, 2, 2))
M[:, :, 0, 0] = np.array([[1, 2],
                          [2, 3]])

M[:, :, 0, 1] = np.array([[3, 2],
                          [2, 4]])

M[:, :, 1, 0] = np.array([[5, 4],
                          [4, 5]])

M[:, :, 1, 1] = np.array([[1, 3],
                          [2, 1]])

M_reshaped = M.transpose((2, 3, 0, 1)).reshape((4, 2, 2))
inv_reshaped = np.linalg.inv(M_reshaped)
inv_M = inv_reshaped.reshape((2, 2, 2, 2)).transpose((2, 3, 0, 1))

# Resultado
print(f"Inversión de matrices (01.07.2024) -------")
print(inv_M[:, :, 0, 0])
print(inv_M[:, :, 0, 1])
print(inv_M[:, :, 1, 0])
print(inv_M[:, :, 1, 1])

# ------------------------------------------------------------------------------------------------------------
# Definimos un campo vectorial por pixel
F = np.ones((2,2,2))
F[:,0,0] = np.array([1,2])
F[:,0,1] = np.array([4,5])
F[:,1,0] = np.array([3,7])
F[:,1,1] = np.array([0,3])

# Reestructurar la matriz para aplicar np.linalg.inv de manera vectorizada
M_reshaped = M.transpose((2, 3,0, 1)).reshape((4, 2, 2))
F_reshaped = F.transpose((1, 2, 0)).reshape((4, 2, 1))  # Vector columna
C_reshaped = np.matmul(np.linalg.inv(M_reshaped),F_reshaped)
C = (C_reshaped.reshape((2, 2, 2, 1)).transpose((2, 3, 0, 1))).reshape(2,2,2)

# Resultado
print(f"Producto matriz-vector (01.07.2024) -------")
print(C[:,0,0])
print(C[:,0,1])
print(C[:,1,0])
print(C[:,1,1])

# ------------------------------------------------------------------------------------------------------------
# Parámetro lambda
lambda_reg = 1e-5

# Crear una matriz identidad de las dimensiones apropiadas
identity_matrix = np.eye(2)

# Añadir lambda*I a cada submatriz de M de manera vectorizada
Q_plus_lambda_I = M + lambda_reg * identity_matrix[:, :, np.newaxis, np.newaxis]

# Resultado
print(f"Regularización Levenberg-Marquart (03.07.2024) -------")
print("M original:")
print(M[:,:,1,1])
print("M + lambda*I:")
print(Q_plus_lambda_I[:,:,1,1])

# ------------------------------------------------------------------------------------------------------------
M = np.zeros((3,2,2))
M[0] = np.array([[1,3],
                 [2,5]]
                )
M[1] = np.array([[2,2],
                 [4,0]])

M[2] = np.array([[7,1],
                 [4,0]])

def escalar_delta_bool(x):
    # Return 1 if x > 2
    # return 0 if x <= 2
    return int(x > 2)

delta_bool = np.vectorize(escalar_delta_bool)
M_2 = delta_bool(M)

print(f"Vectorizar función escalar (04.07.2024) -------")
print(M_2)
