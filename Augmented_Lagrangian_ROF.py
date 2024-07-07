import numpy as np

from lib_dif_operators import *
from lib_generic import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import csv
from numpy import genfromtxt
import sys

from numpy.fft import fft2, ifft2, fftshift

# Load the img_original image
img_clean = open_img('input/img_clean/kodim1.png', echo=False)
img_noise = open_img('input/img_denoising_gaussian/sigma25kodim1.png', echo=False)

# Extract the BGR components
blue_clean, green_clean, red_clean = cv2.split(img_clean)
u = np.array([red_clean, green_clean, blue_clean]).astype(float)

# Extract the BGR components of the noised image
blue_noise, green_noise, red_noise = cv2.split(img_noise)
u_0 = np.array([red_noise, green_noise, blue_noise]).astype(float)

# Número impar de renglones y columnas
u = u[0,0:-1,0:-1]
u_0 = u_0[0,0:-1,0:-1]
save_img(u, "output/kodim1_clear.png")
save_img(u_0, "output/kodim1_degraded.png")


def escalar_delta_bool(x):
    # Return 1 if x > 1
    # return 0 if x <= 1
    return int(x > 1)

delta_bool = np.vectorize(escalar_delta_bool)

def dx_f2(m):
    rows, cols = m.shape
    w = m[:, 1:] - m[:, :-1]
    w = np.concatenate((w, np.zeros((rows, 1))), axis=1)
    return w

def dy_f2(m):
    rows, cols = m.shape
    w = m[1:, :] - m[:-1, :]
    w = np.concatenate((w, np.zeros((1, cols))), axis=0)
    return w

def dx_b2(m):
    rows, cols = m.shape
    w = - m[:, 1:] + m[:, :-1]
    w = np.concatenate((np.zeros((rows, 1)), w), axis=1)
    return w

def dy_b2(m):
    rows, cols = m.shape
    w = - m[1:, :] + m[:-1, :]
    w = np.concatenate((np.zeros((1, cols)), w), axis=0)
    return w

def gradient_2(m,type="forward"):
    """
    Gradient of a function in format [u1,u2,...] where ui are nd_arrays of
    size (rows x cols)
    :param m: a vector of nd_arrays of size (n x rows x cols)
    :return: a vector of nd_arrays of size (2n x rows x cols) of the form
            [dx_u1, dy_u1, dx_u2, dy_u2, ...]
    """
    # Get the number of components of m
    num_comp = len(m.shape)
    if num_comp == 2:
        n = 1
    else:
        n = m.shape[0]

    grad = []
    if n > 1:
        for i in range(n):
            if type == "forward":
                grad.append(dx_f2(m[i]))
                grad.append(dy_f2(m[i]))
            elif type == "backward":
                grad.append(dx_b2(m[i]))
                grad.append(dy_b2(m[i]))
    elif n == 1:
        if type == "forward":
            grad = [dx_f2(m), dy_f2(m)]
        elif type == "backward":
            grad = [dx_b2(m), dy_b2(m)]

    return np.array(grad)

def divergence_2(m,type="forward"):
    """
    Divergence of a function in format [u1,u2,...] where ui are nd_arrays of
    size (rows x cols). The function must have 2n components
    :param m: a vector of nd_arrays of size (2n x rows x cols)
    :return: a vector of nd_arrays of size (n x rows x cols) of the form
            [dx_u1 + dy_u1, dx_u2 + dy_u2, ...]
    """
    # Get the (number of components) / 2 of m
    num_comp = len(m.shape)
    if num_comp == 2:
        n = 1
    else:
        n = int((m.shape[0])/2)

    div = []
    if n > 1:
        for i in range(n):
            if type == "forward":
                div.append(dx_f2(m[2 * i]) + dy_f2(m[2 * i + 1]))
            elif type == "backward":
                div.append(dx_b2(m[2 * i]) + dy_b2(m[2 * i + 1]))
    elif n == 1:
        if type == "forward":
            div = dx_f2(m[0]) + dy_f2(m[1])
        elif type == "backward":
            div = dx_b2(m[0]) + dy_b2(m[1])
    return np.array(div)

# Terminado: Exact minimizer
def Agmented_Lagrangian_1(u_0):
    save_img(u, "output/image_clear.png")
    save_img(u_0, "output/image_denoised.png")
    # Parameters
    alpha = 0.04 #0.04
    r = 0.2 #4 #0.2
    gamma = 1.1 #1 #1.1

    # Constants
    (rows, cols) = u_0.shape
    print(rows,cols)
    r_mid = int((rows - 1) / 2)
    c_mid = int((cols - 1) / 2)

    tol = 0.01
    iterMax = 200

    sigma_1 = +1  # Correct
    sigma_2 = +1  # Correct
    # Kernel D_x
    D_xb = np.zeros((rows, cols))
    D_xb[r_mid, c_mid] = sigma_2 # sigma_1 = +1, sigma_2 = +1
    D_xb[r_mid, c_mid + sigma_1] = -sigma_2 #  sigma_1 = +1, sigma_2 = 1
    TF_D_xb = fft2(fftshift(D_xb))

    # Kernel D_y
    D_yb = np.zeros((rows, cols))
    D_yb[r_mid, c_mid] = sigma_2
    D_yb[r_mid + sigma_1, c_mid] = -sigma_2
    TF_D_yb = fft2(fftshift(D_yb))

    # Kernel Laplacian
    Laplacian = np.zeros((rows, cols))
    Laplacian[r_mid, c_mid] = -4
    Laplacian[r_mid - 1, c_mid] = 1
    Laplacian[r_mid + 1, c_mid] = 1
    Laplacian[r_mid, c_mid-1] = 1
    Laplacian[r_mid, c_mid+1] = 1
    TF_Laplacian = fft2(fftshift(Laplacian))  # Correct

    # Initial states
    mu = np.ones((2,rows,cols)) # El punto inicial no tiene relevancia
    theta = 0
    u_k = (1-theta)*u_0  # np.zeros((rows,cols)) is better but we get small PSNR. u_0 generates salt-pepper noise

    k=0
    mse = 1
    while (mse > tol and k < iterMax):
        # Update M field
        w = r * gradient_2(u_k,type="forward") - mu # Correct

        w_1 = (1/r)*(1-(1/(np.linalg.norm(w,axis=0)+1)))*w # 2 is better  by 0.01 but 1 is more theoretical
        w_2 = delta_bool(np.linalg.norm(w,axis=0))
        q = w_1*w_2 # Correcto (se comprobó comparando con un algoritmo con bucles)

        # Update u_k
        u_kp1 = np.real(ifft2((alpha * fft2(u_0) - TF_D_xb * (fft2(mu[0]) + r * fft2(q[0])) - TF_D_yb * (
                    fft2(mu[1]) + r * fft2(q[1]))) / (alpha - r * TF_Laplacian)))  # Correct

        # Update mu
        mu = mu + r*(q-gradient_2(u_kp1,type="forward"))  # Correct

        # Update gamma
        if r < 100:
            r = r*gamma

        # Compute the mse
        mse = np.sqrt(np.sum((u_kp1-u_k)**2)/u_0.size)

        # Update u_k
        u_k = np.copy(u_kp1)
        k += 1

        # Print Info
        print(f"-----------\n Iter k: {k}. r = {r}")
        print("MSE = ", mse)
        print(f"PSNR en iteración k={k} es {psnr(np.clip(u_k[1:-1, 1:-1], 0, 255) / 255., u[1:-1, 1:-1] / 255.)}")
        save_img(u_k, "output/image_iteration_" + str(k) + ".png")
    return u_k

# Terminado: Fixed Pont
def Agmented_Lagrangian_2(u_0):
    save_img(u, "output/image_clear.png")
    save_img(u_0, "output/image_denoised.png")
    # Parameters
    alpha = 0.04
    r = 0.2
    gamma = 1.1

    # Constants
    (rows, cols) = u_0.shape
    print(rows,cols)
    r_mid = int((rows - 1) / 2)
    c_mid = int((cols - 1) / 2)

    tol = 0.01
    iterMax = 200

    sigma_1 = +1  # Correct
    sigma_2 = +1  # Correct
    # Kernel D_x
    D_xb = np.zeros((rows, cols))
    D_xb[r_mid, c_mid] = sigma_2 # sigma_1 = +1, sigma_2 = +1
    D_xb[r_mid, c_mid + sigma_1] = -sigma_2 #  sigma_1 = +1, sigma_2 = 1
    TF_D_xb = fft2(fftshift(D_xb))

    # Kernel D_y
    D_yb = np.zeros((rows, cols))
    D_yb[r_mid, c_mid] = sigma_2
    D_yb[r_mid + sigma_1, c_mid] = -sigma_2
    TF_D_yb = fft2(fftshift(D_yb))

    # Kernel Laplacian
    Laplacian = np.zeros((rows, cols))
    Laplacian[r_mid, c_mid] = -4
    Laplacian[r_mid - 1, c_mid] = 1
    Laplacian[r_mid + 1, c_mid] = 1
    Laplacian[r_mid, c_mid-1] = 1
    Laplacian[r_mid, c_mid+1] = 1
    TF_Laplacian = fft2(fftshift(Laplacian))  # Correct

    # Initial states
    mu = np.ones((2,rows,cols)) # El punto inicial no tiene relevancia
    q = np.ones((2,rows,cols))
    theta = 0
    u_k = (1-theta)*u_0  # np.zeros((rows,cols)) is better but we get small PSNR. u_0 generates salt-pepper noise

    k=0
    mse = 1
    while (mse > tol and k < iterMax):
        # Update M field
        q_l = np.copy(q)
        q_lp1 = np.copy(q)
        for l in range(100):
            S_q = np.sqrt(q_l[0]**2 + q_l[1]**2 + 0.0001)
            F_q = np.array([(1/S_q)*q_l[0] + r*(q_l[0]-dx_f2(u_k)) + mu[0], (1/S_q)*q_l[1] + r*(q_l[1]-dy_f2(u_k)) + mu[1]])
            eta = (1 / S_q) + r
            q_lp1[0] = (1 / eta) * (r * dx_f2(u_k) - mu[0])  # Correcto
            q_lp1[1] = (1 / eta) * (r * dy_f2(u_k) - mu[1])  # Correcto

            # Printo info de q_l
            mse_l = np.sqrt(np.sum((q_lp1 - q_l) ** 2) / q_l.size)
            if l % 10 == 0:
                print(f"MSE(l) = {mse_l}. cont_l = {l}")
                # Imprimimos el valor de F(q_l) que debería tender a cero
                print(f"F(q) = {np.sqrt(np.sum(F_q ** 2))}, l = {l}")
            # Actualizamos q_l
            q_l = np.copy(q_lp1)
        q = np.copy(q_l)

        # Update u_k
        u_kp1 = np.real(ifft2((alpha * fft2(u_0) - TF_D_xb * (fft2(mu[0]) + r * fft2(q[0])) - TF_D_yb * (
                    fft2(mu[1]) + r * fft2(q[1]))) / (alpha - r * TF_Laplacian)))  # Correct

        # Update mu
        mu = mu + r*(q-gradient_2(u_kp1,type="forward"))  # Correct

        # Update gamma
        if r < 100:
            r = r*gamma

        # Compute the mse
        mse = np.sqrt(np.sum((u_kp1-u_k)**2)/u_0.size)

        # Update u_k
        u_k = np.copy(u_kp1)
        k += 1

        # Print Info
        print(f"-----------\n Iter k: {k}. r = {r}")
        print("MSE = ", mse)
        print(f"PSNR en iteración k={k} es {psnr(np.clip(u_k[1:-1, 1:-1], 0, 255) / 255., u[1:-1, 1:-1] / 255.)}")
        save_img(u_k, "output/image_iteration_" + str(k) + ".png")
    return u_k

# Terminado: Newton exact
def Agmented_Lagrangian_3(u_0):
    save_img(u, "output/image_clear.png")
    save_img(u_0, "output/image_denoised.png")
    # Parameters
    alpha = 0.04
    r = 0.2
    gamma = 1.1

    # Constants
    (rows, cols) = u_0.shape
    print(rows,cols)
    r_mid = int((rows - 1) / 2)
    c_mid = int((cols - 1) / 2)

    tol = 0.01
    iterMax = 200

    sigma_1 = +1  # Correct
    sigma_2 = +1  # Correct
    # Kernel D_x
    D_xb = np.zeros((rows, cols))
    D_xb[r_mid, c_mid] = sigma_2 # sigma_1 = +1, sigma_2 = +1
    D_xb[r_mid, c_mid + sigma_1] = -sigma_2 #  sigma_1 = +1, sigma_2 = 1
    TF_D_xb = fft2(fftshift(D_xb))

    # Kernel D_y
    D_yb = np.zeros((rows, cols))
    D_yb[r_mid, c_mid] = sigma_2
    D_yb[r_mid + sigma_1, c_mid] = -sigma_2
    TF_D_yb = fft2(fftshift(D_yb))

    # Kernel Laplacian
    Laplacian = np.zeros((rows, cols))
    Laplacian[r_mid, c_mid] = -4
    Laplacian[r_mid - 1, c_mid] = 1
    Laplacian[r_mid + 1, c_mid] = 1
    Laplacian[r_mid, c_mid-1] = 1
    Laplacian[r_mid, c_mid+1] = 1
    TF_Laplacian = fft2(fftshift(Laplacian))  # Correct

    # Initial states
    mu = np.zeros((2,rows,cols)) # El punto inicial no tiene relevancia
    q = np.zeros((2,rows,cols))
    theta = 0
    u_k = (1-theta)*u_0  # np.zeros((rows,cols)) is better but we get small PSNR. u_0 generates salt-pepper noise

    k=0
    mse = 1
    while (mse > tol and k < iterMax):
        # Update M field
        q_l = np.copy(q)
        q_lp1 = np.copy(q)
        for l in range(200):
            S_q = np.sqrt(q_l[0]**2 + q_l[1]**2 + 0.0001)
            F_q = np.array([(1/S_q)*q_l[0] + r*(q_l[0]-dx_f2(u_k)) + mu[0], (1/S_q)*q_l[1] + r*(q_l[1]-dy_f2(u_k)) + mu[1]])
            eta = (1/S_q) + r
            det_J = eta*(eta - (q_l[0]**2+q_l[1]**2)/(S_q**3))
            q_lp1[0] = q_l[0] - (1/det_J)*(
                    ((q_l[1]/S_q**3) * (q_l[0]*mu[1]-q_l[1]*mu[0]-r*(q_l[0]*dy_f2(u_k)-q_l[1]*dx_f2(u_k))))+
                    eta*(eta*q_l[0] + mu[0] - r*dx_f2(u_k)))  # Correcto
            q_lp1[1] = q_l[1] - (1/det_J)*(
                    ((-q_l[0]/S_q**3)*(q_l[0]*mu[1]-q_l[1]*mu[0]-r*(q_l[0]*dy_f2(u_k)-q_l[1]*dx_f2(u_k))))+
                    eta*(eta*q_l[1] + mu[1] - r*dy_f2(u_k)))  # Correcto
            # Printo info de q_l
            mse_l = np.sqrt(np.sum((q_lp1 - q_l) ** 2) / q_l.size)
            if l % 20 == 0:
                print(f"MSE(l) = {mse_l}. cont_l = {l}")
                # Imprimimos el valor de F(q_l) que debería tender a cero
                print(f"F(q) = {np.sqrt(np.sum(F_q ** 2))}, l = {l}")
            # Actualizamos q_l
            q_l = np.copy(q_lp1)
        q = np.copy(q_l)


        # Update u_k
        u_kp1 = np.real(ifft2((alpha * fft2(u_0) - TF_D_xb * (fft2(mu[0]) + r * fft2(q[0])) - TF_D_yb * (
                    fft2(mu[1]) + r * fft2(q[1]))) / (alpha - r * TF_Laplacian)))  # Correct

        # Update mu
        mu = mu + r*(q-gradient_2(u_kp1,type="forward"))  # Correct

        # Update gamma
        if r < 100:
            r = r*gamma

        # Compute the mse
        mse = np.sqrt(np.sum((u_kp1-u_k)**2)/u_0.size)

        # Update u_k
        u_k = np.copy(u_kp1)
        k += 1

        # Print Info
        print(f"-----------\n Iter k: {k}. r = {r}")
        print("MSE = ", mse)
        print(f"PSNR en iteración k={k} es {psnr(np.clip(u_k[1:-1, 1:-1], 0, 255) / 255., u[1:-1, 1:-1] / 255.)}")
        save_img(u_k, "output/image_iteration_" + str(k) + ".png")
    return u_k

# Terminado: Newton approximated
def Agmented_Lagrangian_4(u_0):
    save_img(u, "output/image_clear.png")
    save_img(u_0, "output/image_denoised.png")
    # Parameters
    alpha = 0.04
    r = 0.2
    gamma = 1.1

    # Constants
    (rows, cols) = u_0.shape
    print(rows,cols)
    r_mid = int((rows - 1) / 2)
    c_mid = int((cols - 1) / 2)

    tol = 0.01
    iterMax = 200

    sigma_1 = +1  # Correct
    sigma_2 = +1  # Correct
    # Kernel D_x
    D_xb = np.zeros((rows, cols))
    D_xb[r_mid, c_mid] = sigma_2 # sigma_1 = +1, sigma_2 = +1
    D_xb[r_mid, c_mid + sigma_1] = -sigma_2 #  sigma_1 = +1, sigma_2 = 1
    TF_D_xb = fft2(fftshift(D_xb))

    # Kernel D_y
    D_yb = np.zeros((rows, cols))
    D_yb[r_mid, c_mid] = sigma_2
    D_yb[r_mid + sigma_1, c_mid] = -sigma_2
    TF_D_yb = fft2(fftshift(D_yb))

    # Kernel Laplacian
    Laplacian = np.zeros((rows, cols))
    Laplacian[r_mid, c_mid] = -4
    Laplacian[r_mid - 1, c_mid] = 1
    Laplacian[r_mid + 1, c_mid] = 1
    Laplacian[r_mid, c_mid-1] = 1
    Laplacian[r_mid, c_mid+1] = 1
    TF_Laplacian = fft2(fftshift(Laplacian))  # Correct

    # Initial states
    mu = np.zeros((2,rows,cols)) # El punto inicial no tiene relevancia
    q = np.zeros((2,rows,cols))
    theta = 0
    u_k = (1-theta)*u_0  # np.zeros((rows,cols)) is better but we get small PSNR. u_0 generates salt-pepper noise

    k=0
    mse = 1
    while (mse > tol and k < iterMax):
        # Update M field
        q_l = np.copy(q)
        q_lp1 = np.copy(q)
        for l in range(200):
            S_q = np.sqrt(q_l[0]**2 + q_l[1]**2 + 0.0001)
            F_q = np.array([(1/S_q)*q_l[0] + r*(q_l[0]-dx_f2(u_k)) + mu[0], (1/S_q)*q_l[1] + r*(q_l[1]-dy_f2(u_k)) + mu[1]])
            eta = (1/S_q) + r
            q_lp1[0] = q_l[0] - (1 / eta) * (eta * q_l[0] + mu[0] - r * dx_f2(u_k))  # Correcto
            q_lp1[1] = q_l[1] - (1 / eta) * (eta * q_l[1] + mu[1] - r * dy_f2(u_k))  # Correcto
            # Printo info de q_l
            mse_l = np.sqrt(np.sum((q_lp1 - q_l) ** 2) / q_l.size)
            if l % 20 == 0:
                print(f"MSE(l) = {mse_l}. cont_l = {l}")
                # Imprimimos el valor de F(q_l) que debería tender a cero
                print(f"F(q) = {np.sqrt(np.sum(F_q ** 2))}, l = {l}")
            # Actualizamos q_l
            q_l = np.copy(q_lp1)
        q = np.copy(q_l)

        # Update u_k
        u_kp1 = np.real(ifft2((alpha * fft2(u_0) - TF_D_xb * (fft2(mu[0]) + r * fft2(q[0])) - TF_D_yb * (
                    fft2(mu[1]) + r * fft2(q[1]))) / (alpha - r * TF_Laplacian)))  # Correct

        # Update mu
        mu = mu + r*(q-gradient_2(u_kp1,type="forward"))  # Correct

        # Update gamma
        if r < 100:
            r = r*gamma

        # Compute the mse
        mse = np.sqrt(np.sum((u_kp1-u_k)**2)/u_0.size)

        # Update u_k
        u_k = np.copy(u_kp1)
        k += 1

        # Print Info
        print(f"-----------\n Iter k: {k}. r = {r}")
        print("MSE = ", mse)
        print(f"PSNR en iteración k={k} es {psnr(np.clip(u_k[1:-1, 1:-1], 0, 255) / 255., u[1:-1, 1:-1] / 255.)}")
        save_img(u_k, "output/image_iteration_" + str(k) + ".png")
    return u_k

def Descent_Gradient_1(u_0):
    # Parameters
    alpha = 0.06
    dt = 0.05
    eta = 0.0001
    tol = 0.0001
    iterMax = 10000

    # Descenso de gradiente
    u_k = np.copy(u_0)
    u_kp1 = np.copy(u_0)
    k = 0
    mse = 1
    while (mse > tol and k < iterMax) or (k < 50):
        # Update u_k
        u_k = np.copy(u_kp1)

        # Compute the gradient
        gd = gradient_2(u_k)

        # Compute the gradient norms
        gd_norm = np.sqrt(gd[0] ** 2 + gd[1] ** 2 + eta)

        # Divide the gradient by its norm (splited)
        g = gd / gd_norm

        # Compute the divergences
        u_kp1 = u_k - dt * (alpha*(u_k - u_0) + divergence_2(g,type="backward"))

        # Compute the mse
        mse = np.mean((u_kp1-u_k)**2)

        # Actualize iterador
        k += 1
        if k % 100 == 1:
            print("K =", k, "MSE = ", mse)
            save_img(u_k, "output/image_iteration_GD_" + str(k) + ".png")
            print(f"PSNR en iteración k={k} es {psnr(np.clip(u_k[1:-1, 1:-1], 0, 255) / 255., u[1:-1, 1:-1] / 255.)}")

    return u_k

#u_k = Agmented_Lagrangian_4(u_0)
u_k = Descent_Gradient_1(u_0)

save_img(u_k, "output/image_clear_1.png")
print(f"PSNR(u_0,u_clear) = {psnr(np.clip(u_0[1:-1, 1:-1], 0, 255) / 255., u[1:-1, 1:-1] / 255.)}")





# u_k = np.real(ifft2(fft2(alpha * u_0 - divergence_2(mu,type="forward") - r*divergence_2(M,type="forward")) / (alpha - r * TF_Laplacian))) # Incorrect