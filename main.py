import numpy as np
import random
import matplotlib.pyplot as plt
from QuadraticFitting import QuadraticFitting as QF

SIZE = 50

a0 = 3
a1 = 2
a2 = 1

x_real = np.linspace(0, 10, SIZE)
y_real = a0 + a1 * x_real + a2 * np.power(x_real, 2)

x_all = []
y_all = []
for i in range(SIZE):
    x_all.append(x_real[i] + random.uniform(-0.5, 0.5))
    y_all.append(y_real[i] + random.uniform(-0.5, 0.5))
for i in range(SIZE):
    x_all.append(random.uniform(0, 10))
    y_all.append(random.uniform(-50, 150))

x_all = np.array(x_all)
y_all = np.array(y_all)


def test_least_square():
    a_result = QF.least_square(x_all, y_all)

    y_pre_real = a_result[0] + a_result[1] * x_real + a_result[2] * np.power(x_real, 2)
    y_pre_all = a_result[0] + a_result[1] * x_all + a_result[2] * np.power(x_all, 2)

    rmse_real = np.sqrt(np.average((y_real - y_pre_real) ** 2))
    rmse_all = np.sqrt(np.average((y_all - y_pre_all) ** 2))

    fig = plt.figure(1)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("Least Square")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax1.scatter(x_all[:SIZE], y_all[:SIZE], c="red", label="noise data")
    ax1.scatter(x_all[SIZE:], y_all[SIZE:], c="orange", label="error data")
    ax1.plot(x_real, y_pre_real)
    text = "a0 = {0:.2f}, a1 = {1:.2f}, a2 = {2:.2f} \n" \
           "RMSE of real data: {3:.2f} \n" \
           "RMSE of all data: {4:.2f}".format(a_result[0], a_result[1], a_result[2], rmse_real, rmse_all)
    plt.text(0, 125, text)
    plt.legend(loc="upper right")
    # plt.show()


def test_ransac():
    a_result = QF.RANSAC(x_all, y_all)

    y_pre_real = a_result[0] + a_result[1] * x_real + a_result[2] * np.power(x_real, 2)
    y_pre_all = a_result[0] + a_result[1] * x_all + a_result[2] * np.power(x_all, 2)

    rmse_real = np.sqrt(np.average((y_real - y_pre_real) ** 2))
    rmse_all = np.sqrt(np.average((y_all - y_pre_all) ** 2))

    fig = plt.figure(2)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("RANSAC")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax1.scatter(x_all[:SIZE], y_all[:SIZE], c="red", label="noise data")
    ax1.scatter(x_all[SIZE:], y_all[SIZE:], c="orange", label="error data")
    ax1.plot(x_real, y_pre_real)
    text = "a0 = {0:.2f}, a1 = {1:.2f}, a2 = {2:.2f} \n" \
           "RMSE of real data: {3:.2f} \n" \
           "RMSE of all data: {4:.2f}".format(a_result[0], a_result[1], a_result[2], rmse_real, rmse_all)
    # plt.text(0, 125, text)
    plt.legend(loc="upper right")
    plt.show()


def test_image():
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax1.scatter(x_real, y_real, label="real data")
    ax1.scatter(x_all[:SIZE], y_all[:SIZE], c="red", label="noise data")
    ax1.scatter(x_all[SIZE:], y_all[SIZE:], c="orange", label="error data")
    plt.ylim([-55, 155])
    plt.legend(loc="upper right")
    plt.show()

def main():
    test_least_square()
    test_ransac()


if __name__ == "__main__":
    main()
