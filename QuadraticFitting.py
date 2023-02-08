import math
import random
import numpy as np


class QuadraticFitting:
    @staticmethod
    def least_square(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        size = x.size
        x_train = np.zeros((size, 3))
        y_train = np.zeros((size, 1))

        for i in range(size):
            x_train[i, :] = [1, x[i], x[i] ** 2]
            y_train[i, :] = y[i]
        a_train = np.dot(np.linalg.pinv(x_train), y_train)
        return a_train.flatten()

    @staticmethod
    def RANSAC(x: np.ndarray, y: np.ndarray, sigma=0.25, ratio=0.8) -> np.ndarray:
        size = x.size
        iters = 1E6
        a_best = np.array([])
        pretotal = 0

        P = 0.99
        iter_i = 0
        while iter_i < iters:
            sample_index = random.sample(range(size), 3)
            [x_1, x_2, x_3] = x[sample_index]
            [y_1, y_2, y_3] = y[sample_index]

            x_train = np.array([[1, x_1, x_1 ** 2],
                                [1, x_2, x_2 ** 2],
                                [1, x_3, x_3 ** 2]])
            y_train = np.array([[y_1],
                                [y_2],
                                [y_3]])
            a_train = np.dot(np.linalg.pinv(x_train), y_train).flatten()

            total_inlier = 0
            for index in range(size):
                y_estimate = a_train[0] + a_train[1] * x[index] + a_train[2] * x[index] ** 2
                if abs(y_estimate - y[index]) < sigma:
                    total_inlier = total_inlier + 1

            if total_inlier > pretotal:
                iters = math.log(1 - P) / math.log(1 - pow(total_inlier / size, 2))
                pretotal = total_inlier
                a_best = a_train.copy()

            if total_inlier >= size * ratio:
                break

            iter_i = iter_i + 1

        return a_best
