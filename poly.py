import numpy as np


class PolynomialRegression:
    def __init__(self, x_vector, y_vector):
        self.x_vector = np.array(x_vector)
        self.y_vector = np.array(y_vector)

        self.w0 = 0
        self.w1 = 0
        self.w2 = 0

        self.m0 = self.v0 = self.m1 = self.v1 = self.m2 = self.v2 = 0

    def _get_predicted_value_matrix(self, x):
        formula = lambda x: self.w0 + self.w1 * x + self.w2 * (x**2)
        return formula(x)

    def _get_gradient_matrix(self):
        gradient_matrix = [
            sum((self.y_vector - self._get_predicted_value_matrix(self.x_vector))),
            sum(
                (self.y_vector - self._get_predicted_value_matrix(self.x_vector))
                * self.x_vector
            ),
            sum(
                (self.y_vector - self._get_predicted_value_matrix(self.x_vector))
                * (self.x_vector**2)
            ),
        ]

        gradient_matrix = np.array(gradient_matrix) * -2

        return gradient_matrix

    def fit(
        self,
        step_size=0.00000001,
        num_iterations=100000,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        for i in range(1, num_iterations + 1):
            gradient_matrix = self._get_gradient_matrix()

            self.m0 = beta1 * self.m0 + (1 - beta1) * gradient_matrix[0]
            self.v0 = beta2 * self.v0 + (1 - beta2) * (gradient_matrix[0] ** 2)

            self.m1 = beta1 * self.m1 + (1 - beta1) * gradient_matrix[1]
            self.v1 = beta2 * self.v1 + (1 - beta2) * (gradient_matrix[1] ** 2)

            self.m2 = beta1 * self.m2 + (1 - beta1) * gradient_matrix[2]
            self.v2 = beta2 * self.v2 + (1 - beta2) * (gradient_matrix[2] ** 2)

            m0_hat = self.m0 / (1 - beta1**i)
            v0_hat = self.v0 / (1 - beta2**i)

            m1_hat = self.m1 / (1 - beta1**i)
            v1_hat = self.v1 / (1 - beta2**i)

            m2_hat = self.m2 / (1 - beta1**i)
            v2_hat = self.v2 / (1 - beta2**i)

            self.w0 -= step_size * m0_hat / (np.sqrt(v0_hat) + epsilon)
            self.w1 -= step_size * m1_hat / (np.sqrt(v1_hat) + epsilon)
            self.w2 -= step_size * m2_hat / (np.sqrt(v2_hat) + epsilon)

    def _show_coefficients(self):
        print(f"w0: {self.w0}\tw1: {self.w1}\tw2: {self.w2}\t")

    def predict(self, x):
        y = self.w0 + self.w1 * x + self.w2 * (x**2)
        return y


x = list(range(-12, 12))
f = lambda x: (x**2) + 3 * x + 7
y = [f(x_val) for x_val in x]
model = PolynomialRegression(x, y)
model.fit(num_iterations=100000)
model._show_coefficients()
