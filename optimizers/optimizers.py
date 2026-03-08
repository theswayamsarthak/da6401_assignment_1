import numpy as np


class BaseOptimizer:
    def __init__(self, learning_rate=0.001, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.t = 0  # step counter, needed for bias correction in adam/nadam

    def update(self, layers):
        raise NotImplementedError

    def step(self, layers):
        self.t += 1
        self.update(layers)


class SGD(BaseOptimizer):
    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b


class Momentum(BaseOptimizer):
    def __init__(self, learning_rate=0.001, weight_decay=0.0, beta=0.9):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.vW = {}
        self.vb = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.vW:
                self.vW[i] = np.zeros_like(layer.W)
                self.vb[i] = np.zeros_like(layer.b)

            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * layer.grad_b

            layer.W -= self.lr * (self.vW[i] + self.weight_decay * layer.W)
            layer.b -= self.lr * self.vb[i]


class NAG(BaseOptimizer):
    # nesterov - look ahead before computing gradient direction
    def __init__(self, learning_rate=0.001, weight_decay=0.0, beta=0.9):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.vW = {}
        self.vb = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.vW:
                self.vW[i] = np.zeros_like(layer.W)
                self.vb[i] = np.zeros_like(layer.b)

            prev_vW = self.vW[i].copy()
            prev_vb = self.vb[i].copy()

            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * layer.grad_b

            layer.W -= self.lr * ((1 + self.beta) * self.vW[i] - self.beta * prev_vW
                                  + self.weight_decay * layer.W)
            layer.b -= self.lr * ((1 + self.beta) * self.vb[i] - self.beta * prev_vb)


class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate=0.001, weight_decay=0.0, beta=0.9, eps=1e-8):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.eps = eps
        self.sW = {}
        self.sb = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.sW:
                self.sW[i] = np.zeros_like(layer.W)
                self.sb[i] = np.zeros_like(layer.b)

            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * layer.grad_W**2
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * layer.grad_b**2

            layer.W -= self.lr * (layer.grad_W / (np.sqrt(self.sW[i]) + self.eps)
                                  + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.sb[i]) + self.eps)


class Adam(BaseOptimizer):
    def __init__(self, learning_rate=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # first and second moment estimates
        self.mW = {}
        self.vW = {}
        self.mb = {}
        self.vb = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.mW:
                self.mW[i] = np.zeros_like(layer.W)
                self.vW[i] = np.zeros_like(layer.W)
                self.mb[i] = np.zeros_like(layer.b)
                self.vb[i] = np.zeros_like(layer.b)

            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.grad_W
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * layer.grad_W**2
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.grad_b
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * layer.grad_b**2

            # bias correction
            mW_hat = self.mW[i] / (1 - self.beta1**self.t)
            vW_hat = self.vW[i] / (1 - self.beta2**self.t)
            mb_hat = self.mb[i] / (1 - self.beta1**self.t)
            vb_hat = self.vb[i] / (1 - self.beta2**self.t)

            layer.W -= self.lr * (mW_hat / (np.sqrt(vW_hat) + self.eps)
                                  + self.weight_decay * layer.W)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


class Nadam(BaseOptimizer):
    # adam with nesterov momentum - usually a bit better than plain adam
    def __init__(self, learning_rate=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mW = {}
        self.vW = {}
        self.mb = {}
        self.vb = {}

    def update(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.mW:
                self.mW[i] = np.zeros_like(layer.W)
                self.vW[i] = np.zeros_like(layer.W)
                self.mb[i] = np.zeros_like(layer.b)
                self.vb[i] = np.zeros_like(layer.b)

            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.grad_W
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * layer.grad_W**2
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.grad_b
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * layer.grad_b**2

            mW_hat = self.mW[i] / (1 - self.beta1**self.t)
            vW_hat = self.vW[i] / (1 - self.beta2**self.t)
            mb_hat = self.mb[i] / (1 - self.beta1**self.t)
            vb_hat = self.vb[i] / (1 - self.beta2**self.t)

            # nesterov lookahead on the moment estimate
            mW_nesterov = self.beta1 * mW_hat + (1 - self.beta1) * layer.grad_W / (1 - self.beta1**self.t)
            mb_nesterov = self.beta1 * mb_hat + (1 - self.beta1) * layer.grad_b / (1 - self.beta1**self.t)

            layer.W -= self.lr * (mW_nesterov / (np.sqrt(vW_hat) + self.eps)
                                  + self.weight_decay * layer.W)
            layer.b -= self.lr * mb_nesterov / (np.sqrt(vb_hat) + self.eps)


def get_optimizer(name, **kwargs):
    name = name.lower()
    opts = {
        "sgd": SGD,
        "momentum": Momentum,
        "nag": NAG,
        "rmsprop": RMSProp,
        "adam": Adam,
        "nadam": Nadam,
    }
    if name not in opts:
        raise ValueError(f"unknown optimizer '{name}'. pick from: {list(opts.keys())}")
    return opts[name](**kwargs)
