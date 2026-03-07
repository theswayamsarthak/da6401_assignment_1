import numpy as np


def _softmax(z):
    # shift by row max before exp to keep things numerically stable
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class CrossEntropyLoss:
    # standard softmax + cross entropy, combined so we don't compute softmax twice
    def __call__(self, logits, y):
        return self.forward(logits, y)

    def forward(self, logits, y):
        probs = _softmax(logits)
        probs = np.clip(probs, 1e-12, 1.0)  # avoid log(0)
        return float(-np.mean((y * np.log(probs)).sum(axis=1)))

    def backward(self, logits, y):
        # grad of CE loss w.r.t. logits simplifies nicely to (p - y)/N
        probs = _softmax(logits)
        return (probs - y) / logits.shape[0]


class MeanSquaredErrorLoss:
    # MSE over softmax outputs - not the best for classification but the assignment asks for it
    def __call__(self, logits, y):
        return self.forward(logits, y)

    def forward(self, logits, y):
        p = _softmax(logits)
        return float(np.mean(((p - y)**2).sum(axis=1)))

    def backward(self, logits, y):
        p = _softmax(logits)
        diff = p - y
        # need full chain rule through softmax jacobian here
        # dL/dz_i = sum_j dL/dp_j * dp_j/dz_i
        # dp_j/dz_i = p_i*(1(i==j) - p_j)
        dl_dp = 2.0 * diff / logits.shape[0]
        # trick: sum_j(dl_dp_j * p_j) appears in every term
        weighted = (dl_dp * p).sum(axis=1, keepdims=True)
        return p * (dl_dp - weighted)


def get_loss(name):
    name = name.lower()
    mapping = {
        "cross_entropy": CrossEntropyLoss,
        "mean_squared_error": MeanSquaredErrorLoss,
    }
    if name not in mapping:
        raise ValueError(f"unknown loss '{name}'. options: {list(mapping.keys())}")
    return mapping[name]()
