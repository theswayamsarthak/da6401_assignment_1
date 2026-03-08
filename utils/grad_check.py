import numpy as np


def numerical_gradient(loss_fn, W, eps=1e-5):
    # central differences: (f(w+eps) - f(w-eps)) / 2eps
    grad = np.zeros_like(W)
    it = np.nditer(W, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        orig = W[ix]

        W[ix] = orig + eps
        lp = loss_fn()
        W[ix] = orig - eps
        lm = loss_fn()
        W[ix] = orig  # restore

        grad[ix] = (lp - lm) / (2 * eps)
        it.iternext()
    return grad


def gradient_check(model, X, y_oh, layer_idx=0, eps=1e-5, tol=1e-7):
    # run backward to get analytic grads
    model.backward(X, y_oh, weight_decay=0.0)
    analytic_W = model.layers[layer_idx].grad_W.copy()
    analytic_b = model.layers[layer_idx].grad_b.copy()

    layer = model.layers[layer_idx]

    def loss_fn():
        return model.loss_fn.forward(model.forward(X), y_oh)

    num_W = numerical_gradient(loss_fn, layer.W, eps)
    num_b = numerical_gradient(loss_fn, layer.b, eps)

    # relative error, small constant in denominator to handle near-zero case
    def rel_err(a, n):
        return np.max(np.abs(a - n) / (np.maximum(np.abs(a), np.abs(n)) + 1e-8))

    eW = rel_err(analytic_W, num_W)
    eb = rel_err(analytic_b, num_b)

    print(f"layer {layer_idx}  grad_W rel err: {eW:.2e}  {'PASS' if eW < tol else 'FAIL'}")
    print(f"layer {layer_idx}  grad_b rel err: {eb:.2e}  {'PASS' if eb < tol else 'FAIL'}")

    return eW, eb
