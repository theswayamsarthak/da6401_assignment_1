import numpy as np


def load_dataset(name):
    name = name.lower()
    if name == "mnist":
        from keras.datasets import mnist
        (X_train_all, y_train_all), (X_test, y_test) = mnist.load_data()
    elif name in ("fashion", "fashion_mnist", "fashion-mnist"):
        from keras.datasets import fashion_mnist
        (X_train_all, y_train_all), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"dataset should be 'mnist' or 'fashion', got '{name}'")

    # flatten 28x28 -> 784 and normalize to [0,1]
    X_train_all = X_train_all.reshape(len(X_train_all), -1).astype(np.float64) / 255.0
    X_test = X_test.reshape(len(X_test), -1).astype(np.float64) / 255.0

    # hold out 10% for validation
    n_val = int(0.1 * len(X_train_all))
    X_val, y_val = X_train_all[:n_val], y_train_all[:n_val]
    X_train, y_train = X_train_all[n_val:], y_train_all[n_val:]

    print(f"loaded {name}: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def get_batches(X, y, batch_size, shuffle=True):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], y[batch]
