import numpy as np


def start(
        sequence: list,
        p: int,
        error: int,
        max_iter: int,
        m: int,
        alpha: float,
        predict: int,
        code_for_learning: str,
        code_for_training: str,
):
    if sequence == None:
        sequence = list(
            map(int, input("Enter sequence, split numbers by space:\n").split())
        )
    q = len(sequence)
    if p == None:
        p = int(input("Enter window size:\n"))
    if p > q:
        raise ("Invalid size of window, must be less then q")
    if error == None:
        error = int(input("Enter max learning error:\n"))
    if max_iter == None:
        max_iter = int(input("Enter max number of iterations:\n"))
    if m == None:
        m = int(input("Enter number of neurons on second layer:\n"))
    if alpha == None:
        alpha = int(input("Enter learning step:\n"))
    if predict == None:
        predict = int(input("Enter count of numbers to predict:\n"))
    if code_for_learning == None:
        code_for_learning = input(
            "Enter learning code:\n"
        )  # on\off for first|on\off for others
    if code_for_training == None:
        code_for_training = input(
            "Enter training code:\n"
        )  # on\off for first|on\off for others

    x = []
    y = []
    i = 0
    while i + p < q:
        x.append(sequence[i: i + p])
        y.append(sequence[i + p])
        i += 1
    y = np.array(y)
    x = np.array(x)
    return run(x, y, p, q, error, max_iter, m, alpha, predict, code_for_learning, code_for_training)


def run(
        x: np.array,
        y: np.array,
        p: int,
        q: int,
        error: int,
        max_iter: int,
        m: int,
        alpha: float,
        predict: int,
        code_for_learning: str,
        code_for_training: str,
):
    error_all = 0
    k = 0
    if code_for_learning[0] == "1":
        context = np.zeros((x.shape[0], m))
    else:
        context = np.random.rand(x.shape[0], m)
    x = np.concatenate((x, context), axis=1)
    # reshape x matrix to make all samples matrixes (4, 1), not vector (4, )
    x = x.reshape(x.shape[0], 1, x.shape[1])
    w1 = (np.random.rand(p + m, m) * 2 - 1) / 10
    w2 = (np.random.rand(m, 1) * 2 - 1) / 10