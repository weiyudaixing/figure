from __future__ import print_function  # to avoid issues between Python 2 and 3 printing
# import os
# import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import deque


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])  # floor devision
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


def split_list(xs):
    """An auxiliary function: split long list to chunks of length 20"""
    chunks = [xs[20 * i: 20 * (i + 1)] for i in range(len(xs) // 20)]
    return chunks


"""The following 4 functions are manual matrix manipulation at float128 accuracy


def transpose_matrix(m):
    return np.asarray([list(x) for x in zip(*m)], dtype=np.longdouble)


def get_matrix_minor(m, i, j):
    m = np.delete(m, i, axis=0)
    m = np.delete(m, j, axis=1)
    return m


def get_matrix_determinant(m):
    # base case for 2x2 matrix
    # m_list = [s.tolist() for s in s]
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1) ** c) * m[0][c] * get_matrix_determinant(get_matrix_minor(m, 0, c))
    return determinant


def get_matrix_inverse(m):
    determinant = get_matrix_determinant(m)
    # special case for 2x2 matrix:
    if len(m) == 2:
        return np.asarray([[m[1][1] / determinant, -1 * m[0][1] / determinant],
                           [-1 * m[1][0] / determinant, m[0][0] / determinant]], dtype=np.longdouble)

    # find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactor_row = []
        for c in range(len(m)):
            minor = get_matrix_minor(m, r, c)
            cofactor_row.append(((-1) ** (r + c)) * get_matrix_determinant(minor))
        cofactors.append(cofactor_row)
    cofactors = transpose_matrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / determinant
    return cofactors

Manual matrix manipulation end"""

"""Functions involved with solving a linear system with LU decomposition"""


def LU_decomposition(A):
    """Perform LU decomposition using the Doolittle factorisation."""
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    N = np.size(A, 0)

    for k in range(N):
        L[k, k] = 1
        U[k, k] = (A[k, k] - np.dot(L[k, :k], U[:k, k])) / L[k, k]
        for j in range(k + 1, N):
            U[k, j] = (A[k, j] - np.dot(L[k, :k], U[:k, j])) / L[k, k]
        for i in range(k + 1, N):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    return L, U


def forward_sub(L, b):
    """ Unit row oriented forward substitution """
    for i in range(L.shape[0]):
        for j in range(i):
            b[i] -= L[i, j] * b[j]
    return b


def backward_sub(U, y):
    """ Row oriented backward substitution """
    for i in range(U.shape[0] - 1, -1, -1):
        for j in range(i + 1, U.shape[1]):
            y[i] -= U[i, j] * y[j]
        y[i] = y[i] / U[i, i]
    return y


"""End of functions involved with LU decomposition"""

"""Functions implementing (extended）least squares regression"""


def square_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)


# Note linear least squares regression is a special case of polynomial least squares regression, with degree = 1
def lu_poly_least_squares(xs, ys, degree):
    # extend the first column with 1s
    xs = np.array(xs, dtype=np.longdouble)
    ys = np.array(ys, dtype=np.longdouble)

    x_e = np.ones(xs.shape)
    for i in range(1, degree + 1):
        new_col = [x ** i for x in xs]
        x_e = np.column_stack((x_e, new_col))
    # print(x_e)

    """A.dot(COEF) = ys"""
    A = x_e.T.dot(x_e)
    L, U = LU_decomposition(A)
    bv = (x_e.T).dot(ys)
    Y = forward_sub(L, bv)
    COEF = backward_sub(U, Y)

    ys_hat = np.full(xs.shape, COEF[0])
    for i in range(1, degree + 1):
        ys_hat = np.array([y_hat + COEF[i] * (x ** i) for (x, y_hat) in zip(xs, ys_hat)])

    error = square_error(ys, ys_hat)
    return error, COEF


def sin_least_squares(xs, ys, degree=1):
    xs = np.array(xs, dtype=np.longdouble)
    ys = np.array(ys, dtype=np.longdouble)

    # construct the X matrix
    # extend the first column with 1s
    x_e = np.ones(xs.shape)
    for i in range(1, degree + 1):
        new_col = [((np.sin(x)) ** i) for x in xs]
        x_e = np.column_stack((x_e, new_col))

    """A.dot(COEF) = ys"""
    """A.dot(COEF) = ys"""
    A = x_e.T.dot(x_e)
    L, U = LU_decomposition(A)
    bv = (x_e.T).dot(ys)
    Y = forward_sub(L, bv)
    COEF = backward_sub(U, Y)
    v = COEF

    # calculate y_hat
    ys_hat = np.full(xs.shape, v[0])
    for i in range(1, degree + 1):
        ys_hat = [y_hat + v[i] * ((np.sin(x)) ** i) for (x, y_hat) in zip(xs, ys_hat)]

    # calculate square error
    error = square_error(ys, ys_hat)
    # print(backward_error(xs,ys,v))
    return error, v


"""End of functions reconstruct signals using least squares solutions"""

"""A function choose which model to use
   This is a helper function used when preselect the best degree of polynomial and unknown function.
   after the polynomial degree and unknown function are fixed,
   decide_func is the right one to call for actual model decision when executing main function"""


def choose_model(xs, ys, degree=2):
    # calculate polynomial errors
    # linear error is a special case of poly error(when degree=1)
    lu_poly_errors = np.zeros(degree + 1,
                              dtype=np.longdouble)  # note the index, initial an array to store polynomial errors
    for i in range(1, degree + 1):
        lu_poly_errors[i], _ = lu_poly_least_squares(xs, ys, i)
    print("lu_polyerrors:", lu_poly_errors)

    # choose the least polynomial errors
    flag = 1
    for i in range(2, degree + 1):
        if lu_poly_errors[i] < 0.87 * lu_poly_errors[i - 1] and not (
                lu_poly_errors[i] < 1 and lu_poly_errors[flag] < 1):
            flag = i

    # print out poly-result
    print("the poly degree is: ", flag, "its square error is:", lu_poly_errors[flag])

    sin_error, sin_co = sin_least_squares(xs, ys, 1)
    print("the sin error is: ", sin_error)

    if sin_error < lu_poly_errors[flag]:
        flag = 0

    return flag


"""The following 3 functions are visualizing and testing functions when testinng the model performance myself"""
"""Not called in actual reconstruct process"""


def visual(xs, ys):
    fig0, ax0 = plt.subplots()
    ax0.scatter(xs, ys)
    plt.show()


def test_data(filename):
    this_xs, this_ys = load_points_from_file(filename)
    print(filename)
    this_xs_chunks = split_list(this_xs)
    this_ys_chunks = split_list(this_ys)
    assert (len(this_xs_chunks) == len(this_ys_chunks))
    for (xs, ys) in zip(this_xs_chunks, this_ys_chunks):
        visual(xs, ys)
        choose_model(xs, ys, 5)

    # test_data("train_data/basic_3.csv")
    # test_data("train_data/basic_4.csv")
    # test_data("train_data/basic_5.csv")

    # test_data("train_data/noise_1.csv")
    # test_data("train_data/noise_2.csv")
    # test_data("train_data/noise_3.csv")

    # test_data("train_data/adv_1.csv")
    # test_data("train_data/adv_2.csv")
    # test_data("train_data/adv_3.csv")


"""Ent of testing functions"""

"""Decide the best fit model"""


def decide_func(xs, ys):
    errors = np.empty(3, dtype=np.longdouble)
    co_eff_deq = deque([])
    threshold = 0.87

    # calculate linear error:
    errors[1], linear_co_eff = lu_poly_least_squares(xs, ys, degree=1)
    co_eff_deq.append(linear_co_eff)
    # print(co_eff_deq)
    # print("linear error: ", errors[1])
    flag = 1

    # calculate poly error:
    errors[2], poly_co_eff = lu_poly_least_squares(xs, ys, degree=2)
    # print(poly_co_eff)
    co_eff_deq.append(poly_co_eff)
    # print(co_eff_deq)
    # print("poly error: ", errors[2])
    if (errors[2] < threshold * errors[1]) and not (
            errors[2] < 1 and errors[1] < 1):
        flag = 2

    # calculate sin error
    errors[0], sin_co_eff = sin_least_squares(xs, ys)
    co_eff_deq.appendleft(sin_co_eff)
    # print(co_eff_deq)
    # print("sin error: ", errors[0])
    if errors[0] < errors[flag]:
        flag = 0

    # print("function flag: ", flag)
    """ flag = 0:sin
        flag = 1:linear
        flag = 2:degree 2 polynomial
    """
    return flag, errors[flag], np.array(co_eff_deq[flag], dtype=np.longdouble)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="read in a data file and plot it if required.",
        add_help=True
    )
    parser.add_argument(
        "-p", "--plot",
        action="store_true",
        help="plot the figure of the input file"
    )
    parser.add_argument(
        'filename',
        nargs='?', type=str,
        help="the name of the input file"
    )
    return parser


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()

    if not args.filename:
        print("Please provide file name")
    else:
        file_name = args.filename
        xs_all, ys_all = load_points_from_file(file_name)
        this_xs_chunks = split_list(xs_all)
        this_ys_chunks = split_list(ys_all)
        n = len(this_xs_chunks)
        flags = np.empty(n)
        errors = np.empty(n, dtype=np.longdouble)
        co_effs = []
        index = 0  # every 20-length chunk has a unique index
        for (xs, ys) in zip(this_xs_chunks, this_ys_chunks):
            # visual(xs, ys)
            flags[index], errors[index], co_eff = decide_func(xs, ys)
            # print(flags[index])
            # print(co_eff)
            co_effs.append(co_eff)
            # print(co_effs)

            if args.plot:
                xs_plot = np.linspace(xs.min(), xs.max(), 100)
                if flags[index] == 1:
                    # print(1)
                    ys_plot = co_eff[0] + co_eff[1] * xs_plot
                elif flags[index] == 2:
                    # print(2)
                    ys_plot = co_eff[0] + co_eff[1] * xs_plot + co_eff[2] * (xs_plot ** 2)
                else:
                    # print(0)
                    ys_plot = co_eff[0] + co_eff[1] * np.sin(xs_plot)

                # setting the axes at the centre
                fig, ax = plt.subplots()
                ax.scatter(xs, ys, color='b')
                ax.plot(xs_plot, ys_plot, 'r')
                plt.show()
            index = index + 1

        print(np.sum(errors))


if __name__ == "__main__":
    main()
