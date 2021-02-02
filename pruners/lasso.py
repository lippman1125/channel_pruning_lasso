import numpy as np
import time
from sklearn.linear_model import Lasso, LassoLars, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def lasso_pruning(X, Y, W, c_new, alpha=1e-4, tolerance=0.02, debug=False):
    # Conv
    # Example: B is sample number
    #        : c_in is channel input
    #        : c_out is channel output
    #        : 3x3 is kernel size
    # X shape: [B, c_in, 3, 3]
    # Y shape: [B, c_out]
    # W shape: [c_out, c_in, 3, 3]
    # Linear
    # X shape: [B, c_in]
    # Y shape: [B, c_out]
    # W shape: [c_out, c_in]
    if debug:
        print("input shape: {}".format(X.shape))
        print("output shape: {}".format(Y.shape))
        print("weight shape: {}".format(W.shape))
        print("curr chn: {} target chn: {}".format(W.shape[1], c_new))
    num_samples = X.shape[0]  # num of training samples
    c_in = W.shape[1]  # num of input channels
    c_out = W.shape[0]  # num of output channels

    # conv
    if len(W.shape) == 4:
        # sample and reshape X to [c_in, B, 9]
        reshape_X = X.reshape((num_samples, c_in, -1)).transpose((1, 0, 2))
        # reshape W to [c_in, 9, c_out]
        reshape_W = W.reshape((c_out, c_in, -1)).transpose((1, 2, 0))
    else:
        # linear
        # sample and reshape X to [c_in, B] and expand to [c_in, B, 1]
        reshape_X = X.transpose((1, 0))[...,np.newaxis]
        # reshape to [c_in, 1, c_out]
        reshape_W =  W.reshape((c_out, c_in, 1)).transpose((1, 2, 0))

    # reshape Y to [B x c_out]
    reshape_Y = Y.reshape(-1)

    # product has size [B x c_out, c_in]
    product = np.matmul(reshape_X, reshape_W).reshape((c_in, -1)).T

    # use LassoLars because it's more robust than Lasso
    solver = LassoLars(alpha=alpha, fit_intercept=False, max_iter=3000)

    # solver = Lasso(alpha=alpha, fit_intercept=False,
    #                max_iter=3000, warm_start=True, selection='random')

    def solve(alpha):
        """ Solve the Lasso"""
        solver.alpha = alpha
        solver.fit(product, reshape_Y)
        nonzero_inds = np.where(solver.coef_ != 0.)[0]
        nonzero_num = sum(solver.coef_ != 0.)
        return nonzero_inds, nonzero_num, solver.coef_

    tic = time.perf_counter()


    left = 0  # minimum alpha is 0, which means don't use lasso regularizer at all
    right = alpha

    # the left bound of num of selected channels
    lbound = c_new
    # the right bound of num of selected channels
    rbound = c_new + tolerance * c_new

    # increase alpha until the lasso can find a selection with size < c_new
    while True:
        _, keep_num, coef = solve(right)
        if debug:
            print("relax right to %.6f" % right)
            print("expected %d channels, but got %d channels" % (c_new, keep_num))
        if keep_num < c_new:
            break
        else:
            right *= 2

    # shrink the alpha for less aggressive lasso regularization
    # if the selected num of channels is less than the lbound
    while True:
        # binary search
        alpha = (left + right) / 2
        keep_inds, keep_num, coef = solve(alpha)
        # print loss
        # product has size [B x c_out, c_in]
        loss = 1 / (2 * float(product.shape[0])) * \
               np.sqrt(np.sum((reshape_Y - np.matmul(product, coef)) ** 2, axis=0)) + \
               alpha * np.sum(np.fabs(coef))

        if debug:
            print('loss: %.6f, alpha: %.6f, feature nums: %d, '
                  'left: %.6f, right: %.6f, left_bound: %.6f, right_bound: %.6f' %
                  (loss, alpha, keep_num, left, right, lbound, rbound))

        if keep_num > rbound:
            left=alpha
        elif keep_num < lbound:
            right=alpha
        else:
            break

        if alpha < 1e-10:
            break

    toc = time.perf_counter()
    if debug:
        print('Lasso Regression time: %.2f s' % (toc - tic))
        print('Chn keep idx: {}'.format(keep_inds))
        print(c_new, keep_num)
    return keep_inds, keep_num