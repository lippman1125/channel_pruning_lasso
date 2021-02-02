import numpy as np
import time
from sklearn.linear_model import Lasso, LassoLars, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def weight_reconstruction(X, Y, W, keep_inds, debug=False):
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
        print("curr chn: {} target chn: {}".format(W.shape[1], len(keep_inds)))
    num_samples = X.shape[0]  # num of training samples
    c_in = W.shape[1]  # num of input channels
    c_out = W.shape[0]  # num of output channels
    d_prime = len(keep_inds)


    reg = LinearRegression(fit_intercept=False)

    tic = time.perf_counter()
    # conv
    if len(W.shape) == 4:
        # sample and reshape X to [B, c_in, 9]
        k_h, k_w = W.shape[2], W.shape[3]
        X_mask = X[:, keep_inds, :].reshape(-1, d_prime*k_h*k_w)
        w_reg = reg.fit(X_mask, Y)
        rec_weight = w_reg.coef_.reshape(-1, d_prime, k_h, k_w)
    else:
        # linear
        X_mask = X[:, keep_inds]
        w_reg = reg.fit(X_mask, Y)
        rec_weight = w_reg.coef_.reshape(-1, d_prime)  # (C_out, C_in')

    toc = time.perf_counter()
    if debug:
        print('Linear Regression time: %.2f s' % (toc - tic))
        print("reconstruction weight shape: {}".format(rec_weight.shape))
        # r2 score: the higher the better
        # r2 = 1 - MSE/VAR
        print("r2 score: {}".format(r2_score(w_reg.predict(X_mask), Y)))
        print("RME: {}".format(mean_squared_error(w_reg.predict(X_mask), Y)))

    return rec_weight