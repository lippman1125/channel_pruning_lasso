import numpy as np
import time
from sklearn.linear_model import Lasso, LassoLars, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def l1norm_pruning(X, Y, W, c_new, debug=False):
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

    keep_num = c_new
    # conv
    if len(W.shape) == 4:
        # Filter_NUM Channel_NUM H W
        filter_norm = np.sum(W, (1,2,3))
    else:
        # linear
        # output_dims input_dims
        filter_norm = np.sum(W, axis=0).transpose()


    keep_inds = np.argsort(-filter_norm)[:keep_num]

    if debug:
        print("Filter norm: {}".format(filter_norm))
        print('Chn keep idx: {}'.format(keep_inds))
        print(c_new, keep_num)
    return keep_inds, keep_num