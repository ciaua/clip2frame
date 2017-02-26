import numpy as np
from sklearn import preprocessing as pp


def standardize(feat, scaler=None):
    if scaler is None:
        scaler = pp.StandardScaler().fit(feat)
    out = scaler.transform(feat)
    return out, scaler


if __name__ == '__main__':
    feat_tr_fp = '../data/data.magnatagatune/sample_exp_data/feat.tr.npy'
    feat_va_fp = '../data/data.magnatagatune/sample_exp_data/feat.va.npy'
    feat_te_fp = '../data/data.magnatagatune/sample_exp_data/feat.te.npy'

    feat_tr = np.load(feat_tr_fp)
    feat_va = np.load(feat_va_fp)
    feat_te = np.load(feat_te_fp)

    k = feat_tr.shape[-1]

    # tr
    n = feat_tr.shape[0]
    feat_tr = feat_tr.reshape((-1, k))
    feat_tr_s, scaler = standardize(feat_tr)
    feat_tr_s = feat_tr_s.reshape((n, 1, -1, k))

    # va
    n = feat_va.shape[0]
    feat_va = feat_va.reshape((-1, k))
    feat_va_s, scaler = standardize(feat_va, scaler=scaler)
    feat_va_s = feat_va_s.reshape((n, 1, -1, k))

    # te
    n = feat_te.shape[0]
    feat_te = feat_te.reshape((-1, k))
    feat_te_s, scaler = standardize(feat_te, scaler=scaler)
    feat_te_s = feat_te_s.reshape((n, 1, -1, k))
