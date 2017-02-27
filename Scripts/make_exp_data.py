import os
import numpy as np
from clip2frame import utils

floatX = 'float32'


def make_batch_feat(feat_fp_list, length):
    feat = [
        _append_zero_row(
            np.load(term), length)[None, None, :length, :].astype(floatX)
        for term in feat_fp_list]
    '''
    feat = [np.load(term)[None, None, :length, :].astype(floatX)
            for term in feat_fp_list]
    '''

    feat = np.vstack(feat)
    return feat


def make_batch_target(target_fp_list):
    target = [np.load(term).astype(floatX)
              for term in target_fp_list]
    target = np.vstack(target)
    return target


def _append_zero_row(array, n_total_row):
    r, c = array.shape
    if r >= n_total_row:
        return array
    else:
        temp = np.zeros((n_total_row-r, c))
        return np.vstack([array, temp])


def make_magna_fixed_length(anno_dir, feat_dir, length=911):
    char_tr = map(str, range(10)) + ['a', 'b']
    char_va = ['c']
    char_te = ['d', 'e', 'f']

    fn_tr_list = []
    fn_va_list = []
    fn_te_list = []
    for root, dirs, files in os.walk(anno_dir):
        for in_fn in files:
            if in_fn.endswith('.npy'):
                char = root[-1]
                in_fp = os.path.join(root, in_fn)
                fn_ = in_fp.replace(anno_dir, '')[1:]
                if char in char_tr:
                    fn_tr_list.append(fn_)
                elif char in char_te:
                    fn_te_list.append(fn_)
                elif char in char_va:
                    fn_va_list.append(fn_)
    fn_tr_list.sort()
    fn_va_list.sort()
    fn_te_list.sort()

    print(len(fn_tr_list), len(fn_te_list), len(fn_va_list))

    fn_list = fn_tr_list
    # print(feat_dir)
    # print(os.path.join(feat_dir, fn_list[0]))
    temp_feat_fp_list = [os.path.join(feat_dir, term) for term in fn_list]
    # print(feat_dir)
    # print(temp_feat_fp_list[0])
    # raw_input(123)
    for fp in temp_feat_fp_list:
        # print(fp)
        if not os.path.exists(fp):
            fn_list.remove(fp.replace(feat_dir, '')[1:])
    # print(len(fn_list))
    anno_fp_list = [os.path.join(anno_dir, fn) for fn in fn_list]
    feat_fp_list = [os.path.join(feat_dir, fn) for fn in fn_list]
    feat = make_batch_feat(feat_fp_list, length).astype(floatX)
    target = make_batch_target(anno_fp_list).astype(floatX)
    feat_tr = feat
    target_tr = target
    fn_tr = fn_list

    fn_list = fn_va_list
    temp_feat_fp_list = [os.path.join(feat_dir, term) for term in fn_list]
    for fp in temp_feat_fp_list:
        if not os.path.exists(fp):
            fn_list.remove(fp.replace(feat_dir, '')[1:])
    anno_fp_list = [os.path.join(anno_dir, fn) for fn in fn_list]
    feat_fp_list = [os.path.join(feat_dir, fn) for fn in fn_list]
    feat = make_batch_feat(feat_fp_list, length).astype(floatX)
    target = make_batch_target(anno_fp_list).astype(floatX)
    feat_va = feat
    target_va = target
    fn_va = fn_list

    fn_list = fn_te_list
    temp_feat_fp_list = [os.path.join(feat_dir, term) for term in fn_list]
    for fp in temp_feat_fp_list:
        if not os.path.exists(fp):
            fn_list.remove(fp.replace(feat_dir, '')[1:])
    anno_fp_list = [os.path.join(anno_dir, fn) for fn in fn_list]
    feat_fp_list = [os.path.join(feat_dir, fn) for fn in fn_list]
    feat = make_batch_feat(feat_fp_list, length).astype(floatX)
    target = make_batch_target(anno_fp_list).astype(floatX)
    feat_te = feat
    target_te = target
    fn_te = fn_list

    return target_tr, feat_tr, fn_tr, target_va, feat_va, fn_va, \
        target_te, feat_te, fn_te


if __name__ == '__main__':
    feat_type = "logmelspec10000.16000_512_512_128.0.raw"

    # The number of frames
    length = 911  # sr=16000, hop=512

    # Number of top tags to be used in the experiment
    n_top = 188

    # Point to the feature and annotation directories
    base_dir = "../Output"
    feat_dir = os.path.join(base_dir, 'feature', feat_type)
    anno_dir = os.path.join(base_dir, 'annotation.top{}'.format(n_top))

    # Output directory
    out_dir = os.path.join(base_dir, 'exp_data', 'top{}'.format(n_top),
                           feat_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_feat_tr_fp = os.path.join(out_dir, 'feat.tr.npy')
    out_target_tr_fp = os.path.join(out_dir, 'target.tr.npy')
    out_fn_tr_fp = os.path.join(out_dir, 'fn.tr.txt')
    out_feat_va_fp = os.path.join(out_dir, 'feat.va.npy')
    out_target_va_fp = os.path.join(out_dir, 'target.va.npy')
    out_fn_va_fp = os.path.join(out_dir, 'fn.va.txt')
    out_feat_te_fp = os.path.join(out_dir, 'feat.te.npy')
    out_target_te_fp = os.path.join(out_dir, 'target.te.npy')
    out_fn_te_fp = os.path.join(out_dir, 'fn.te.txt')

    # loading data
    target_tr, feat_tr, fn_tr, target_va, feat_va, fn_va, \
        target_te, feat_te, fn_te = \
        make_magna_fixed_length(anno_dir, feat_dir, length=length)
    print(feat_tr.shape, feat_va.shape, feat_te.shape)

    np.save(out_feat_tr_fp, feat_tr)
    np.save(out_target_tr_fp, target_tr)
    np.save(out_feat_va_fp, feat_va)
    np.save(out_target_va_fp, target_va)
    np.save(out_feat_te_fp, feat_te)
    np.save(out_target_te_fp, target_te)

    utils.write_lines(out_fn_tr_fp, fn_tr)
    utils.write_lines(out_fn_va_fp, fn_va)
    utils.write_lines(out_fn_te_fp, fn_te)
