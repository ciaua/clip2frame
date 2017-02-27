import os
import numpy as np
from clip2frame import utils

# Path to the MagnaTagATune files downloaded from
# http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
src_dir = '../Magnatagatune/Source/'

# Root output directory
base_dir = '../Output'

# Number of top tags to be used
n_top = 188

anno_fp = os.path.join(src_dir, 'annotation', 'annotations_final.csv')

out_dir = os.path.join(base_dir, 'annotation.top{}'.format(n_top))
for term in range(10)+['a', 'b', 'c', 'd', 'e', 'f']:
    temp_dir = os.path.join(out_dir, str(term))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

label_list_fp = os.path.join(base_dir, 'labels.top{}.txt'.format(n_top))

anno_raw = utils.read_tsv(anno_fp)

anno_sum = np.zeros((1, 188))
for term in anno_raw[1:]:
    anno = np.array([utils.to_int(term[1:-1])])
    anno_sum += anno
    # np.save(out_fp, anno)
    # raw_input(123)

idx = sorted(range(188), key=lambda x: anno_sum[0][x], reverse=True)[:n_top]

label_list = np.array(anno_raw[0][1:-1])
top_label_list = label_list[idx]
print(idx)
print(anno_sum[0][idx])
print(top_label_list)

for term in anno_raw[1:]:
    out_fn = term[-1].replace('.mp3', '.npy')
    out_fp = os.path.join(out_dir, out_fn)
    anno = [np.array(utils.to_int(term[1:-1]))[idx]]
    if np.sum(anno) == 0:
        continue
    else:
        np.save(out_fp, anno)

utils.write_lines(label_list_fp, top_label_list)
