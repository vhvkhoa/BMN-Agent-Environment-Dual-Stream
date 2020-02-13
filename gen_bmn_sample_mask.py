import time
import math
import numpy as np
import argparse
from tqdm import tqdm

from config.defaults import get_cfg


def get_interp1d_bin_mask(seg_xmin, seg_xmax, temporal_dim, num_sample, num_sample_perbin):
    # generate sample mask for a boundary-matching pair
    plen = float(seg_xmax - seg_xmin)
    plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
    total_samples = [
        seg_xmin + plen_sample * ii
        for ii in range(num_sample * num_sample_perbin)
    ]

    p_mask = []
    for idx in range(num_sample):
        bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
        bin_vector = np.zeros([temporal_dim])
        for sample in bin_samples:
            sample_upper = math.ceil(sample)
            sample_decimal, sample_down = math.modf(sample)
            if int(sample_down) <= (temporal_dim - 1) and int(sample_down) >= 0:
                bin_vector[int(sample_down)] += 1 - sample_decimal
            if int(sample_upper) <= (temporal_dim - 1) and int(sample_upper) >= 0:
                bin_vector[int(sample_upper)] += sample_decimal
        bin_vector = 1.0 / num_sample_perbin * bin_vector
        p_mask.append(bin_vector)
    p_mask = np.stack(p_mask, axis=1)
    return p_mask


def get_interp1d_mask(cfg, temporal_dim):
    # generate sample mask for each point in Boundary-Matching Map
    mask_mat = []

    for start_index in tqdm(range(temporal_dim)):
        mask_mat_vector = []

        for duration_index in range(temporal_dim):
            if start_index + duration_index < temporal_dim:
                p_xmin = start_index
                p_xmax = start_index + duration_index
                center_len = float(p_xmax - p_xmin) + 1
                sample_xmin = p_xmin - center_len * cfg.BMN.PROP_BOUNDARY_RATIO
                sample_xmax = p_xmax + center_len * cfg.BMN.PROP_BOUNDARY_RATIO

                p_mask = get_interp1d_bin_mask(
                    sample_xmin, sample_xmax, temporal_dim, cfg.BMN.NUM_SAMPLES,
                    cfg.BMN.NUM_SAMPLES_PER_BIN)

            else:
                p_mask = np.zeros([temporal_dim, cfg.BMN.NUM_SAMPLES])

            mask_mat_vector.append(p_mask)

        mask_mat_vector = np.stack(mask_mat_vector, axis=2)
        mask_mat.append(mask_mat_vector)

    mask_mat = np.stack(mask_mat, axis=3).astype(np.float32)
    return mask_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_cfg()
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    start_time = time.time()
    sample_mask = get_interp1d_mask(cfg, cfg.DATA.MAX_TEMPORAL_DIM)

    print('Sample mask for %d temporal dimensions takes %f secs.' % (cfg.DATA.MAX_TEMPORAL_DIM, time.time() - start_time))
    np.save(cfg.DATA.SAMPLE_MASK_FILE, sample_mask)

    '''
    for tmp_scale in range(990, cfg.DATA.MAX_TEMPORAL_DIM):
        start_time = time.time()
        test_sample_mask_interp = get_interp1d_mask(cfg, tmp_scale)
        interp_time = time.time() - start_time

        start_time = time.time()
        test_sample_mask = np.zeros((tmp_scale, cfg.BMN.NUM_SAMPLES, tmp_scale, tmp_scale))
        for idx in range(tmp_scale):
            end_idx = tmp_scale - idx
            test_sample_mask[:, :, :end_idx, idx] = sample_mask[:tmp_scale, :, :end_idx, idx]
        reconstruct_time = time.time() - start_time

        print(tmp_scale, '\t', np.array_equal(test_sample_mask, test_sample_mask_interp), reconstruct_time / interp_time)
    '''
