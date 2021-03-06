import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpointpath', type=str, default='/gpfsscratch/rech/rnt/uuj49ar/inpainting/results/no_global/phase_3_only/model_cn_step400000')
    #parser.add_argument('--checkpointpath', type=str, default=
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--niter', default=1, type=int, help="number of iterations")
    parser.add_argument('--data_dir', default='/gpfsscratch/rech/rnt/uuj49ar/test_partial_img_align_celeba')
    parser.add_argument('--comparison_data_dir', default='../../results_celebA')
    parser.add_argument('--model_to_compare', default='channel44_cn', choices=['channel0_cn', 'channel44_cn', 'cn', 'no_local_cn', 'no_global_cn', 'patch7'])
    parser.add_argument('--output_folder', default='evaluations/')
    parser.add_argument('--data_parallel', action='store_true')
    parser.add_argument('--recursive_search', action='store_true', default=False)
    parser.add_argument('--max_holes', type=int, default=1)
    parser.add_argument('--hole_min_w', type=int, default=48)
    parser.add_argument('--hole_max_w', type=int, default=96)
    parser.add_argument('--hole_min_h', type=int, default=48)
    parser.add_argument('--hole_max_h', type=int, default=96)
    parser.add_argument('--cn_input_size', type=int, default=160)
    parser.add_argument('--layer_idx', type=int, default=100)
    parser.add_argument('--ld_input_size', type=int, default=96)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--comparison_batch_size', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--mpv', nargs=3, type=float, default=1)
    parser.add_argument('--alpha', type=float, default=4e-4)
    parser.add_argument('--arc', type=str, choices=['celeba', 'places2'], default='celeba')
    opt = parser.parse_args()
    parameters = {key: val for key, val in vars(opt).items() if val is not None}
    return parameters

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    img = plt.imread('../../../preprocessed_celebA/003513/composite.png')
    print(img)