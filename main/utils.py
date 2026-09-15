import os
import sys
import random
import gc
import numpy as np
import torch
import argparse
import warnings
import surfa as sf

warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")


def fatal(message):
    print(message)
    sys.exit(1)


def init_text_file(fname, string, check_if_exists=False):
    if fname is not None:
        if check_if_exists and os.path.isfile(fname):
            print(f'{fname} already exists!')
            return True

        f = open(fname, 'w')
        f.write(string + '\n')
        f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', default='configs/train_base.yaml',
                        help='.yaml file path to configure all parameters; default is '
                        'configs/train.yaml')
    parser.add_argument('-infer_only', '--infer_only', action='store_true',
                        help='Flag to run only inference and no training on input data')
    parser.add_argument('-num_workers', '--num_workers', type=int, default=None,
                        help='Number of workers for each data loader')
    parser.add_argument('-output_dir', '--output_dir', default=None,
                        help='directory to store all model outputs')
    parser.add_argument('-print_time', '--print_time', action='store_true',
                        help='Flag to print date/time at start and end of running (useful for '
                        'slurm)')
    parser.add_argument('-resume', '--resume', action='store_true',
                        help='Flag to resume training from model checkpoint')
    parser.add_argument('-synth_off', '--synth_off', action='store_true',
                        help='Flag to turn off synthetic data generator')
    parser.add_argument('-use_cuda', '--use_cuda', action='store_true',
                        help='Flag to use cuda for  gpu assistance (will use only cpu if not '
                        'specified')
    parser.add_argument('-use_multiple_gpus', '--use_multiple_gpus', action='store_true',
                        help='Flag to use multiple gpus (default to false if use_cuda=False')
    return parser.parse_args()


def arg_error(instr, cls=None):
    if cls is None:
        fatal(f'{instr}')
    else:
        fatal(f'Error in {cls.__class__.__name__}.__init__: {instr}')


def print_gc(exclude_params=True):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if exclude_params:
                    if 'parameter' not in str(type(obj)):
                        print(type(obj), obj.size())
                else:
                    print(type(obj), obj.size())
        except:
            pass

def print_gpu_memory_info(idx=0):
    device = torch.device('cuda', idx)
    mem_res = torch.cuda.memory_reserved(device)
    mem_alloc = torch.cuda.memory_allocated(device)
    mem_total = torch.cuda.get_device_properties(device).total_memory
    print(f'Memory reserved: {mem_res / (1024 ** 3):.2f} GB')
    print(f'Memory allocated: {mem_alloc / (1024 ** 3):.2f} GB')
    print(f'Total memory: {mem_total / (1024 ** 3):.2f} GB')


def print_model_size(model, unit=None):
    param_sz = 0.
    for param in model.parameters():
        param_sz += param.nelement() * param.element_size()

    buffer_sz = 0.
    for buffer in model.buffers():
        buffer_sz += buffer.nelement() * buffer.element_size()

    sz = param_sz + buffer_sz
    unit = unit if unit is not None else 'GB' if sz > 1e9 else 'MB' if sz > 1e6 else 'KB'
    f = 3 if unit == 'GB' else 2 if unit == 'MB' else 1
    print(f'model size: {sz / (1024 ** f):.2f} {unit}')


def print_tensor_size(x, unit=None):
    sz = x.nelement() * x.element_size()
    unit = unit if unit is not None else 'GB' if sz > 1e9 else 'MB' if sz > 1e6 else 'KB'
    f = 3 if unit == 'GB' else 2 if unit == 'MB' else 1
    print(f'{sz / (1024 ** f):.2f} {unit}')


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    os.environ["PYTHONHASHSEED"] = str(seed)


def unsqueeze_repeat(x, dims, repeats=None):
    x = x if type(x) is torch.Tensor else torch.tensor(x)
    dims = [dims] if not isinstance(dims, list) else unsqueeze_dims
    if repeats is not None and len(repeats) != len(x.shape) + len(dims):
        fatal('In unsqueeze_repeat(), len(repeats) must equal len(x.shape) + len(unsqueeze_dims)')

    for d in dims:
        x.unsqueeze(d)
    return x if repeats is None else x.repeat(repeats)


# -------------------------------------------------------------------------------------------------#
#                                        Image utilities
# -------------------------------------------------------------------------------------------------#

def largest_connected_component(x, vals=None, bgval=0):
    """
    Extracts the largest connected components for each foreground label in a
    multi-label image
    """
    x = x.cpu().numpy() if torch.is_tensor(x) else x
    vals = np.unique(x) if vals is None else vals
    vals = [i for i in vals if i != bgval]
    x_cc = np.tile(np.zeros(x.shape), (len(vals) + 1, 1, 1, 1))

    for j in range(len(vals)):
        x_j = np.squeeze(np.where(x == vals[j], 1, 0))
        x_j_cc, n_cc = ndimage.label(x_j, np.ones((3, 3, 3)))

        if n_cc > 1:
            cc_vals = np.unique(x_j_cc)[1:]
            n_cc = np.array([(x_j_cc == i).sum() for i in cc_vals])
            try:
                largest_cc_val = cc_vals[n_cc == n_cc.max()].item()
            except TypeError:
                largest_cc_val = cc_vals[np.array(n_cc == n_cc.max(), dtype=int)[0]].item()
        else:
            largest_cc_val = 1
            x_cc[j + 1, ...] = np.where(x_j_cc == largest_cc_val, vals[j], 0)

    return np.sum(x_cc, axis=0, dtype=x.dtype)


def load_volume(path,                 # Path to load
                conform=True,         # Flag to conform image
                is_int=False,         # Flag if image is int or float
                is_slice=False,       # flag for if slice or volume
                orientation='RAS',    # Output image orientation
                return_geoms=False,   # Flag to return x geometries
                shape=256,            # Output image dimensions
                to_tensor=True,       # Flag to convert sf.Volume to tensor
                voxsize=1.0):         # Output image resolution
    """
    Loads an input volume (using surfa) and conforms to a specific geometry (if
    conform=True). Returns the image as a tensor (if to_tensor=True) along with
    the original and conformed geometries.
    """
    # Load
    x = sf.load_slice(path) if is_slice else sf.load_volume(path)
    geom = x.geom

    # Conform
    n_dims = len(x.data.shape)
    shape = [shape] * n_dims if isinstance(shape, int) else shape
    
    if not is_slice:
        x = x.conform(
            shape=shape if conform else geom.shape,
            voxsize=voxsize if conform else geom.voxsize,
            orientation=orientation if conform else geom.orientation,
            dtype=np.int32 if is_int else np.float32,
            method='nearest' if is_int else 'linear'
        )
    else:
        x = x.conform(dtype=np.int32 if is_int else np.float32)

    x = (
        torch.Tensor(x.data).to(torch.int if is_int else torch.float) if to_tensor else x
    ).squeeze()

    return [x, geom, x.geom] if return_geoms else x


def min_max_norm(x, m=0, M=1, eps=1e-8):
    """
    Min-max normalization
    """    
    if type(x) is torch.Tensor:
        y = (M - m) * (x - x.min()) / (x.max() - x.min() + eps) + m
    elif type(x) is np.ndarray:
        y = (M - m) * (x - np.min(x)) / (np.max(x) - np.min(x) + eps) + m
    else:
        y = (M - m) * (x - min(x)) / (max(x) - min(x) + eps) + m
    return y


def pad_volume(x, crop_window, full_shape=(256, 256, 256)):
    pad_width = [[cw[0], fs - cw[1]] for cw, fs in zip(crop_window, full_shape)]
    return np.pad(x, pad_width=pad_width)


def replace_labels(x, labels_out, labels_in=None):
    if labels_in is None:
        labels_in = x.unique()

    if len(labels_in) != len(labels_out):
        print("labels_in:", labels_in)
        print("labels_out:", labels_out)
        fatal('len(labels_in) must equal len(labels_out)')

    y = x.clone()
    for Lin, Lout in zip(labels_in, labels_out):
        y[x == Lin] = Lout

    return y


def save_volume(x,                     # image data
                path,                  # path to save image
                input_geom=None,       # input image geometry
                conform_geom=None,     # conformed image geometry
                crop_bounds=None,      # bounds of data cropping
                label_lut=None,        # lut associated w/ image
                is_labels=False,       # flag if output is label image
                is_onehot=False,       # flag if output is onehot
                rescale=False,         # flag to rescale image intensities
                return_output=False):  # flag to return conformed output
    """
    Saves an output image with the option to first conform the image to its
    original geometry. This requires both the conform_geom and the input_geom.
    Also has the option to return the conformed output image.
    """

    # Reform image to original size/geometry
    x = (x.cpu().numpy() if torch.is_tensor(x) else x).squeeze()
    x = x.astype(np.int32 if is_labels else np.float32)
    x = pad_volume(x, crop_bounds) if crop_bounds is not None else x
    x = min_max_norm(x, 0., 255.) if rescale and not is_labels else x
    x = sf.Volume(x, geometry=conform_geom)

    if input_geom is not None:
        x = x.conform(
            shape=input_geom.shape,
            voxsize=input_geom.voxsize,
            orientation=input_geom.orientation,
            method='nearest' if is_labels else 'linear'
        )
    
    if label_lut is not None and is_labels:
        x.labels = label_lut

    # Write image
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    x.save(path)

    return x if return_output else None
