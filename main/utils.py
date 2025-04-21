import os
import sys
import random
import numpy as np
import torch
import argparse
import surfa as sf



def check_config(config:dict, name:str):
    return True if name in config and config[name] is not None else False


def fatal(message:str):
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
    parser.add_argument('-config', '--config',
                        default='configs/train_base.yaml',
                        help='.yaml file path to configure all parameters; '
                        'default is configs/train.yaml')
    parser.add_argument('-infer_only', '--infer_only', action='store_true',
                        help='Flag to run only inference and no training on '
                        'input data')
    parser.add_argument('-output_dir', '--output_dir', default=None,
                        help='directory to store all model outputs')
    parser.add_argument('-print_time', '--print_time', action='store_true',
                        help='Flag to print date/time at start and end of '
                        'running (useful for slurm)')
    parser.add_argument('-resume', '--resume', action='store_true',
                        help='Flag to resume training from model checkpoint')
    parser.add_argument('-use_cuda', '--use_cuda', action='store_true',
                        help='Flag to use cuda for  gpu assistance (will '
                        'use only cpu if not specified')
    return parser.parse_args()


def arg_error(instr:str, cls=None):
    if cls is None:
        fatal(f'{instr}')
    else:
        fatal(f'Error in {cls.__class__.__name__}.__init__: {instr}')
    
    
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


def unsqueeze_repeat(x, dims:list[int], repeats=None):
    x = x if type(x) is torch.Tensor else torch.tensor(x)
    dims = [dims] if not isinstance(dims, list) else unsqueeze_dims
    if repeats is not None:
        if not len(repeats) == len(x.shape) + len(dims):
            fatal('In unsqueeze_repeat(), len(repeats) must equal '
                  'len(x.shape) + len(unsqueeze_dims)')
    for d in dims:
        x.unsqueeze(d)
    return x if repeats is None else x.repeat(repeats)
    

#-----------------------------------------------------------------------------#
#                               Image utilities                               #
#-----------------------------------------------------------------------------#

def largest_connected_component(x, vals=None, bgval=0):
    """
    Extracts the largest connected components for each foreground label in a
    multi-label image
    """
    x = x.cpu().numpy() if torch.is_tensor(x) else x
    vals = np.unique(x) if vals is None else vals
    vals = [i for i in vals if i != bgval]
    x_cc = np.tile(np.zeros(x.shape), (len(vals)+1,1,1,1))

    for j in range(len(vals)):
        x_j = np.squeeze(np.where(x==vals[j], 1, 0))
        x_j_cc, n_cc = ndimage.label(x_j, np.ones((3,3,3)))

        if n_cc > 1:
            cc_vals = np.unique(x_j_cc)[1:]
            cc_counts = np.array([(x_j_cc==i).sum() for i in cc_vals])
            try:
                largest_cc_val = cc_vals[cc_counts==cc_counts.max()].item()
            except:
                largest_cc_val = cc_vals[np.array(cc_counts==cc_counts.max(),
                                                  dtype=int)[0]].item()
        else:
            largest_cc_val = 1
            x_cc[j+1, ...] = np.where(x_j_cc==largest_cc_val, vals[j], 0)

    return np.sum(x_cc, axis=0, dtype=x.dtype)



def load_volume(path:str,                 # Path to load
                shape=(256,256,256),      # Output image dimensions
                voxsize=1.0,              # Output image resolution
                orientation='RAS',        # Output image orientation
                is_int:bool=False,        # Flag if image is int or float
                conform:bool=True,        # Flag to conform image
                to_tensor:bool=True,      # Flag to convert sf.Volume to tensor
                return_geoms:bool=False,  # Flag to return x geometries
):
    """
    Loads an input volume (using surfa) and conforms to a specific geometry (if
    conform=True). Returns the image as a tensor (if to_tensor=True) along with
    the original and conformed geometries.
    """
    # Load
    x = sf.load_volume(path)
    geom = x.geom
    
    # Conform
    x = x.conform(shape=shape if conform else geom.shape,
                      voxsize=voxsize if conform else geom.voxsize,
                      orientation=orientation if conform else geom.orientation,
                      dtype=np.int32 if is_int else np.float32,
                      method='nearest' if is_int else 'linear'
    )
    x = torch.Tensor(x.data).to(torch.int if is_int else torch.float) \
        if to_tensor else x
    return [x, geom, x.geom] if return_geoms else x


def min_max_norm(x, m:float=0, M:float=1, eps:float=1e-8):
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
    pad_width = [[cw[0], fs - cw[1]] \
                 for cw, fs in zip(crop_window, full_shape)]
    return np.pad(x, pad_width=pad_width)



def save_volume(x,                      # image data
                path,                     # path to save image
                input_geom=None,          # input image geometry
                conform_geom=None,        # conformed image geometry
                crop_bounds=None,         # bounds of data cropping
                label_lut=None,           # lut associated w/ image
                is_labels:bool=False,     # flag if output is label image
                is_onehot:bool=False,     # flag if output is onehot
                rescale:bool=False,
                return_output:bool=False  # flag to return conformed output
):
    """
    Saves an output image with the option to first conform the image to its
    original geometry. This requires both the conform_geom and the input_geom.
    Also has the option to return the conformed output image.
    """
    # Reform image to original size/geometry
    x = (x.cpu().numpy() if torch.is_tensor(x) else x).squeeze()
    x = x.astype(np.int32 if is_labels else np.float32)
    x = pad_volume(x, crop_bounds) if crop_bounds is not None else x
    x = min_max_norm(x, 0., 255.) if rescale else x
    x = sf.Volume(x, geometry=conform_geom)
    
    if input_geom is not None:
        x = x.conform(shape=input_geom.shape,
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
