import os
import sys
import random
import numpy as np
import torch
import argparse
import surfa as sf


def fatal(message:str):
    print(message)
    sys.exit(1)


def init_text_file(fname, string, check_if_exists=True):
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
    parser.add_argument('-use_cuda', '--use_cuda', action='store_true',
                        help='Flag to use cuda for  gpu assistance (will '
                        'use only cpu if not specified')
    parser.add_argument('-resume', '--resume', action='store_true',
                        help='Flag to resume training from model checkpoint')
    return parser.parse_args()
    
    
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
    dims = [dims] if not isinstance(dims, list) else unsqueeze_dims
    if repeats is not None:
        if not len(repeats) == len(x.shape) + len(dims):
            fatal('In unsqueeze_repeat(), len(repeats) must equal '
                  'len(x.shape) + len(unsqueeze_dims)')
    for d in unsqueeze_dims:
        x.unsqueeze(d)
    return x if repeats is None else x.repeat(repeats)
    

#-----------------------------------------------------------------------------#
#                               Image utilities                               #
#-----------------------------------------------------------------------------#

###
def get_crop_window(vol, patch_sz=[96, 96, 96], bffr:int=2):
    vol_sz = vol.shape
    bbox = [[0, vol_sz[i]-1] for i in range(len(vol_sz))]

    while np.sum(vol[bbox[0][0],:,:]) == 0: bbox[0][0] += 1
    while np.sum(vol[bbox[0][1],:,:]) == 0: bbox[0][1] -= 1
    while np.sum(vol[:,bbox[1][0],:]) == 0: bbox[1][0] += 1
    while np.sum(vol[:,bbox[1][1],:]) == 0: bbox[1][1] -= 1
    while np.sum(vol[:,:,bbox[2][0]]) == 0: bbox[2][0] += 1
    while np.sum(vol[:,:,bbox[2][1]]) == 0: bbox[2][1] -= 1

    bbox = [[bb[0] - (bffr), bb[1] + (bffr)] for bb in bbox]
    center = [(bb[0] + bb[1])//2 for bb in bbox]
    bounds = [[c - ps//2, c + ps//2] for c, ps in zip(center, patch_sz)]

    for i in range(len(bbox)):
        if bounds[i][0] < 0:  shift = bounds[i][0]
        elif bounds[i][1] > vol_sz[i]:  shift = bounds[i][1] - vol_sz[i]
        else:  shift = 0
        bounds[i] = [bounds[i][0] - shift, bounds[i][1] - shift]

    return bounds


###
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


###
def load_volume(path:str,                 # Path to load
                shape=(256,256,256),      # Output image dimensions
                voxsize=1.0,              # Output image resolution
                orientation='RAS',        # Output image orientation
                is_int:bool=False,        # Flag if image is int or float
                conform:bool=True,        # Flag to conform image
                to_tensor:bool=True,      # Flag to convert sf.Volume to tensor
                return_geoms:bool=False,  # Flag to return img geometries
):
    """
    Loads an input volume (using surfa) and conforms to a specific geometry (if
    conform=True). Returns the image as a tensor (if to_tensor=True) along with
    the original and conformed geometries.
    """
    # Load
    img = sf.load_volume(path)
    geom = img.geom
    
    # Conform
    img = img.conform(shape=shape if conform else geom.shape,
                      voxsize=voxsize if conform else geom.voxsize,
                      orientation=orientation if conform else geom.orientation,
                      dtype=np.int32 if is_int else np.float32,
                      method='nearest' if is_int else 'linear'
    )
    x = torch.Tensor(img.data).to(torch.int if is_int else torch.float) \
        if to_tensor else img
    return [x, geom, img.geom] if return_geoms else x


###
def pad_volume(img, crop_window, full_shape=(256, 256, 256)):
    pad_width = [[cw[0], fs - cw[1]] \
                 for cw, fs in zip(crop_window, full_shape)]
    return np.pad(img, pad_width=pad_width)


###
def save_volume(img,                      # image data
                path,                     # path to save image
                input_geom=None,          # input image geometry
                conform_geom=None,        # conformed image geometry
                crop_bounds=None,         # bounds of data cropping
                label_lut=None,           # lut associated w/ image
                is_labels:bool=False,     # flag if output is label image
                return_output:bool=False  # flag to return conformed output
):
    """
    Saves an output image with the option to first conform the image to its
    original geometry. This requires both the conform_geom and the input_geom.
    Also has the option to return the conformed output image.
    """
    # Reform image to original size/geometry
    img = (img.cpu().numpy() if torch.is_tensor(img) else img).squeeze()
    img = img.astype(np.int32 if is_labels else np.float32)
    img = pad_volume(img, crop_bounds) if crop_bounds is not None else img
    img = sf.Volume(img, geometry=conform_geom)

    if input_geom is not None:
        img = img.conform(shape=input_geom.shape,
                          voxsize=input_geom.voxsize,
                          orientation=input_geom.orientation,
                          method='nearest' if is_labels else 'linear'
        )
    if label_lut is not None:
        img.labels = label_lut
        
    # Write image
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    img.save(path)

    return img if return_output else None
