from collections import OrderedDict
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
import math
from enum import Enum, auto, unique

import torch
from torch.utils.data import Dataset
from utils import make_tuple

# --- Constants for data handling ---
# These prefixes are used to identify and organize image files.
REF_PREFIX_1 = '00'
PRE_PREFIX = '01'
REF_PREFIX_2 = '02'
COARSE_PREFIX = 'M'
FINE_PREFIX = 'L'
SCALE_FACTOR = 1
PIXEL_VALUE_SCALE = 1 # Deprecated, normalization is handled in PatchSet.transform

@unique
class Mode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    PREDICTION = auto()
    TESTING = auto()

def get_pair_path(im_dir, n_refs):
    """
    Organizes a set of dataset files in a predefined order.
    """
    paths = []
    order = OrderedDict()
    # Defines the expected file prefixes, e.g., 00_M for coarse reference, 00_L for fine reference.
    order[0] = REF_PREFIX_1 + '_' + COARSE_PREFIX
    order[1] = REF_PREFIX_1 + '_' + FINE_PREFIX
    order[2] = PRE_PREFIX + '_' + COARSE_PREFIX
    order[3] = PRE_PREFIX + '_' + FINE_PREFIX

    if n_refs == 2:
        order[2] = REF_PREFIX_2 + '_' + COARSE_PREFIX
        order[3] = REF_PREFIX_2 + '_' + FINE_PREFIX
        order[4] = PRE_PREFIX + '_' + COARSE_PREFIX
        order[5] = PRE_PREFIX + '_' + FINE_PREFIX

    for prefix in order.values():
        for path in Path(im_dir).glob('*.tif'):
            if path.name.startswith(prefix):
                paths.append(path.expanduser().resolve())
                # This break is crucial to prevent including a second image
                # representing the same time and satellite in a folder.
                break
    if n_refs == 2:
        assert len(paths) == 6 or len(paths) == 5
    else:
        assert len(paths) == 3 or len(paths) == 4
    return paths

def load_image_pair(directory: Path,patch_padding, mode: Mode):
    paths = get_pair_path(directory,n_refs=1)
    images = []
    for p in paths:
        with rasterio.open(str(p)) as ds:
            im = ds.read()
            image = np.zeros(
                (4,im.shape[1] + patch_padding[0] * 2, im.shape[2] + patch_padding[1] * 2)).astype(
                np.float32)
            image[:,patch_padding[0]:im.shape[1] + patch_padding[0], patch_padding[1]:im.shape[2] + patch_padding[1]] = im[0:4,:,:]
            images.append(image)
    return images


class PatchSet(Dataset):
    """
    Loads and splits each image into smaller patches.
    Note: Pillow's Image is column-major, while Numpy's ndarray is row-major.
    """

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None,patch_padding=None, mode=Mode.TRAINING):
        super(PatchSet, self).__init__()
        patch_size = make_tuple(patch_size)
        patch_stride = make_tuple(patch_stride) if patch_stride else patch_size
        patch_padding = make_tuple(patch_padding)

        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.mode = mode

        self.image_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)
        # Calculate the number of patches after splitting the image.
        self.n_patch_x = math.ceil((image_size[0] + patch_padding[0] - patch_size[0] + 1) / patch_stride[0])
        self.n_patch_y = math.ceil((image_size[1] + patch_padding[0] - patch_size[1] + 1) / patch_stride[1])
        self.num_patch = self.num_im_pairs * self.n_patch_x * self.n_patch_y

    @staticmethod
    def transform(data):
        """
        Normalizes image data to the [0, 1] range.
        The original pixel values are assumed to be scaled up to 10000.
        """
        data[data < 0] = 0
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        out = data.div_(10000)
        return out

    def map_index(self, index):
        id_n = index // (self.n_patch_x * self.n_patch_y)
        residual = index % (self.n_patch_x * self.n_patch_y)
        id_x = self.patch_stride[0] * (residual % self.n_patch_x)
        id_y = self.patch_stride[1] * (residual // self.n_patch_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        images = load_image_pair(self.image_dirs[id_n],self.patch_padding, mode=self.mode)
        patches = [None] * len(images)

        for i in range(len(patches)):
            im = images[i][:,
                 id_x: (id_x + self.patch_size[0] + self.patch_padding[0]*2),
                 id_y: (id_y + self.patch_size[1] + self.patch_padding[0]*2)]
            patches[i] = self.transform(im)

        del images[:]
        del images

        # Use only the first 4 channels
        patches = [(patch[0:4, :, :]) for patch in patches]
        return patches

    def __len__(self):
        return self.num_patch

if __name__ == "__main__":
    """Test script for the `load_image_pair` function."""

    # 1. Specify the directory containing the test image pair.
    #    IMPORTANT: Users should change this to their local test data directory.
    test_image_directory = Path('./data/cia/test/your_test_sample_folder')

    # 2. Define necessary parameters.
    test_patch_padding = (0, 0)
    test_mode = Mode.TESTING

    print(f"--- Testing `load_image_pair` function ---")
    print(f"Using directory: {test_image_directory}")
    print(f"Parameters: padding={test_patch_padding}, mode={test_mode}")

    # 3. Call the function and print basic information.
    try:
        # Check if the directory exists.
        if not test_image_directory.is_dir():
            raise FileNotFoundError(f"Test directory not found: {test_image_directory}")

        # Load images.
        loaded_images = load_image_pair(test_image_directory, test_patch_padding, test_mode)

        # Print success message and basic info.
        print(f"\n[SUCCESS] `load_image_pair` returned {len(loaded_images)} images.")
        if loaded_images:
            print("Loaded image shapes:")
            for i, img in enumerate(loaded_images):
                print(f"  Image {i}: {img.shape} (type: {img.dtype})")
        else:
            print("Warning: The returned image list is empty. Please check the directory content and `get_pair_path` logic.")

    except Exception as e:
        # Print error message if any exception occurs.
        print(f"\n[FAILURE] An error occurred while testing `load_image_pair`: {e}")

    print("\n--- Test finished ---")
  