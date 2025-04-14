import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import copy

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 1,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        self._base_raw_idx = copy.deepcopy(self._raw_idx)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def set_dyn_len(self, new_len):
        self._raw_idx = self._base_raw_idx[:new_len]

    def set_classes(self, cls_list):
        self._raw_labels = self._load_raw_labels()
        new_idcs = [self._raw_labels == cl for cl in cls_list]
        new_idcs = np.sum(np.vstack(new_idcs), 0)  # logical or
        new_idcs = np.where(new_idcs)  # find location
        self._raw_idx = self._base_raw_idx[new_idcs]
        assert all(sorted(cls_list) == np.unique(self._raw_labels[self._raw_idx]))
        print(f"Training on the following classes: {cls_list}")

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    # ---------------
    # NEW: YOLO code
    # ---------------
    def _load_raw_yolo_bboxes(self, raw_idx):
        """
        Return a list of bounding boxes in YOLO format:
          class, x_center, y_center, w, h  (all float, normalized to [0,1])
        or an empty list if none exist.
        """
        # By default, we map image_fnames[raw_idx] from e.g. "images/foo.jpg" => "annotations/foo.txt"
        # You can customize this path logic as needed.
        img_fname = self._image_fnames[raw_idx]
        base, _ext = os.path.splitext(img_fname)  # e.g. "foo"
        yolo_txt_path = os.path.join(self._annotations_dir, base + ".txt")  # "annotations/foo.txt"

        if not os.path.isfile(yolo_txt_path):
            return []

        bboxes = []
        with open(yolo_txt_path, "r") as f:
            lines = f.read().strip().splitlines()
            for line in lines:
                # e.g. "0 0.589844 0.634766 0.203125 0.496094"
                parts = line.split()
                cls_id = int(parts[0])
                xcen = float(parts[1])
                ycen = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                bboxes.append((cls_id, xcen, ycen, w, h))
        return bboxes

    def _create_bb_mask(self, bboxes, img_height, img_width, num_classes):
        """
        Create a multi‐channel bounding‐box mask of shape [num_classes, H, W].
        For each bounding box:
          - Convert YOLO coords -> pixel coords
          - Fill with 1’s in that class channel
        """
        mask = np.zeros((num_classes, img_height, img_width), dtype=np.float32)

        for (cls_id, xcen, ycen, w, h) in bboxes:
            # Convert center/width/height from normalized [0,1] to pixel coords
            box_x = xcen * img_width
            box_y = ycen * img_height
            box_w = w * img_width
            box_h = h * img_height

            x1 = int(round(box_x - box_w / 2))
            x2 = int(round(box_x + box_w / 2))
            y1 = int(round(box_y - box_h / 2))
            y2 = int(round(box_y + box_h / 2))

            # Clamp coords to valid range
            x1, x2 = np.clip([x1, x2], 0, img_width)
            y1, y2 = np.clip([y1, y2], 0, img_height)

            # Fill region for that class
            mask[cls_id, y1:y2, x1:x2] = 1.0

        return mask
    # ---------------

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        # Load the image
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8

        # Possibly horizontal flip
        do_flip = (self._xflip[idx] != 0)
        if do_flip:
            image = image[:, :, ::-1]  # flip in W dimension

        # Load old "label" if using
        label = self.get_label(idx)

        # Load YOLO bboxes
        bboxes = self._load_raw_yolo_bboxes(self._raw_idx[idx])

        # Create bounding‐box mask. Suppose you know you have e.g. num_classes=5
        # or you discover from your data. Hardcode or pass as config.
        num_classes = 10  # <-- REPLACE with correct # of classes
        bb_mask = self._create_bb_mask(
            bboxes=bboxes,
            img_height=self.image_shape[1],
            img_width=self.image_shape[2],
            num_classes=num_classes,
        )

        # Flip the mask if needed
        if do_flip:
            bb_mask = bb_mask[:, :, ::-1]

        # Return image, label, and bounding‐box mask
        return image.copy(), label, bb_mask  # <--- triple instead of just (image, label)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        # [N, C, H, W]
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        # e.g. 3
        assert len(self.image_shape) == 3
        return self.image_shape[0]

    @property
    def resolution(self):
        # e.g. 256
        assert len(self.image_shape) == 3
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        annotations_dir = None, # <--- Our new argument to specify YOLO annotation path
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self._annotations_dir = annotations_dir  # store it for our YOLO reading
        if annotations_dir is None:
            # If not provided, assume it is path + "/annotations"
            self._annotations_dir = os.path.join(self._path, "annotations")

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        # Gather all image files
        self._image_fnames = sorted(
            fname for fname in self._all_fnames
            if self._file_ext(fname) in PIL.Image.EXTENSION
        )
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        """
        If you still want to support dataset.json for class labels, do it here.
        Otherwise, you can just return None.
        """
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

