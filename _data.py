import numpy as np

from tqdm import tqdm
from typing import Literal
from pathlib import Path
from skimage import io, color, transform, feature
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tempfile import tempdir

RESIZE_SHAPE = (224, 224)
PATH_RAWFILE = "./data/pneumonia.npy"
TRAIN_TEST_SPLIT = (5232, 624)

# To speed up calculation, we save previously generated dataset into your temporary
# systems folder. If you are running this on a mac or a linux machine, they will be
# automatically garbage-collected by the system when certain conditions are met (mac:
# system reboot or file not accessed in 3 days; linux: system reboot). If you are
# running this on a windows machine, you might have to delete them by hand.
TEMP_FOLDER = f"{tempdir}/pneumonia-detection/"


def pre_import_hook():
  Path(TEMP_FOLDER).mkdir(exist_ok=True)

  if Path(PATH_RAWFILE).is_file():
    return

  image_paths = [str(posix) for posix in list(Path("./data/").rglob("*.jpeg"))]
  image_pixels = ([], [])

  for image_path in tqdm(image_paths, desc="Pre-Import Hook"):
    image_file = io.imread(image_path)
    if len(image_file.shape) == 3:
      image_file = color.rgb2gray(image_file)
    image_pixel = transform.resize(
      image_file, RESIZE_SHAPE, preserve_range=True
    ).flatten()
    is_test = int("test" in image_path)
    is_pneumonia = "PNEUMONIA" in image_path
    image_pixels[is_test].append(np.concatenate([image_pixel, [int(is_pneumonia)]]))
  np.save(PATH_RAWFILE, image_pixels[0] + image_pixels[1])


def resolve_filename(
  include_raw: bool = True,
  include_hog: bool = False,
  include_lbp: bool = False,
  pca_mode: Literal["global", "local", "none"] = "none",
) -> str:
  tags = ["pneumonia"]
  if include_raw:
    tags.append("raw")
  if include_hog:
    tags.append("hog")
  if include_lbp:
    tags.append("lbp")
  if pca_mode == "global":
    tags.append("pcag")
  elif pca_mode == "local":
    tags.append("pcal")
  filename = "_".join(tags) + ".npy"

  return filename


def load_pneumonia(
  include_raw: bool = True,
  include_hog: bool = False,
  include_lbp: bool = False,
  pca_mode: Literal["global", "local", "none"] = "none",
) -> tuple[np.ndarray, np.ndarray]:
  # Apply Argument Value Check
  valid_pca_modes = {"global", "local", "none"}
  if pca_mode not in valid_pca_modes:
    raise ValueError(
      f"{repr(pca_mode)} is not a valid PCA mode. Pick between {valid_pca_modes}."
    )
  if pca_mode == "global" and include_raw:
    raise ValueError("Cannot include_raw when pca_mode is set to `global`.")
  if pca_mode == "local" and not (include_raw + include_hog + include_lbp):
    raise ValueError("Include at least one of raw, hog, or lbp to apply local PCA.")
  if pca_mode == "none" and not (include_raw + include_hog + include_lbp):
    raise ValueError("No data is loaded.")

  temp_filename = resolve_filename(include_raw, include_hog, include_lbp, pca_mode)
  temp_filepath = f"{TEMP_FOLDER}{temp_filename}"

  TRAIN_SIZE, _ = TRAIN_TEST_SPLIT
  IMAGE_PIXELS = RESIZE_SHAPE[0] * RESIZE_SHAPE[1]

  # Reuse Previously Computed Dataset
  if Path(temp_filepath).is_file():
    return np.split(np.load(temp_filepath), [TRAIN_SIZE])

  src_file = np.load(PATH_RAWFILE)

  src_features, src_labels = np.split(src_file, [IMAGE_PIXELS], axis=1)
  target_features = []

  for src_feature in tqdm(src_features):
    target_feature = []

    if include_raw:
      target_feature.append(src_feature)

    if include_hog or include_lbp:
      image_2d = src_feature.reshape(RESIZE_SHAPE)
    if include_hog:
      target_feature.append(
        feature.hog(
          image_2d,
          orientations=9,
          pixels_per_cell=(8, 8),
          cells_per_block=(2, 2),
          block_norm="L2-Hys",
          transform_sqrt=True,
          feature_vector=True,
          visualize=False,
        )
      )
    if include_lbp:
      image_lbp = feature.local_binary_pattern(
        image_2d.astype(np.uint8), 24, 3, "uniform"
      )
      image_hist = np.histogram(
        image_lbp.ravel(), bins=26, range=(0, 26), density=True
      )[0].astype("float")
      image_hist /= image_hist.sum() + 1e-6
      target_feature.append(image_hist)
    if len(target_feature) > 0:
      target_features.append(np.concatenate(target_feature))
    else:
      target_features.append([])

  sc = StandardScaler()
  if pca_mode == "global":
    pca = PCA(n_components=2)
    print("Calculating PCA...")
    target_dataset = np.concatenate(
      [
        sc.fit_transform(
          np.concatenate([target_features, pca.fit_transform(src_features)], axis=1)
        ),
        src_labels,
      ],
      axis=1,
    )
  elif pca_mode == "local":
    pca = PCA(n_components=2)
    print("Calculating PCA...")
    target_dataset = np.concatenate(
      [sc.fit_transform(pca.fit_transform(target_features)), src_labels], axis=1
    )
  else:
    target_dataset = np.concatenate(
      [sc.fit_transform(target_features), src_labels], axis=1
    )
  np.save(temp_filepath, target_dataset)
  return np.split(target_dataset, [TRAIN_SIZE])


PARAMETER_PERMUTATION = [
  [True, True, True, "none"],
  [False, True, True, "none"],
  [True, False, True, "none"],
  [True, True, False, "none"],
  [False, False, True, "none"],
  [True, False, False, "none"],
  [False, True, False, "none"],
  [True, True, True, "local"],
  [False, True, True, "local"],
  [True, False, True, "local"],
  [True, True, False, "local"],
  [False, False, True, "local"],
  [True, False, False, "local"],
  [False, True, False, "local"],
  [False, True, True, "global"],
  [False, False, True, "global"],
  [False, True, False, "global"],
  [False, False, False, "global"],
]

pre_import_hook()
