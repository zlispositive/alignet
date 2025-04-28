# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Different ImageNet and THINGS data sources for AligNet."""

from __future__ import annotations

import dataclasses
import functools
from typing import Dict, Union

from etils import epath
import grain.python as pygrain
import numpy as np
import tensorflow_datasets as tfds


@dataclasses.dataclass(frozen=True)
class AligNetTriplets(pygrain.RandomAccessDataSource):
  """Dataset class for loading triplets of input data."""

  dataset_path: str
  split: str

  @classmethod
  def from_args(
      cls,
      root_dir: str,
      split: str = "train",  # or "val"
      transform: str = "uncertainty_distillation",
      sampling: str = "cluster_border_500",
  ) -> AligNetTriplets:
    """Create Source by specifying split,transform, etc. instead of paths."""
    dataset_path = epath.Path(root_dir) / transform / f"{split}_{sampling}.npz"
    return cls(
        dataset_path=dataset_path,
        split=split,
    )

  @functools.cached_property
  def data(self) -> np.ndarray:
    path = epath.Path(self.dataset_path)
    with path.open("rb") as f:
      data = dict(np.load(f))
    return data

  @functools.cached_property
  def imagenet_ds(self):
    split_i1k = "train" if self.split in ["train", "val"] else "validation"
    return tfds.data_source("imagenet2012", split=split_i1k)

  def __len__(self) -> int:
    return self.data["indices"].shape[0]

  def __getstate__(self):
    # ignore data, and imagenet_ds for pickling
    return {"dataset_path": self.dataset_path, "split": self.split}

  def __getitem__(
      self, record_key: int
  ) -> Dict[str, Union[np.ndarray, int, str, None]]:
    indices = self.data["indices"][record_key]
    similarities = self.data["similarities"][record_key]
    return self._get_record(indices=indices, similarities=similarities)

  def _get_record(
      self,
      indices: np.ndarray,
      similarities: np.ndarray,
  ) -> Dict[str, Union[np.ndarray, int, str, None]]:
    """Returns triplets of images along with labels and file names; similarities are optional."""
    record = {}
    for i, idx in enumerate(indices, start=1):
      i1k_record = self.imagenet_ds[int(idx)]
      record[f"image_{i:02d}"] = i1k_record["image"]
      record[f"label_{i:02d}"] = np.array(i1k_record["label"], dtype=np.int32)
      record[f"file_name_{i:02d}"] = i1k_record["file_name"]
    record["image_indices"] = indices
    record["image_similarities"] = similarities.astype(np.float32)
    return record

  def __repr__(self):
    clsname = type(self).__name__
    return f"{clsname}({self.dataset_path})"
