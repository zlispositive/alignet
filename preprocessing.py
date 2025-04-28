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

"""Preprocessing Ops for AligNet."""
from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import cv2
from etils import enp
from kauldron import kd


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Stack(kd.data.MapTransform):
  """Vstack a set of images."""

  in_keys: Sequence[str]
  out_key: str

  def map(self, elements: Any) -> Any:
    if not self.in_keys:
      raise ValueError("in_keys must not be empty.")
    xnp = enp.lazy.get_xnp(elements[self.in_keys[0]])
    stacked = xnp.stack([elements[k] for k in self.in_keys], axis=0)
    elements[self.out_key] = stacked
    return elements


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ResizeImage(kd.data.ElementWiseTransform):
  """Resizes an image using OpenCV."""

  size: tuple[int, int]
  interpolation: str = "cubic"

  def map_element(self, element: Any) -> Any:
    interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }[self.interpolation]

    return cv2.resize(element, self.size, interpolation=interpolation)

