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

"""Wraps the Vision Transformer model from Big Vision.

See big_vision.models.vit.py for details.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence, TypeVar, TypedDict

import einops
from flax import linen as nn
import jax
from jax import lax
from kauldron import kd
from kauldron.typing import Float, Shape, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
from kauldron_projects.alignet import bv_utils
from kauldron_projects.alignet import bv_vit

_StateT = TypeVar("_StateT", bound=kd.train.TrainState)


class Model(nn.Module):
  """ViT model."""

  variant: str = "B/16"
  model_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  image: Optional[kd.kontext.Key] = kd.kontext.REQUIRED
  is_training = kd.nn.train_property()

  @typechecked
  @nn.compact
  def __call__(self, image: Float["*b h w c"]) -> dict[str, Any]:
    model = bv_vit.Model(variant=self.variant, **self.model_kwargs)

    out = dict()
    out["logits"], tmp = model(image)
    out.update(tmp)
    return out


@dataclasses.dataclass(frozen=True, kw_only=True)
class Loader(kd.ckpts.AbstractPartialLoader):
  """Loader for pretrained weights compatible with kd.ckpt.Checkpointer."""

  init_file: str
  dont_load: Optional[Sequence[str]] = ()
  prefix: Optional[str] = None  # Prefix under which the params are stored.

  def transform(self, state: _StateT) -> _StateT:
    """Transform the state by updating it with pre-trained bv weights."""

    # Obtain (randomly) initialized params from state.
    init_params = state.params
    if self.prefix is not None:
      assert "." not in self.prefix, f"Unsupported prefix: {self.prefix=}"
      for k in self.prefix.split("/"):
        init_params = init_params[k]

    # The next part of the function is a copy of bv_vit.load()
    init_file = bv_vit.VANITY_NAMES.get(self.init_file, self.init_file)
    restored_params = bv_utils.load_params(init_file)
    restored_params = bv_vit.fix_old_checkpoints(restored_params)

    # Kauldron does not support Scan-style checkpoints.
    if "encoderblock" in restored_params["Transformer"]:
      restored_params = bv_vit.scan_to_pyloop(restored_params)

    # possibly use the random init for some of the params (such as, the head).
    restored_params = bv_utils.merge_params(
        restored_params, init_params, dont_load=self.dont_load
    )

    # resample posemb if needed.
    if init_params and "pos_embedding" in init_params:
      restored_params["pos_embedding"] = bv_vit.resample_posemb(
          old=restored_params["pos_embedding"], new=init_params["pos_embedding"]
      )

    # BV shards tensorstore-based ceckpoints to cpu0, which messes with
    # kauldron's sharding/replication, so we have to remove the sharding again:
    restored_params = jax.tree.map(jax.device_get, restored_params)

    if self.prefix is not None:
      sp = state.params
      *path, last_key = self.prefix.split("/")
      for k in path:
        sp = sp[k]
      sp[last_key] = restored_params
    else:
      state.params = restored_params
    return state


class TripletOutput(TypedDict):
  """Output shapes of TripletWrapper model."""

  logits: Float["*b num_classes"] | None
  pre_logits: Float["*b hidden"]
  triplet_logits: Float["*b out_features"]


class TripletWrapper(nn.Module):
  """Wrapper for models not designed with triplets in mind."""

  model: nn.Module
  image: kd.kontext.Key = "batch.image"
  out_features: int = 1024
  stop_grad: bool = False
  gap: bool = False  # add a global average pooling

  @typechecked
  @nn.compact
  def __call__(self, image: Float["*b h w c"]) -> TripletOutput:
    # collapse batch axis
    flat_images = einops.rearrange(image, "... h w c -> (...) h w c")
    out = self.model(flat_images)
    batch_dims = Shape("*b")
    if isinstance(out, dict):
      logits = out["logits"]
      pre_logits = out["pre_logits"]
    else:
      assert isinstance(out, Float["..."])
      pre_logits = out
      logits = None

    if self.gap:
      check_type(pre_logits, Float["B H W hidden"])
      pre_logits = einops.reduce(
          pre_logits, "B H W hidden -> B hidden", reduction="mean"
      )

    # re-expand batch axis
    if logits is not None:
      logits = logits.reshape(batch_dims + logits.shape[1:])
    pre_logits = pre_logits.reshape(batch_dims + pre_logits.shape[1:])

    if self.stop_grad:
      pre_logits = lax.stop_gradient(pre_logits)
    triplet_logits = nn.Dense(self.out_features, name="triplet_head")(
        pre_logits
    )
    return {
        "logits": logits,
        "pre_logits": pre_logits,
        "triplet_logits": triplet_logits,
    }
