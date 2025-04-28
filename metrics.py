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

"""Collection of metrics for alignet."""

from __future__ import annotations

import dataclasses
from typing import Optional

import flax.struct
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Float, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class TripletAccuracy(kd.metrics.Metric):
  """Fraction of triplets where (i,j) is the most similar pair."""

  logits: kd.kontext.Key = (
      kd.kontext.REQUIRED
  )  # e.g. "preds.logits" of CLS token
  mask: Optional[kd.kontext.Key] = None

  @flax.struct.dataclass
  class State(kd.metrics.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      logits: Float["*b 3 d"],
      mask: Optional[Float["*b"]] = None,
  ) -> TripletAccuracy.State:
    dot_ij = jnp.sum(logits[:, 0, :] * logits[:, 1, :], axis=-1)
    dot_ik = jnp.sum(logits[:, 0, :] * logits[:, 2, :], axis=-1)
    dot_jk = jnp.sum(logits[:, 1, :] * logits[:, 2, :], axis=-1)
    triplet_sims = jnp.stack([dot_jk, dot_ik, dot_ij], axis=-1)
    check_type(triplet_sims, Float["*b 3"])
    # assumes that the last object in the triplet is the ooo
    # => the dot-product between i and j should be maximal (most similar)
    # NOTE: Putting the dot_ij last rather than first prevents an issue with
    #       high accuracy if all values are the same.
    correct = triplet_sims.argmax(axis=-1) == 2
    return self.State.from_values(values=correct, mask=mask)
