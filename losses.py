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

"""Collection of triplet odd-one-out losses for training on AligNet."""

from __future__ import annotations

import dataclasses
from typing import Sequence

import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class ContrastiveTripletLoss(kd.losses.Loss):
  """A contrastive triplet loss (https://arxiv.org/pdf/2306.04507.pdf)."""

  logits: kd.kontext.Key = kd.kontext.REQUIRED
  tau: float = 1.0
  normalize: bool = False

  @typechecked
  def get_values(self, logits: Float["*b 3 d"]) -> Float["*b"]:
    dot_ij, dot_ik, dot_jk = _dot_products(logits, self.normalize)
    triplet_sims = jnp.stack([dot_ij, dot_ik, dot_jk], axis=-1)
    return -dot_ij / self.tau + jax.nn.logsumexp(
        triplet_sims / self.tau, axis=-1
    )


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class KLDTripletLoss(kd.losses.Loss):
  """A KL divergence loss for using soft targets."""

  logits: kd.kontext.Key = kd.kontext.REQUIRED
  target_sims: kd.kontext.Key = kd.kontext.REQUIRED
  tau: float = 1.0
  normalize: bool = False

  @typechecked
  def get_values(
      self, logits: Float["*b 3 d"], target_sims: Float["*b 3"]
  ) -> Float["*b"]:
    dot_ij, dot_ik, dot_jk = _dot_products(logits, self.normalize)
    predicted_sims = jnp.stack([dot_ij, dot_ik, dot_jk], axis=-1)
    target = jax.nn.softmax(target_sims, axis=-1)
    predicted = jax.nn.softmax(predicted_sims / self.tau, axis=-1)
    kld = _neg_entropy(target) + _cross_entropy(target, predicted)
    return kld


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class JSDTripletLoss(kd.losses.Loss):
  """A Jensen-Shannon divergence loss for using soft targets."""
  logits: kd.kontext.Key = kd.kontext.REQUIRED
  target_sims: kd.kontext.Key = kd.kontext.REQUIRED
  tau: float = 1.0
  normalize: bool = False

  @typechecked
  def get_values(
      self, logits: Float["*b 3 d"], target_sims: Float["*b 3"]
  ) -> Float["*b"]:
    dot_ij, dot_ik, dot_jk = _dot_products(logits, self.normalize)
    predicted_sims = jnp.stack([dot_ij, dot_ik, dot_jk], axis=-1)
    p = jax.nn.softmax(target_sims, axis=-1)
    q = jax.nn.softmax(predicted_sims / self.tau, axis=-1)
    m = (p + q) / 2
    kld_pm = _neg_entropy(p) + _cross_entropy(p, m)
    kld_qm = _neg_entropy(q) + _cross_entropy(q, m)
    jsd = (kld_pm + kld_qm) / 2
    return jsd


@typechecked
def _neg_entropy(p: Float["*b 3"]) -> Float["*b"]:
  return jnp.sum(jnp.where(p == 0.0, 0.0, p * jnp.log(p)), axis=-1)


@typechecked
def _cross_entropy(p: Float["*b 3"], q: Float["*b 3"]) -> Float["*b"]:
  return -jnp.sum(p * jnp.log(q), axis=-1)


@typechecked
def _dot_products(
    logits: Float["*b 3 d"], normalize: bool = False
) -> Sequence[Float["*b"]]:
  """Calculates the inner products between all pairs of inputs."""
  if normalize:
    logits /= jnp.linalg.norm(logits, ord=None, axis=-1, keepdims=True)
  dot_ij = jnp.sum(logits[:, 0, :] * logits[:, 1, :], axis=-1)
  dot_ik = jnp.sum(logits[:, 0, :] * logits[:, 2, :], axis=-1)
  dot_jk = jnp.sum(logits[:, 1, :] * logits[:, 2, :], axis=-1)
  return dot_ij, dot_ik, dot_jk
