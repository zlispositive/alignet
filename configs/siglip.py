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

r"""SigLIP triplet training."""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  import optax
  from kauldron_projects.alignet import losses
  from kauldron_projects.alignet import metrics
  from kauldron_projects.alignet import bv_wrapper
  from kauldron_projects.alignet import data
  from kauldron_projects.alignet import preprocessing
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = kd.train.Trainer()
  cfg.seed = 42
  cfg.aux = {
      "data_dir": "/path/to/alignet/data",
      "batch_size": 512,
      "transform": "untransformed",  # "uncertainty_distillation",
      "sampling": "cluster_border_500",
  }

  # Dataset
  cfg.train_ds = make_triplet_ds(
      training=True,
      sampling=cfg.ref.aux["sampling"],
      transform=cfg.ref.aux["transform"],
      root_dir=cfg.ref.aux["data_dir"],
      batch_size=cfg.ref.aux["batch_size"],
  )

  # Model
  cfg.model = bv_wrapper.TripletWrapper(
      image="batch.image_triplet",
      model=bv_wrapper.Model(variant="B/16", model_kwargs={"pool_type": "map"}),
      out_features=1024,
  )

  # load pretrained weights
  cfg.init_transform = bv_wrapper.Loader(
      init_file="SigLIP B/16 224", prefix="model/_Model_0"
  )

  # Training
  cfg.num_train_steps = 100_000

  # Losses
  cfg.train_losses = {
      "triplet": losses.JSDTripletLoss(
          logits="preds.triplet_logits",
          target_sims="batch.image_similarities",
          tau=100,
          normalize=False,
      ),
  }

  # Metrics
  cfg.train_metrics = {
      "accuracy": metrics.TripletAccuracy(logits="preds.triplet_logits"),
      "grad_norm": kd.metrics.SkipIfMissing(
          kd.metrics.TreeReduce(
              metric=kd.metrics.Norm(
                  tensor="grads", axis=None, aggregation_type="concat"
              )
          )
      ),
  }
  cfg.schedules = {
      "learning_rate": optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=3e-5,
          warmup_steps=5_000,
          decay_steps=cfg.ref.num_train_steps,
      )
  }

  cfg.optimizer = kd.optim.named_chain(**{
      "adam": optax.scale_by_adam(),
      "decay": kd.optim.decay_to_init(weight_decay=1.0),
      "lr": optax.scale_by_learning_rate(cfg.ref.schedules["learning_rate"]),
  })

  # Checkpointer
  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=10000,
      max_to_keep=3,
  )
  cfg.evals = get_triplet_evals(
      run_strategy=kd.evals.EveryNSteps(5000),
      batch_size=cfg.ref.aux["batch_size"],
      root_dir=cfg.ref.aux["data_dir"],
  )
  # TODO(klausg): Add things eval

  return cfg


def make_triplet_ds(
    training: bool,
    sampling: str,
    transform: str,
    root_dir: str,
    batch_size: int,
    resize: tuple[int, int] = (224, 224),
):
  return kd.data.py.DataSource(
      data_source=data.AligNetTriplets.from_args(
          root_dir=root_dir,
          split="train" if training else "val",
          transform=transform,
          sampling=sampling,
      ),
      shuffle=True if training else False,
      transforms=[
          preprocessing.ResizeImage(
              size=resize,
              key=["image_01", "image_02", "image_03"],
          ),
          preprocessing.Stack(
              out_key="image_triplet",
              in_keys=["image_01", "image_02", "image_03"],
          ),
          preprocessing.Stack(
              out_key="label_triplet",
              in_keys=["label_01", "label_02", "label_03"],
          ),
          kd.data.Rearrange(key="label_triplet", pattern="... -> ... 1"),
          kd.data.Elements(
              keep=["image_triplet", "image_similarities", "label_triplet"]
          ),
          kd.data.ValueRange(
              key="image_triplet", in_vrange=(0, 255), vrange=(0, 1)
          ),
      ],
      batch_size=batch_size,
  )


def get_triplet_evals(
    run_strategy: kd.evals.RunStrategy,
    batch_size: int,
    root_dir: str,
    transform: str = "uncertainty_distillation",
    resize: tuple[int, int] = (224, 224),
) -> dict[str, kd.evals.Evaluator]:
  """Create an evaluator for Alignnet triplets (with different samplings)."""

  return {
      f"eval_{sampling}": kd.evals.Evaluator(
          run=run_strategy,
          num_batches=2,
          ds=make_triplet_ds(
              training=False,
              sampling=sampling,
              transform=transform,
              root_dir=root_dir,
              batch_size=batch_size,
              resize=resize,
          ),
          cache=True,
          metrics={
              "accuracy": metrics.TripletAccuracy(
                  logits="preds.triplet_logits"
              ),
          },
          losses={},
      )
      for sampling in [
          "class_border",
          "within_class",
          "between_class",
      ]
  }
