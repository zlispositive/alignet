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

"""Utils very specific to this project, not generic.

forked from big_vision/utils.py

https://github.com/google-research/big_vision/blob/6d6c28a9634fd2f48f0f505f112d063dfc9bdf96/big_vision/utils.py
Credit to the Big Vision Authors.
"""

import collections
import dataclasses
import io
import os
import re
from typing import Mapping

from absl import logging
import flax
import jax
from jax.experimental.array_serialization import serialization as array_serial
import ml_collections as mlc
import numpy as np
import tensorflow.io.gfile as gfile  # pylint: disable=consider-using-from-import


# pylint: disable=logging-fstring-interpolation


def merge_params(loaded, inited, dont_load=(), match_dtype=False):
  """Makes `loaded` pytree match `init`, warning or failing on mismatch.

  Args:
    loaded: pytree of parameters, typically loaded from a checkpoint.
    inited: pytree of parameter, typically coming from model init.
    dont_load: List of regexes for parameters which shall not be taken from
      `loaded`, either because they should remain at their init value, or
      because they are missing on either side.
    match_dtype: returned pytree as leaves converted to dtype from `inited`.

  Returns:
    If successful, a new pytree which matches the structure of `init`
    but contains values from `loaded`, except for `dont_load`.

    If structures don't match and mismatches are not covered by regexes in
    `dont_load` argument, then raises an exception with more information.
  """
  if inited is None:  # A useful shortcut for example for colabs.
    return loaded

  dont_load = check_and_compile_patterns(dont_load)

  def should_merge(name):
    return not any(pattern.fullmatch(name) for pattern in dont_load)

  loaded_flat, _ = tree_flatten_with_names(loaded)
  inited_flat, _ = tree_flatten_with_names(inited)
  loaded_flat = {k: v for k, v in loaded_flat}
  inited_flat = {k: v for k, v in inited_flat}

  # Let's first build the pytree from all common keys.
  merged = {}
  for name, init_val in inited_flat.items():
    # param is present in both. Load or ignore it!
    if name in loaded_flat and should_merge(name):
      merged[name] = loaded_flat[name]
      if match_dtype:
        merged[name] = loaded_flat[name].astype(init_val.dtype)
    else:
      logging.info("Ignoring checkpoint and using init value for %s", name)
      merged[name] = init_val

  def pp(title, names, indent="  "):  # Just pretty-printing
    if names:
      return f"{title}:\n" + "\n".join(f"{indent}{k}" for k in sorted(names))
    else:
      return ""

  # Now, if there are keys that only exist in inited or loaded, be helpful:
  not_in_loaded = inited_flat.keys() - loaded_flat.keys()
  not_in_inited = loaded_flat.keys() - inited_flat.keys()
  logging.info(pp("Parameters in model but not in checkpoint", not_in_loaded))
  logging.info(pp("Parameters in checkpoint but not in model", not_in_inited))

  # And now see if any of them are not explicitly ignored => an error
  not_in_loaded = {k for k in not_in_loaded if should_merge(k)}
  not_in_inited = {k for k in not_in_inited if should_merge(k)}

  if not_in_loaded or not_in_inited:
    raise ValueError(
        pp("Params in checkpoint", loaded_flat.keys())
        + "\n"
        + pp("Params in model (code)", inited_flat.keys())
        + "\n"
        + pp(
            "Params in model (code) but not in checkpoint and not"
            " `dont_load`ed",
            not_in_loaded,
            indent=" - ",
        )
        + "\n"  # Special indent for tests.
        + pp(
            "Params in checkpoint but not in model (code) and not"
            " `dont_load`ed",
            not_in_inited,
            indent=" + ",
        )
    )  # Special indent for tests.

  return recover_tree(merged.keys(), merged.values())


def check_and_compile_patterns(patterns):
  """Validates and compiles a list of param-patterns.

  The validation consists of checking for common mistakes, currently only that
  the pattern does not start with a slash, because unlike FLAX, our parameter
  names don't start with a slash.

  Args:
    patterns: a single (string) pattern (regex), or a list of patterns.

  Returns:
    A list of compiled and verified regexes.
  """
  if isinstance(patterns, str):
    patterns = [patterns]

  assert isinstance(patterns, (list, tuple)), patterns

  def check_and_compile(pattern):
    assert not pattern.startswith(
        "/"
    ), f"Big vision parameter names never start with '/': '{pattern}"
    return re.compile(pattern)

  return list(map(check_and_compile, patterns))


def tree_flatten_with_names(tree):
  """Populates tree_flatten with leaf names.

  This function populates output of tree_flatten with leaf names, using a
  custom traversal that produces names is provided. The custom traversal does
  NOT have to traverse tree in the same order as jax, as we take care of
  automatically aligning jax' and custom traversals.

  Args:
    tree: python tree.

  Returns:
    A list of values with names: [(name, value), ...]
  """
  vals, tree_def = jax.tree.flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traverasal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are saved by our programs
  without need to access the exact source code of the experiment. In particular,
  it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
  subtree of parameters.

  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.

  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if "/" not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split("/", 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def load_params(ckpt, **kw):
  """Loads the parameters of a big_vision checkpoint, both old or new format.

  Args:
    ckpt: Path to the checkpoint (.npz, .ts) or dict-like.
    **kw: forwarded to the underlying load function (_np or _ts).

  Returns:
    A pytree that is the checkpoint, potentially sharded.

  Notes:
    The `ckpt` string can contain an colon-separated "submodel" indicator, like
    `img` in the example `/path/to/file.npz:img`.
    This is used to load sub-parts of a model, for example the image load the
    image encoder out of a two_tower (SigLIP) checkpoint, or distillation.
    This way, ANY model that uses this function can load itself from a
    checkpoint that contains multiple sub-models.
  """
  key = None  # Whether we want to extract only a sub-key of the model.
  params = None
  if isinstance(ckpt, str):  # Most common case of passing a checkpoint path.
    # Potentially read out the sub-part to load from after the colon
    # '/path/to/file:img/head' => '/path/to/file', 'img/head'
    # 'gs://path/to/file' => 'gs://path/to/file', None
    if match := re.match(r"^(.*?/.*?)(?::([\w/]+))?$", ckpt):
      ckpt, key = match.groups()
    else:
      raise ValueError(f"Weird ckpt path: {ckpt} ; Maybe prepend ./ ?")

    # Use the checkpoint filename to detect when we're loading old-style .npz
    # checkpoints, as opposed to new-style tensorstore checkpoint folders.
    if ".npz" in ckpt:  # Not a perfect heuristic, but good enough.
      checkpoint = load_checkpoint_np(ckpt, **kw)
      checkpoint = jax.tree.map(recover_dtype, checkpoint)
      if "params" in checkpoint:
        # Checkpoint with optax state (after (internal link)).
        params = checkpoint["params"]
      elif "opt" in checkpoint:
        # Checkpoint with Flax optimizer.
        params = checkpoint["opt"]["target"]
      else:
        # When open-sourcing, we often shared only the params directly.
        params = checkpoint
    else:
      # Here we're now loading new-style tensorstore checkpoints.
      # We can be a more efficient and load params and `key` only right away.
      regex = f"params/{key}($|/.*)" if key else "params/.*"
      assert "regex" not in kw, "For a custom regex, use tsload directly."
      kw["regex"] = regex
      checkpoint = load_checkpoint_ts(ckpt, **kw)
      params = checkpoint["params"]

  if key is not None:
    params = tree_get(params, key)

  return params


def load_checkpoint_np(npz):
  """Loads a jax pytree from a npz file.

  Args:
    npz: Either path to the checkpoint file (.npz), or a dict-like.

  Returns:
    A pytree that is the checkpoint.
  """
  if isinstance(npz, str):  # If not already loaded, then load.
    npz = npload(npz)
  keys, values = zip(*list(npz.items()))
  checkpoint = recover_tree(keys, values)
  return checkpoint


def load_checkpoint_ts(path, **tsload_kw):
  """Loads a big_vision checkpoint saved by `save_checkpoint_ts`."""
  to_load = path

  try:
    # When passing a general path (not a specific step), get the last available.
    with gfile.GFile(f"{path}-LAST", "r") as f:
      to_load = f"{path}-{f.read().strip()}"
  except Exception:  # pylint:disable=broad-exception-caught
    # Differs based on backend, so blanket catch.
    pass

  return tsload(to_load, **tsload_kw)


def tsload(path, *, tree=None, shardings=None, regex=None):
  """Loads tensorstore-based array-tree from disk.

  If `tree` argument is provided, then array names to load and target structure
  is derived from the tree. If `tree` is None, then array names to load are
  derived from array filenames on the disk, and, optionally, `regex` is applied
  to filter these names. The`tree` argument is then automatically derived from
  array names with `recover_tree` util.

  Arrays are loaded to CPU/TPU/GPU memory as specified by the `shardings`
  argument, which is a pytree of CPU/TPU/GPU shardings (can be mixed within a
  single pytree). `shardings` should a prefix tree of the `tree` argument. We
  automatically broadcast `shardings` to a full `tree`. For example, a user can
  specify `shardings=jax.sharding.SingleDeviceSharing(jax.devices('cpu')[0])`,
  which  will be broadcasted to a full tree.

  Args:
    path: a directory where the checkpoint arrays are stored.
    tree: a target pytree, which defines array names to load and the target tree
      structure. If tree is None, then `tree` is inferred from the names of
      arrays stored on the disk.
    shardings: a prefix pytree (with respect to `tree`) of the target shardings.
    regex: regex to filter array names from the disk, if `tree` is not provided.

  Returns:
    A pytree of loaded arrays that has the same structure as `shardings` arg.
  """
  if (tree is not None) and (regex is not None):
    raise ValueError("If tree is specified, regex filtering is not allowed.")

  if tree is None:
    # Some file-systems (gs://) list folders with a trailing /, get rid of it.
    path_names = set(
        [p.rstrip("/").replace("~", "/") for p in gfile.listdir(path)]
    )
    regex = re.compile(regex) if regex is not None else re.compile(".*")
    path_names = [p for p in path_names if regex.match(p)]
    tree = recover_tree(path_names, [0] * len(path_names))

  names_and_vals, tree_def = tree_flatten_with_names(tree)
  names_to_load, _ = zip(*names_and_vals)

  if shardings is None:
    shardings = jax.sharding.SingleDeviceSharding(
        jax.local_devices(backend="cpu")[0]
    )
  shardings = list(jax.tree.leaves(tree_broadcast(shardings, tree)))

  names_to_load = [
      os.path.join(path, name.replace("/", "~")) for name in names_to_load
  ]
  specs = [array_serial.get_tensorstore_spec(n) for n in names_to_load]
  arrays = array_serial.run_deserialization(shardings, specs, concurrent_gb=64)
  return tree_def.unflatten(arrays)


def tree_get(tree, name):
  """Get an entry of pytree by flattened key name, eg a/b/c, with nice error.

  Args:
    tree: the pytree to be queried.
    name: the path to extract from the tree, see below for examples.

  Returns:
    A few examples:
      tree = {'a': 1, 'b': {'c': 2, 'd': 3}}
      tree_get(tree, 'a') == 1
      tree_get(tree, 'b/c') == 2
      tree_get(tree, 'b') == {'c': 2, 'd': 3}
  """
  flattened = dict(_traverse_with_names(tree, with_inner_nodes=True))
  try:
    return flattened[name]
  except KeyError as e:

    class Msg(str):  # Reason: https://stackoverflow.com/a/70114007/2366315

      def __repr__(self):
        return str(self)

    msg = "\n".join([name, "Available keys:", *flattened, ""])
    # Turn into configdict to use its "did you mean?" error message!
    msg = mlc.ConfigDict(flattened)._generate_did_you_mean_message(name, msg)  # pylint: disable=protected-access
    raise KeyError(Msg(msg)) from e


def _traverse_with_names(tree, with_inner_nodes=False):
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  # Don't output the non-leaf nodes. If the optimizer doesn't have a state
  # the tree leaves can be Nones which was interpreted as a leaf by this
  # function but not by the other functions (like jax.tree.map).
  if tree is None:
    return
  elif isinstance(tree, Mapping):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key], with_inner_nodes):
        yield (key + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  elif isinstance(tree, (list, tuple)):
    for idx in range(len(tree)):
      for path, v in _traverse_with_names(tree[idx], with_inner_nodes):
        yield (str(idx) + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  else:
    yield "", tree


def recover_dtype(a):
  """Numpy's `save` stores bfloat16 type as "void" type, so we recover it."""
  if hasattr(a, "dtype") and a.dtype.type is np.void:
    assert a.itemsize == 2, "Unknown dtype!"
    return a.view(jax.numpy.bfloat16)
  else:
    return a


def tree_broadcast(prefix, target):
  """Broadcasts a prefix tree to a full tree.

  Input-output examples:
  1. prefix: {"x": 10, "y": 20}
     target: {"x": {"a": 1, "b": 2}, "y": 3}

     Result: {"x": {"a": 10, "b": 10}, "y": 20}

  2. prefix: 100
     target: {"x": {"a": 1, "b": 2}, "y": 3}

     Result: {"x": {"a": 100, "b": 100}, "y": 100}

  3. prefix: {"x": 10}
     target: {"x": {"a": 1, "b": 2}, "y": 3}

     Result: ValueError

  Args:
    prefix: prefix pytree.
    target: boradcast target for a prefix tree.

  Returns:
    prefix tree broadcasted to a target tree.
  """

  def _broadcast(leaf, subtree):
    return jax.tree.map(lambda _: leaf, subtree)

  return jax.tree.map(_broadcast, prefix, target)


def npload(fname):
  """Loads `fname` and returns an np.ndarray or dict thereof."""
  # Load the data; use local paths directly if possible:
  if os.path.exists(fname):
    loaded = np.load(fname, allow_pickle=False)
  else:
    # For other (remote) paths go via gfile+BytesIO as np.load requires seeks.
    with gfile.GFile(fname, "rb") as f:
      data = f.read()
    loaded = np.load(io.BytesIO(data), allow_pickle=False)

  # Support loading both single-array files (np.save) and zips (np.savez).
  if isinstance(loaded, np.ndarray):
    return loaded
  else:
    return dict(loaded)


#################
