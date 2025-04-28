# AligNet
Code for finetuning a pretrained SigLIP model on the AligNet dataset.

## Installation

* Clone Repository

  ``` bash
  git clone https://github.com/google-deepmind/alignet.git
  ```
* Install requirements

  ```bash
  pip install -r alignet/requirments.txt
  ```

## Download data

### AligNet triplets
Download the data: TODO

```bash
wget ALIGNET_DATA_URL_TODO $HOME/alignet
```

### ImageNet
AligNet training depends on the tensorflow_datasets
[`imagenet2012`](https://www.tensorflow.org/datasets/catalog/imagenet2012)
dataset in `array_record` format.

This dataset requires you to download the source data manually into
`download_config.manual_dir` (defaults to
`~/tensorflow_datasets/downloads/manual/`):
`manual_dir` should contain two files: `ILSVRC2012_img_train.tar` and
`ILSVRC2012_img_val.tar`. You need to register on
https://image-net.org/download-images in order to get the link to download the
dataset.

After downloading the files (approx 150GB) the creation of the dataset can be
triggered by running (takes about an hour):

```python
import tensorflow_datasets as tfds
ds = tfds.data_source("imagenet2012", split="train")
```

NOTE: Note the use of `tfds.data_source` rather than `tfds.load`. This is needed
 because otherwise TFDS defaults to generating the dataset in TFRecords format
 which doesn't support random access, and does not work with AligNet finetuning.

## Usage
To run AligNet finetuning on a pretrained SigLIP Vit-B model:

* Navigate to the parent directory of the `alignet` repository.
* Adjust `--cfg.aux.data_dir` to point to the directory containing the AligNet
  triplets.
* Point `--cfg.workdir` to the directory to which checkpoints etc should be
  saved.
* Run:

  ```bash
  python -m kauldron.main \
    --cfg=alignet/configs/siglip.py \
    --cfg.workdir=/tmp/kauldron/workdir \
    --cfg.aux.data_dir=/path/to/alignet/dataset
  ```

## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```
@article{publicationname,
      title={Publication Name},
      author={Author One and Author Two and Author Three},
      year={2025},
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
