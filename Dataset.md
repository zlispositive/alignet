# AligNet dataset

AligNet is a synthetically generated dataset of image triplets and corresponding
human-like triplet odd-one-out choices.

## Objective

Alignment with human mental representations is becoming central to
representation learning: we want neural network models that perform well on
downstream tasks **and** align with the hierarchical nature of human semantic
cognition. We believe that aligning neural network representations with human
conceptual knowledge will lead to models that generalize better, are more
robust, safer, and practically more useful. To obtain such models, we generated
a synthetic human-like similarity judgment dataset on a much larger scale than
has previously been possible.

## Details

AligNet is a dataset of triplets and corresponding odd-one-out choices.
Each triplet contains 3 image filenames (the images are sampled from ImageNet)
and the predicted similarity between those three images (obtained from a
pre-trained neural network)
To increase the reproducibility of our research, we split AligNet into a
training and a validation set. The train split `alignet_train.npz` contains 10M
triplets and the validation split `alignet_valid.npz` contains 10k triplets.
The files are stored in
[Numpy’s compressed array format](ttps://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).
Each file contains three arrays of *n* entries each, where `n=10M` for training
and `n=10k` for validation. Row *i* describes the *i*th triplet. Note that
within each triplet we sorted the images such that the last image is always the
one that is most dissimilar to the other two (i.e., the "odd-one-out"),
according to a prediction made by a model we trained (see the
[AligNet paper](https://arxiv.org/pdf/2409.06509) for details).

* `filenames`: (n, 3) strings: Identifies the images used for this triplet.
  Each row contains the names of image files from the ImageNet2012 dataset as
  [filename0, filename1, filename2], where filename2 is the image that is
  typically considered the "odd one out" of the triplet.
* `similarities`: (n, 3) floats: the similarity values of the three pairs of
  images calculated using the pretrained model representations:
  `[s01, s02, s12]`, where `sij` is the similarity between image i and image j.
   Note that the data was sorted such that `s12 < s01` and `s12 < s02`.
* `indices`: The indices of the three images in a
  [`tfds.datasource`](https://www.tensorflow.org/datasets/api_docs/python/tfds/data_source)
  of `imagenet2012`. Note that this matrix is redundant and allows easier access
  to the data (without going through the filenames) when using tfds.

Please see the [AligNet paper](https://arxiv.org/pdf/2409.06509) for details on how the similarities have been computed.

## Supplementary Information

In addition to the official AligNet dataset, we provide other versions of the
data that we think might be useful to the broader community for running
additional experiments. These are other versions of the AligNet data that we
used for ablation studies. Concretely, we used variants that contain the
penultimate (or embedding) layer activations using:

* 3 different ways to sample triplets, depending on the imagenet labels of the
  images in the class:
    * *between-class*: all 3 images correspond to three different classes (this
    is similar to vanilla random sampling)
    * *class-border*: 2 images are sampled (without replacement) from the same
    class and one from a different
    * *within-class*: all images in a triplet are sampled (without replacement)
    from the same class

## Representations

For each ImageNet image, we include the last-layer activations of the
open-source foundation model (so400m-siglip-webli384) used to compute the
triplet similarities, as well as the cluster assignments obtained from
clustering these activations into 500 clusters using k-Means.

## Download Links

TBA

## Code
This repository contains the code to create AligNet from the last-layer
activations of various ImageNet models as input.

* The *colab* folder contains the colabs necessary to sample triplets of image
 files according to various criteria.
* The *uncertainty_distillation* folder contains the code to train an
uncertainty distillation or glocal transform.
* The *response_generation* code uses the generated triplets and transforms to
create the final transform.

## License

The AligNet dataset is under the
[CC-BY License](https://creativecommons.org/licenses/by/4.0/), and the
accompanying code is provided under an
[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
Other parts of the datasets are under the original license of their sub-parts.

## Citation

If you use this dataset, we’d appreciate if you could cite the corresponding
paper as follows:

```
@article{muttenthaler2024aligning,
  title={Aligning Machine and Human Visual Representations across Abstraction Levels},
  author={Muttenthaler, Lukas and Greff, Klaus and Born, Frieda and Spitzer, Bernhard and Kornblith, Simon and Mozer, Michael C and M{\"u}ller, Klaus-Robert and Unterthiner, Thomas and Lampinen, Andrew K},
  journal={arXiv preprint arXiv:2409.06509},
  year={2024}
}
```

## Dataset Metadata

The following table is necessary for this dataset to be indexed by search
engines such as [Google Dataset Search](https://g.co/datasetsearch).
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">AligNet Dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/google-research/kauldron_projects/tree/master/alignet</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/google-research/kauldron_projects/tree/master/alignet</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">
      A dataset of synthetic Human Preference Triplets based on ImageNet.
      </code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">DeepMind</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/DeepMind</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">TBA</code></td>
  </tr>
</table>
</div>

This is not an officially supported Google product.
