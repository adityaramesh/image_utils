"""
Performs ZCA whitening on images. Note: the input should have been processed
with image-wide global contrast normalization before being whitened. This
script does not do any centering, as
  1. Subtracting the mean image is not the appropriate form of preprocessing
  for ZCA, and
  2. Global contrast normalization is already implemented by `normalize.py`.
"""

import os
import sys
import h5py
import argparse

import math
import numpy as np
from scipy.linalg import svd

parser = argparse.ArgumentParser(description="Performs ZCA whitening on images.")
parser.add_argument("-epsilon", type=float, nargs=1, default=[1e-7],
    help="For conditioning of the scaling transformation used in ZCA.")
parser.add_argument("-max_sample_size", type=int, nargs=1, default=[4000],
    help="Maximum sample size to use for ZCA.")

parser.add_argument("-input", type=str, nargs=1,
    help="Path to input file.")
parser.add_argument("-output", type=str, nargs=1,
    help="Path to output file.")
parser.add_argument("-stats_input", type=str, nargs=1,
    help="Path to HDF5 file with precomputed SVD.")
parser.add_argument("-stats_output", type=str, nargs=1,
    help="Path to HDF5 file in which to save SVD.")

args            = parser.parse_args()
input_fp        = args.input[0]
output_fp       = None if not args.output else args.output[0]
epsilon         = args.epsilon[0]
max_sample_size = args.max_sample_size[0]
stats_input_fp  = None if not args.stats_input else args.stats_input[0]
stats_output_fp = None if not args.stats_output else args.stats_output[0]

assert(epsilon >= 0 and epsilon < 1)

if not output_fp and not stats_output_fp:
    print("Nothing to do: both `output` and `stats_output_fp` are undefined.",
        file=sys.stderr)
    sys.exit()

def ensure_not_exists(fp):
    if os.path.isfile(fp):
       raise FileExistsError("File `{}` already exists.".format(fp))

if output_fp:
    ensure_not_exists(output_fp)
if stats_output_fp:
    ensure_not_exists(stats_output_fp)

print("Loading data.")
input_file = h5py.File(input_fp)
inputs     = np.array(input_file["inputs"], dtype="float64")

count, channels, width, height = inputs.shape[0], 0, 0, 0

if len(inputs.shape) == 3:
    channels = 1
    width    = inputs.shape[1]
    height   = inputs.shape[2]
elif len(inputs.shape) == 4:
    channels = inputs.shape[1]
    width    = inputs.shape[2]
    height   = inputs.shape[3]
    assert(channels == 3)
else:
    raise RuntimeError("Input must either be 3D or 4D.")

inputs = inputs.reshape(count, channels * width * height)
s, v = None, None

if stats_input_fp:
    stats_file = h5py.File(stats_input_fp)
    s = np.array(stats_file["singular values"])
    v = np.array(stats_file["eigenvectors"])
else:
    sample = None
    assert(max_sample_size > 0)

    if max_sample_size < count:
        print("Using {}/{} random instances for ZCA".format(max_sample_size, count))
        perm = np.random.permutation(count)[:max_sample_size]
        sample = inputs[perm]
    else:
        print("Using all {} instances for ZCA.".format(count))
        sample = inputs

    print("Computing SVD.")
    _, s, v = svd(sample, full_matrices=False, overwrite_a=True)

    if stats_output_fp:
        print("Saving SVD.")
        stats_file = h5py.File(stats_output_fp)
        stats_file["singular values"] = s
        stats_file["eigenvectors"] = v

if output_fp:
    print("Computing ZCA transformation.")
    scale = None

    # Note: lambda_i = s_i^2 / (n - 1). If epsilon > 0, then we must square the
    # singular values before applying epsilon.
    if epsilon == 0:
        scale = np.diag(math.sqrt(sample.shape[0] - 1) / s)
    else:
        scale = np.diag(np.pow(s**2 / (sample.shape[0] - 1) + epsilon, -0.5))

    w_zca = np.dot(v.T, np.dot(scale, v))

    print("Whitening data.")
    inputs = np.dot(inputs, w_zca.T)

    if channels == 1:
        inputs = inputs.reshape(count, width, height)
    else:
        inputs = inputs.reshape(count, channels, width, height)

    print("Saving data.")
    output_file = h5py.File(output_fp)
    output_file["inputs"] = inputs

    for ds in ["targets", "classes"]:
        if ds in input_file:
            input_file.copy(ds, output_file)
