"""
Performs ZCA whitening on images.
"""

import os
import h5py
import argparse

import numpy as np
from scipy.linalg import svd

parser = argparse.ArgumentParser(description="Performs ZCA whitening on images.")
parser.add_argument("-epsilon", type=float, nargs=1, default=1e-7,
    help="For conditioning of the scaling transformation used in ZCA.")
parser.add_argument("-max_sample_size", type=int, nargs=1, default=4000,
    help="Maximum sample size to use for ZCA.")

parser.add_argument("-input", type=str, nargs=1, help="Path to input file.")
parser.add_argument("-output", type=str, nargs=1, help="Path to output file.")
parser.add_argument("-stats_input", type=str, nargs=1,
    help="Path to HDF5 file with precomputed mean and SVD.")
parser.add_argument("-stats_output", type=str, nargs=1,
    help="Path to HDF5 file in which to save mean and SVD.")

args            = parser.parse_args()
input_fp        = args.input[0]
output_fp       = args.output[0]
max_sample_size = args.max_sample_size[0]
stats_input_fp  = None if not args.stats_input else args.stats_input[0]
stats_output_fp = None if not args.stats_output else args.stats_output[0]

assert(args.epsilon > 0 and args.epsilon < 1)

def ensure_not_exists(fp):
    if os.path.isfile(fp):
       raise FileExistsError("File `{}` already exists.".format(fp))

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
mean, s, v = None, None, None

if stats_input_fp:
    stats_file = h5py.File(stats_input_fp)
    mean = np.array(stats_file["mean"])
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

    print("Computing mean.")
    mean = np.mean(sample, axis=0)
    print("Centering input.")
    sample = sample - mean
    print("Computing SVD.")
    _, s, v = svd(sample, full_matrices=False, overwrite_a=True)

    if stats_output_fp:
        print("Saving mean and SVD.")
        stats_file = h5py.File(stats_output_fp)
        stats_file["mean"] = mean
        stats_file["singular values"] = s
        stats_file["eigenvectors"] = v

print("Computing ZCA transformation.")
s_cond = np.sqrt(s**2 + args.epsilon)
w_zca = np.dot(v.T, np.dot(np.diag(1 / s_cond), v))

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
