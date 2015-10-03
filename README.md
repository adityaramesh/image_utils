# Overview

This repository contains utilities for image classification in Torch. Features:
- Preprocessing:
  - Color space conversion.
  - Local and global contrast normalization.
  - ZCA whitening.
- Training:
  - Model training script specialized for image classification.

# Supported File Formats

Torch7 and HDF5 files are supported for both input and output. Torch7 files
must have the extension "t7", and HDF5 files must have the extension "hdf5"
(please open an issue if this is a problem for you).

Both input and output files should follow the following format:
  - "inputs" (Torch7) or "/inputs" (HDF5): A `count x channels x width x height`
  array of input images.
  - "targets" (Torch7) or "/targets" (HDF5) [optional]: A 1D array of integers
  representing the classes of the inputs.
  - "classes" (Torch7) or "/classes" (HDF5) [should be present only if
  `targets` is defined]: If the file is in Torch7 format, then this is an
  integer describing the total number of classes. If the file is in HDF5
  format, then this is a 1D array with a single element that describes the same
  information.  The awkward restriction for HDF5 files is due to limitations in
  the [torch-hdf5](https://github.com/deepmind/torch-hdf5) package. Please file
  an issue if you have a better solution.
