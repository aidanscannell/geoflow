#!/usr/bin/env python3
import typing
from typing import Tuple

import tensor_annotations.tensorflow as ttf
from tensor_annotations import axes

# Axes types
NumData = typing.NewType("NumData", axes.Axis)
InputDim = typing.NewType("InputDim", axes.Axis)
OutputDim = typing.NewType("OutputDim", axes.Axis)
NumInducing = typing.NewType("NumInducing", axes.Axis)
One = typing.NewType("One", axes.Axis)

InputData = ttf.Tensor2[NumData, InputDim]
OutputData = ttf.Tensor2[NumData, OutputDim]

JacMeanAndVariance = Tuple[
    ttf.Tensor3[NumData, InputDim, OutputDim],
    ttf.Tensor4[NumData, OutputDim, InputDim, InputDim],
]
