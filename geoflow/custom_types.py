#!/usr/bin/env python3
import typing
from tensor_annotations import axes

# Axes types
NumData = typing.NewType("NumData", axes.Axis)
InputDim = typing.NewType("InputDim", axes.Axis)
OutputDim = typing.NewType("OutputDim", axes.Axis)
# NumInducing = typing.NewType("NumInducing", axes.Axis)
One = typing.NewType("One", axes.Axis)
