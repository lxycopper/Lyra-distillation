# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identifiers for tensor / metadata slots produced by video datasets (e.g. Radym)."""

from enum import Enum


class DataField(str, Enum):
    """String-valued enum so instances work as stable dict keys."""

    IMAGE_RGB = "IMAGE_RGB"
    CAMERA_C2W_TRANSFORM = "CAMERA_C2W_TRANSFORM"
    CAMERA_INTRINSICS = "CAMERA_INTRINSICS"
    METRIC_DEPTH = "METRIC_DEPTH"
    DYNAMIC_INSTANCE_MASK = "DYNAMIC_INSTANCE_MASK"
    BACKWARD_FLOW = "BACKWARD_FLOW"
    OBJECT_BBOX = "OBJECT_BBOX"
    CAPTION = "CAPTION"
