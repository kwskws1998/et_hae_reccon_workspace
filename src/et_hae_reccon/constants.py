"""Shared constants for emotion eye-tracking heatmap modeling."""

FEATURE_NAMES = ["nFix", "FFD", "GPT", "TRT", "fixProp"]
TRT_FEATURE = "TRT"
TRT_INDEX = FEATURE_NAMES.index(TRT_FEATURE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EPS = 1e-8
