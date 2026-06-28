"""Helpers used at inference time only.

These wrap third-party models (Depth-Anything-V2, GroundingDINO+SAM) that
are vendored under the top-level `infer_helpers/` directory and are NOT
required to train or load the SmolVLA policy. Imports are intentionally
local to each module so that `import lerobot` keeps working without the
inference env installed.
"""
