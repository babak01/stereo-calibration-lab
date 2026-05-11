# Validation metrics

## RMS reprojection error

Calibration RMS measures how well detected image points fit the estimated camera model. It is useful, but not enough by itself.

## Baseline

The stereo translation vector gives the distance between camera centers. If a physical baseline is known, compare the estimated baseline against it.

## Vertical epipolar error

After stereo rectification, corresponding points should lie on the same horizontal image row.

```text
vertical_error_px = abs(y_left_rectified - y_right_rectified)
```

Report mean, median, P95, and maximum vertical error.

## Holdout validation

Use images not used during calibration to check whether rectification generalizes.
