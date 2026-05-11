# Workflow

1. Capture synchronized left/right image pairs.
2. Verify the physical calibration target dimensions.
3. Run dataset QC and reject weak frames before calibration.
4. Run stereo calibration.
5. Generate rectification maps.
6. Validate rectification numerically using vertical epipolar error.
7. Generate raw-vs-rectified visual overlays.
8. Run 6-DoF fiducial pose estimation.
9. Export CSV/JSON reports and demo media.

The repository is intentionally device-agnostic. Vendor acquisition code should stay outside the public repo.
