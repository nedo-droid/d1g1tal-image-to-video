# Image-to-video renderer v17

This version separates the three environment systems:

- `water`: uses the white annotation in the environment mask as a strict gate. Highlight opacity is fixed by config at `0.50`.
- `palms`: uses the green annotation to define allowed regions, then finds dark palm-like structures with Hough trunk candidates and compares them against the palm sketches. Every detected instance gets its own local bend/rotation around its own pivot.
- `eagle`: manual trace paths are preserved, and an automatically generated outline-completion stage guarantees that the complete geometric contour exists before the logo reveal. Neon fade starts only after the reveal/hold stage.

## High-quality logo removal

A single flattened image does not contain the pixels that were behind an opaque logo. For a clean result, provide one of:

1. `--background clean_background.png` — preferred.
2. `--neighbor-frame-dir ./neighbors` — a directory containing real source frames from the same scene before/without the logo.

Do not put generated output frames into `neighbors`.

Without either input, v17 uses an explicitly approximate `NS + mirror` fallback. It is intentionally not treated as a high-quality clean plate.

## Example

```bash
python3 image_to_video_manual_v17.py 32510.png \\
  --config eagle_scene_v17.json \\
  --environment-mask 32529.png \\
  --palm-templates palm_sketches_v15.json \\
  --background clean_background.png \\
  --duration 12 --fps 24 --width 1536 --height 864 \\
  --output eagle_final.mp4
```

## Debug output

Use `--debug-dir v17_debug` to write:

- `logo_remove_mask.png`
- `neon_completion_outline.png`
- `palm_all.png`
- `palm_group_*.png`
- `water_annotation.png`
- `clean_plate.png`

The renderer reports the exact timeline and number of detected palm instances in JSON.
