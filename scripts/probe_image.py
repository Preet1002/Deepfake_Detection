"""Score one image under many preprocessing variants to find what defeats it.

When a detector calls an obvious fake "real", the cause is usually one of:

  * resolution - training crops were 256px; a 1024px source resized straight to
    224 loses the high-frequency artefacts the SRM stream keys on
  * compression - training images were all JPEG; a pristine PNG sits off the
    manifold the model ever saw
  * framing - the datasets ship tight aligned face crops, so a portrait with
    shoulders and background is a different composition
  * the generator itself - a genuinely unseen synthesis method, which no
    inference-time fix will repair

Each hypothesis predicts a different pattern here, so this tells you which one
you are looking at instead of guessing.

    python -m scripts.probe_image path/to/face.png \
        --checkpoint runs/dfnet_multi/best.pt
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image, ImageFilter

from src.predict import Detector


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def resize(img: Image.Image, size: int, resample) -> Image.Image:
    return img.resize((size, size), resample)


def centre_crop(img: Image.Image, fraction: float) -> Image.Image:
    w, h = img.size
    side = int(min(w, h) * fraction)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def variants(img: Image.Image) -> list[tuple[str, Image.Image]]:
    """(name, image) pairs, each isolating one hypothesis."""
    out: list[tuple[str, Image.Image]] = [("as-is", img)]

    # Resolution: the datasets ship 256px crops. Passing through that size
    # reproduces the downsampling the training images already went through.
    for size in (512, 256, 128):
        out.append((f"resize {size}px bicubic", resize(img, size, Image.BICUBIC)))
    out.append(("resize 256px lanczos", resize(img, 256, Image.LANCZOS)))
    out.append(("resize 256px nearest", resize(img, 256, Image.NEAREST)))

    # Compression: every training image was JPEG at some quality.
    for quality in (95, 75, 50):
        out.append((f"jpeg q{quality}", jpeg(img, quality)))
    out.append(("resize 256px + jpeg q75", jpeg(resize(img, 256, Image.BICUBIC), 75)))

    # Framing: tighter crops approach the aligned dataset composition.
    for fraction in (0.85, 0.7, 0.55):
        out.append((f"centre crop {fraction:.2f}", centre_crop(img, fraction)))

    # Sanity checks: a stable model should barely move on these.
    out.append(("hflip", img.transpose(Image.FLIP_LEFT_RIGHT)))
    out.append(("blur 0.5", img.filter(ImageFilter.GaussianBlur(0.5))))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--checkpoint", default="runs/dfnet_multi/best.pt")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--detect-faces", action="store_true",
                        help="run face detection first (default: treat the "
                             "image as an already-cropped face)")
    args = parser.parse_args(argv)

    detector = Detector(args.checkpoint)

    for path in args.images:
        image = Image.open(path).convert("RGB")
        print(f"\n=== {path}  ({image.width}x{image.height}, "
              f"{Path(path).suffix.lstrip('.').upper()}) ===")
        print(f"{'variant':<26} {'P(fake)':>8}  verdict")

        baseline = None
        for name, variant in variants(image):
            result = detector.predict(variant, detect_faces=args.detect_faces,
                                      explain=False)
            p = result.fake_probability
            if baseline is None:
                baseline = p
            verdict = "FAKE" if p >= args.threshold else "real"
            shift = "" if name == "as-is" else f"  ({p - baseline:+.3f})"
            print(f"{name:<26} {p:8.4f}  {verdict}{shift}")

    print(f"\n(threshold {args.threshold}; shifts are relative to 'as-is')")
    print("Reading it: if a resize or JPEG variant crosses into FAKE, the model "
          "works but\nthe input was off-distribution - fixable in preprocessing. "
          "If nothing moves,\nthe generator itself is unseen and only training "
          "data fixes it.")


if __name__ == "__main__":
    main()
