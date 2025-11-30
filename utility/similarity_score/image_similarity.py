#!/usr/bin/env python3
import argparse
import os
import glob

from PIL import Image
import numpy as np
import torch


def parse_args():
    # python image_similarity.py --help.
    parser = argparse.ArgumentParser(
        description=(
            "Compute a similarity score between images based on per-pixel variance.\n"
            "Precondition: All images must have the same size."
            "Computation can run on GPU if available."
        )
    )
    # python image_similarity.py --image-dir path/to/images.
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing input images "
             "(default: 'images' folder).",
    )
    # python image_similarity.py --format "*.jpg,*.png".
    parser.add_argument(
        "--format",
        default="*.jpg,*.jpeg,*.png,*.bmp,*.webp",
        help="Comma-separated list of glob patterns to find images (default: %(default)s).",
    )
    # python image_similarity.py --output-npy similarity.npy.
    parser.add_argument(
        "--output-npy",
        default="similarity.npy",
        help="Path to save per-pixel variance matrix as NumPy .npy (default: %(default)s).",
    )
    # python image_similarity.py --cpu.
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if a CUDA GPU is available.",
    )
    return parser.parse_args()


def find_images(image_dir, formats):
    """
    Find image files in the given directory matching the specified formats.
    Returns a sorted list of unique file paths.
    """
    formats = [p.strip() for p in formats.split(",") if p.strip()]
    paths = []
    for fmt in formats:
        paths.extend(glob.glob(os.path.join(image_dir, fmt)))
    paths = sorted(set(paths))
    return paths


def load_images_as_array(image_paths):
    """
    Load images as float arrays and ensure they all have the same size.
    Returns:
      imgs: np.ndarray of shape (N, H, W, 3)
      size: (W, H)
    """
    imgs = []
    sizes = set()

    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}.")
            continue

        sizes.add(img.size)  # (width, height)
        imgs.append(np.asarray(img, dtype=np.float32))

    # Safety Check 1: if any images were loaded.
    if not imgs:
        print("No valid images could be loaded.")
        return None, None
    
    # Safety Check 2: all images must have the same size.
    if len(sizes) != 1:
        print("Error: Not all images have the same size.")
        print(f"Found sizes: {sizes}.")
        print("Please resize or crop your images so they all match.")
        return None, None

    stack = np.stack(imgs, axis=0)  # (N, H, W, 3)
    return stack, list(sizes)[0]


def compute_per_pixel_variance(images_tensor):
    """
    Compute per-pixel variance across a stack of images.
    Parameters:
        images_tensor: torch.Tensor of shape (N, H, W, 3), 
                       dtype float32 or float64, on CPU or GPU.
    Returns:
        var_map: torch.Tensor of shape (H, W), 
                 dtype float64, on the same device as images_tensor.
                 Contain the per-pixel variance averaged over RGB channels.
    """
    # Convert to float64 for better numerical precision.
    x = images_tensor.to(torch.float64)  # (N, H, W, 3)

    # Variance across images (dim=0) with ddof=0 (unbiased=False),
    # matching numpy.var(..., axis=0, ddof=0).
    var_rgb = torch.var(x, dim=0, unbiased=False)  # (H, W, 3)

    # Average variance across RGB channels -> (H, W).
    var_map = var_rgb.mean(dim=-1)
    return var_map


def main():
    args = parse_args()

    # Default image directory: 'images'.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_image_dir = os.path.join(script_dir, "images")
    image_dir = args.image_dir or default_image_dir

    print(f"Using image directory: {image_dir}")

    image_paths = find_images(image_dir, args.format)
    if not image_paths:
        print(f"No images found in {image_dir} with format(s) {args.format}.")
        return

    print(f"Found {len(image_paths)} image file(s). Loading and checking sizes...")

    stack, _ = load_images_as_array(image_paths)
    if stack is None:
        return

    n, h, w, c = stack.shape
    print(f"Loaded {n} valid image(s) of size {w}x{h}, channels={c}.")

    # Decide device: GPU if available and not forced to CPU, otherwise CPU
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Compute per-pixel variance map
    print("Computing per-pixel variance...")
    var_map = compute_per_pixel_variance(torch.from_numpy(stack).to(device))

    # Save the per-pixel variance matrix
    np.save(args.output_npy, var_map.cpu().numpy())
    print(f"Saved per-pixel variance matrix to {args.output_npy}.")

    # Final similarity score: mean of per-pixel variances
    final_score = float(var_map.mean())

    print("\n=== Final similarity score ===")
    print(
        f"Mean per-pixel variance across all images: {final_score:.6f}\n"
        "(Lower means more similar; 0.0 means all images are identical.)"
    )


if __name__ == "__main__":
    main()
