# Image-Filter-Engine
# Image Processing with NumPy & Matplotlib

A lightweight image processing pipeline built from scratch using NumPy array operations. No OpenCV. No PIL. Just raw array math.

---

## What It Does

The script loads an image and applies three transformations:

| Transformation | Description |
|---|---|
| Grayscale | Converts RGB to single-channel using luminance weights |
| Brightness x1.5 | Scales all pixel values up by 50% |
| Brightness x0.5 | Scales all pixel values down by 50% (darkening) |

All four versions (original + 3 transformations) are displayed side-by-side in a single Matplotlib figure.

---

## How It Works

### Image Representation

When loaded via `plt.imread()`, an image becomes a NumPy array of shape `(H, W, 3)` — height × width × RGB channels. Each pixel value is an integer in the range `[0, 255]`.

```
image.shape → (H, W, 3)
image.dtype → uint8
```

---

### 1. Grayscale Conversion

```python
def to_Grayscale(image):
    weights = np.array([0.2126, 0.7152, 0.0722])
    gray = np.dot(image, weights)
    return gray.astype(np.uint8)
```

**How it works:**

Each RGB pixel `[R, G, B]` is collapsed into a single value using a weighted dot product:

```
gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
```

These weights come from the **ITU-R BT.709 standard** (the same standard used in HDTV). They reflect human eye sensitivity — we perceive green most strongly, red moderately, and blue the least.

`np.dot(image, weights)` applies this across all `H × W` pixels simultaneously, producing an output of shape `(H, W)`.

The result is cast to `uint8` to keep values in the valid `[0, 255]` range.

---

### 2. Brightness Control

```python
def brightness_control(image, factor=1.5):
    bright = image.astype(np.float64) * factor
    bright = np.clip(bright, 0, 255)
    return bright.astype(np.uint8)
```

**How it works:**

- The image is cast to `float64` first — necessary because `uint8` arithmetic overflows silently (e.g. `200 * 1.5 = 44` in uint8 due to wrap-around).
- Every pixel value is multiplied by `factor`. Values above 1.0 brighten; below 1.0 darken.
- `np.clip(bright, 0, 255)` clamps all values back into the valid range, preventing blown-out whites or negative values.
- Cast back to `uint8` for display.

**Usage:**

```python
brightened = brightness_control(image, factor=1.5)   # +50% brightness
darkened   = brightness_control(image, factor=0.5)   # -50% brightness
```

---

### 3. Visualization

```python
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].imshow(image);                              axes[0].set_title("Original")
axes[1].imshow(to_Grayscale_image, cmap='gray');    axes[1].set_title("Grayscale")
axes[2].imshow(fifty_percent_brightness);           axes[2].set_title("Brightness x1.5")
axes[3].imshow(fifty_percent_dark);                 axes[3].set_title("Brightness x0.5")

for ax in axes: ax.axis('off')
plt.tight_layout()
plt.show()
```

Creates a 1×4 grid of subplots. The grayscale image uses `cmap='gray'` because it's a 2D array — without this, Matplotlib would apply a default false-color map.

---

## Output

```
┌──────────┬──────────┬──────────────────┬────────────────┐
│ Original │Grayscale │ Brightness x1.5  │Brightness x0.5 │
└──────────┴──────────┴──────────────────┴────────────────┘
```

---

## Requirements

```
numpy
matplotlib
scikit-image
```

Install:

```bash
pip install numpy matplotlib scikit-image
```

---

## Usage

1. Clone the repo or copy the script.
2. Replace the image path:

```python
image = plt.imread('/path/to/your/image.jpg')
```

3. Run:

```bash
python image_processing.py
```

---

## Project Structure

```
.
├── image_processing.py   # Main script
└── README.md
```

---

## Key Concepts Demonstrated

- NumPy array broadcasting and vectorized operations
- Float/integer dtype management to avoid overflow
- Weighted dot product for channel reduction
- `np.clip()` for value clamping
- Matplotlib subplot layout and colormap handling
