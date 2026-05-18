# FilPHANGS

FilPHANGS is an automated pipeline for identifying and characterizing filamentary structures in astronomical images. It was developed for JWST PHANGS observations but is applicable to any continuum image where filamentary dust structures are expected.

The pipeline decomposes an image into physical scales using constrained diffusion, runs the SOAX snake-tracing algorithm at each scale, constructs composite filament maps, fits PSFs along each detected filament, and extracts physical properties including length, line mass, surface density, and curvature.

Associated paper: coming soon.

---

## Prerequisites

### SOAX

SOAX is the underlying filament tracer. Download the batch executable from:
https://www.lehigh.edu/~div206/soax/downloads.html

Binaries are available for Windows and macOS (High Sierra). Note the path to the `.exe` or binary — you will set it in `Main.py`.

### Julia and CloudClean (optional)

Source removal (compact sources such as HII regions and star clusters) uses the Julia package [CloudClean.jl](https://github.com/andrew-saydjari/CloudClean.jl). You do not need to install the CloudClean repository itself — only Julia is required. Download Julia from https://julialang.org/downloads/ and install it with a kernel accessible to Jupyter (`julia-1.11` by default).

Source removal is optional. Set `Rem_sources = False` in your image table to skip it for a given image.

### Python

Python 3.10 or later is required.

---

## Installation

```bash
conda create -n filphangs python=3.11
conda activate filphangs
pip install -r requirements.txt
```

SOAX is a standalone binary and is not installed via pip — see the Prerequisites section above.

---

## Data Preparation

### Image naming

Images placed in `OriginalImages/` must contain the galaxy label and filter band separated by an underscore, e.g. `NGC0628_F770W_someDescriptor.fits`. The pipeline will standardize names on first run — additional descriptors in the filename are preserved.

### Directory structure

Create a base directory for all pipeline outputs. Only the `OriginalImages/` subdirectory and the two metadata files need to exist before the first run. Everything else is created automatically.

```
FilPHANGS_base/
├── OriginalImages/          # Input FITS files go here
├── ImageData.xlsx           # Per-image metadata (see below)
├── SoaxParams.txt           # SOAX parameter file (provided in repo)
└── Figures/                 # Created automatically
```

After the first run, each image gets its own output directory:

```
FilPHANGS_base/
└── NGC0628_F770W/
    ├── BkgSubDivRMS/        # Background-subtracted SNR images used by SOAX
    ├── BlockedPng/          # Downsampled PNG inputs to SOAX
    ├── CDD/                 # Scale-decomposed FITS images (16 pc, 32 pc, ...)
    ├── Composites/          # Stacked filament maps across 10 SOAX runs per scale
    ├── SoaxOutput/          # Raw SOAX text output and reconstructed FITS per scale
    ├── Source_Removal/      # Source masks and inpainted images (if enabled)
    └── SyntheticMap/        # PSF-based synthetic filament maps and property CSVs
```

### ImageData.xlsx

The pipeline reads per-image metadata from an Excel file. Each row corresponds to one image. Required columns:

| Column | Description | Example |
|---|---|---|
| `label` | Galaxy name, matched to filename | `NGC0628` |
| `Band` | Filter name, matched to filename | `F770W` |
| `Telescope` | Telescope identifier | `JWST` |
| `INSTR` | Instrument identifier | `MIRI` |
| `Image_Type` | Descriptor appended to standardized filename | `lev3` |
| `current_dist` | Distance in Mpc | `9.84` |
| `res` | Angular resolution in arcsec | `0.269` |
| `pixscale` | Pixel scale in arcsec/pixel | `0.11` |
| `Power of 2 min` | Minimum CDD decomposition scale as log2(pc) | `4` (= 16 pc) |
| `Power of 2 max` | Maximum CDD decomposition scale as log2(pc) | `8` (= 256 pc) |
| `Rem_sources` | Whether to run compact source removal | `True` / `False` |
| `SSFR` | Specific star formation rate (log scale) | `-10.5` |
| `Inclination Angle` | Galaxy inclination in degrees | `7.0` |

The physical pixel scale used internally is `pixscale (arcsec/px) * 4.848 * distance (Mpc)` = pc/px.

---

## Configuration

Open `Identification/Main.py` and update the paths and parameters in the configuration block near the top of the `if __name__ == "__main__"` section.

### Paths

```python
base_dir        = Path(...)   # Base output directory (must contain OriginalImages/)
csv_path        = Path(...)   # Path to ImageData.xlsx
param_file_path = Path(...)   # Path to SoaxParams.txt
batch_path      = Path(...)   # Path to the SOAX batch executable
julia_path      = Path(...)   # Path to JuliaCloudClean_Output1.ipynb (source removal only)
region_dir_path = Path(...)   # Optional: directory of region mask FITS files
dynamic_alphaCO_path = Path(...)  # Optional: directory of alphaCO conversion maps
```

### Key parameters

```python
# Filament detection threshold
min_aspect_ratio = 8.2
```
The minimum length-to-width ratio for a detected filament. Filament width is fixed to the 16 pc PSF resolution element (`16 pc / ScalePix` in pixels). At the reference pixel scale of 5.25 pc/px, a ratio of 8.2 corresponds to a minimum snake length of 25 pixels in SOAX. This threshold increases conservatively at larger (more heavily downsampled) scales — see the log output for the effective ratio at each scale.

```python
min_fg_int = 1638       # SOAX minimum foreground intensity (0–65535 scale)
noise_min  = 1e-2       # Floor on background RMS noise (prevents division by near-zero)
flatten_perc = 90       # Percentile used for the arctan intensity rescaling before SOAX
min_intensity = 0       # Pixels below this value in the original image are set to zero
```

---

## Running

```bash
cd Identification
python Main.py
```

A log file `filphangs.log` is written alongside all console output. A typical galaxy takes 1–3 hours end-to-end and produces approximately 3 GB of output.

---

## Output

The primary scientific output for each image and scale is a CSV file in `SyntheticMap/` containing one row per detected filament:

| Column | Description |
|---|---|
| `Length_{scale}` | Filament length in parsecs |
| `Line_Density_{scale}` | Line mass in solar masses per parsec |
| `Mass_{scale}` | Total molecular mass in solar masses |
| `Curvature_{scale}` | Angular width of the orientation distribution (radians) |
| `Regions_{scale}` | Galaxy region index from the region mask (−1 if no mask provided) |

A PSF-based synthetic image of the filament network is also saved as a FITS file in `SyntheticMap/`.

---

## Optional features

**Region masks** — if a path to region mask FITS files is provided, each detected filament is assigned to the dominant region it overlaps, enabling comparisons between environments (e.g. arm vs. interarm). Region masks must follow the PHANGS region map format.

**Dynamic alphaCO** — by default, molecular masses are computed using a constant CO-to-H2 conversion factor of 5.5 (see the associated paper for justification). If a directory of spatially-resolved alphaCO maps is provided, pixel-level conversion factors are used instead.

---

## Supported bands

Source removal and property extraction are validated for JWST MIRI bands F770W, F1000W, F1130W, F2100W and NIRCam bands F200W, F300M, F335M, F360M. Filament networks can be identified (without property extraction) in attenuation maps and non-JWST images.

---

## Troubleshooting

**SOAX produces no output** — check that `batch_path` points to the correct executable and that `min_fg_int` is not set too high for your image's dynamic range. SOAX output files appear in `SoaxOutput/<scale>/`.

**Julia kernel not found** — the notebook launcher expects a kernel named `julia-1.11`. After installing Julia, run `julia -e 'using IJulia; installkernel("Julia")'` to register it, then check the kernel name in Jupyter and update `JuliaCloudClean_Output1.ipynb` if it differs.

**No CDD files produced** — the scale filter in `Modified_Constrained_Diffusion.py` requires that each physical scale is both resolved (above 1.33× the PSF FWHM) and smaller than half the image. Check that `Power of 2 min` and `Power of 2 max` in `ImageData.xlsx` are appropriate for your pixel scale and distance.

**`filphangs.log` grows large** — the log file is opened in append mode (`mode='a'`). Delete or truncate it between full pipeline runs if disk space is a concern.

**SIP distortion warnings from astropy** — these are harmless for images that have already been drizzled. They can be silenced by adding `-SIP` to the CTYPE keywords in the FITS header or by suppressing the warning class in `Main.py`.

---

## Attribution

If you use FilPHANGS in your work, please cite the associated paper (link forthcoming).

Curvature calculations use algorithms from [FilFinder](https://github.com/e-koch/FilFinder) (Koch et al.).
