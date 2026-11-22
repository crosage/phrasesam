# PhraseSAM

PhraseSAM selects foreground phrases from SAM 3 responses on an unlabeled RGB image pool, calibrates class-correction rules, and uses them to segment new images. The code starts from the recorded LLM candidate phrases, including background. The vocabulary file covers all eight OVR-S8 datasets and includes the saved main and separate-pool profiles from the paper.

## Setup

Use Python 3.10 or 3.11 and a CUDA-enabled PyTorch installation. Install the dependencies for the SAM 3 image model and obtain the SAM 3 checkpoint from [Meta](https://huggingface.co/facebook/sam3). The code uses the SAM 3 implementation in [SegEarth-OV3](https://github.com/earth-insights/SegEarth-OV-3), at revision `87e5cc18c8bc9c0b4713b975587281ac1c97c79a`. Pass the checkout directory to `--sam3-root`.

```bash
python -m pip install -r requirements.txt
```

## Predict with a saved profile

```bash
python phrasesam.py predict --dataset uavid --profile main \
  --sam3-root /path/to/SegEarth-OV-3 --checkpoint /path/to/sam3.pt \
  --images /path/to/rgb --output predictions
```

Use `--profile separate` for the paper's independently configured profile. Each output PNG has the original input dimensions and stores zero-based class IDs in the order of `class_names` in `vocabularies.json`. Inputs can be a single image or a flat directory of RGB images. Use the same evaluated tiles as in the paper to reproduce tiled benchmark predictions.

## Fit a profile from an unlabeled image pool

```bash
python phrasesam.py configure --dataset uavid \
  --sam3-root /path/to/SegEarth-OV-3 --checkpoint /path/to/sam3.pt \
  --images /path/to/unlabeled-pool --output uavid.json
```

The configuration command uses all supplied pool images, keeps all background phrases, screens the foreground candidate phrases and calibrates the proposed correction rules. Pass `--configuration uavid.json` to `predict` to use this profile. The background floor defaults to the chosen saved profile's value and can be specified with `--background-floor`. Native observations are cached in a temporary directory during configuration. Use `--cache-dir` to put this cache on a drive with free space.

Use the saved profiles for the exact reported phrase lists and rules. Fitting a new profile recomputes the response statistics and can produce a different phrase set and rules.
