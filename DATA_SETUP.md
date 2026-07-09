# cifar10_resnet — Claude-authored integration (eval-not-triggered repro)

This branch holds the decorator-style Tensorleap integration authored by the
`tensorleap-integration-creation` skill for the CIFAR-10 ResNet classifier. It was
captured to reproduce a platform issue: **`leapdev push --eval` builds to `FINISHED`
but no `Evaluate` job is ever created/queued.**

## Integration files (what gets bundled)
`leap.yaml` `include:` — these are the only files pushed:
- `leap_integration.py` — entry file (decorator style: preprocess, input encoder,
  GT encoder, `@tensorleap_load_model`, loss, metrics, visualizers, integration test).
- `cifar10_resnet/project_config.yaml` — config: `LABELS_NAMES` (10 CIFAR classes),
  `DATA_VOLUME_DIR`, `SUBSET_TRAIN: 500`, `SUBSET_VAL: 200`.
- `requirements.txt` — platform build deps.
- The model `model/resnet.h5` is **not** bundled; it is uploaded separately via `-m`.

## Data setup
CIFAR-10. **No manual dataset staging is required** — the integration loads the data
from the Tensorleap data volume and, on a cache-miss, downloads CIFAR-10 via
`keras.datasets.cifar10.load_data()` and caches `cifar10_x.npy` + `cifar10_y.npy`
into the data-volume directory.

- The directory is `DATA_VOLUME_DIR` in `cifar10_resnet/project_config.yaml`
  (defaults to a machine-local absolute path). Override it without editing the file
  by exporting `CIFAR10_DATA_DIR=/your/data-volume/cifar10_resnet`.
- On the platform: the data volume must be mounted; the integration creates the
  directory and caches the arrays there on the first evaluate.
- Only `SUBSET_TRAIN` (500) + `SUBSET_VAL` (200) samples are used, for a fast eval.

## Reproduce the push
```bash
# 1) create + select a project (fresh server has none)
leapdev projects create cifar10_resnet
leapdev projects select <new-project-id>   # writes projectId into leap.yaml

# 2) push with evaluation
leapdev push -m model/resnet.h5 -n resnet-v1 -b 8 --type H5_TF2 --eval --yes

# 3) watch jobs
leapdev run list
```

### Observed
- The **Push** job reaches `FINISHED` — all build stages pass (Parsing Dataset,
  Data Loader Prep, Parsing Model, Convert, Build Model, Run Model Inference,
  Testing Loss / Visualizers / Metrics).
- **No `Evaluate` job appears** in `leapdev run list` despite `--eval` — so the
  evaluate step is never created/queued. (Local `check_dataset()` returns
  `isValid=True` with the full interface set, so the integration itself validates.)

Environment: local dev server via `leapdev`, server `v1.6.39`.
