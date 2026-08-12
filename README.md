# Deepfake Detection — DFNet

Detects AI-generated / manipulated **face images** using a two-stream CNN trained
**from scratch** (no pretrained weights), with a Grad-CAM explanation and a web UI.

```
Upload image → detect & crop face → DFNet → P(fake) + Grad-CAM heatmap
```

---

## 1. What makes this more than a generic image classifier

A plain CNN on RGB learns whatever separates the two folders it was given, which
on most deepfake datasets means it memorises the generator's colour statistics
and collapses on anything else. DFNet is built against that failure mode:

| Component | What it does | Why it matters |
|---|---|---|
| **RGB stream** | Standard conv stem on the face crop | Catches semantic artefacts: asymmetric eyes, warped teeth, blending seams |
| **Noise stream (SRM)** | 3 fixed high-pass filters → 9 residual channels | Catches generator upsampling fingerprints and the *absence* of camera sensor noise — works even when the face looks perfect |
| **Early fusion** | 1×1 conv merges both stems | Both signals available at every depth, at ~1.1× the cost of one backbone |
| **SE attention** | Per-channel reweighting in each residual block | Lets the net lean on residual channels when RGB is uninformative |
| **Degradation augmentation** | Random JPEG / downscale / blur during training | Without it the detector dies the moment an image passes through WhatsApp |

The SRM kernels are *fixed DSP filters*, not learned or pretrained weights — see
[`src/models/layers.py`](src/models/layers.py). Everything trainable is randomly
initialised and learned by you.

Default model: **~8.0M parameters**.

---

## 2. Setup

Requires Python 3.11 or 3.12 (**not 3.14** — PyTorch has no wheels for it yet).

```bash
cd /Users/preetsagar/Deepfake_Detection
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

On your M1 this should print `True` — training uses the GPU via Metal.

---

## 3. Prove the pipeline works (2 minutes, no dataset needed)

```bash
python scripts/make_dummy_data.py --dst data/dummy --per-class 120

cat > /tmp/dummy.yaml <<'EOF'
data:  {root: data/dummy, img_size: 96, batch_size: 16, num_workers: 0}
model: {stem_channels: 16, stage_channels: [32, 64], blocks_per_stage: [1, 1]}
train: {epochs: 3, lr: 0.001, warmup_epochs: 0, out_dir: runs/smoke}
EOF

python -m src.train --config /tmp/dummy.yaml
python -m src.evaluate --checkpoint runs/smoke/best.pt --split test
python -m src.predict --checkpoint runs/smoke/best.pt --image data/dummy/test/fake/00000.png
```

The dummy data is synthetic noise, so the numbers are meaningless — this only
proves the plumbing runs. Delete `data/dummy` and `runs/smoke` afterwards.

---

## 4. Train on Kaggle GPU — the fast path

An epoch takes **25–40 min on the M1** and **~6 min on a Kaggle T4**. Kaggle also
*hosts the dataset*, so there is no 4 GB download and no preparation step at all.
Free tier: 30 GPU-hours/week, 9-hour session limit.

1. Push this project to GitHub after any change:
   ```bash
   git add -A && git commit -m "your message" && git push
   ```
   The Kaggle notebook clones <https://github.com/Preet1002/Deepfake_Detection>,
   so **the code must be pushed before you re-run the notebook** — otherwise
   Kaggle trains an older version. The repo must stay **public** for the clone
   to work without credentials.
2. Upload [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) to
   <https://kaggle.com/code>. The repo URL is already filled in.
3. In the right-hand panel set **Accelerator = GPU T4 x2**, **Internet = On**,
   and add the dataset **`xhlulu/140k-real-and-fake-faces`**.
4. `Run All`. The notebook does a 4-minute shakedown run first, then the full
   ~2–3 hour training, then evaluation, curves and Grad-CAM examples.
5. Download `best.pt` from the notebook output into `runs/dfnet/` on your Mac.
   The checkpoint carries its own config, so `evaluate.py` and the web app load
   it with no changes.

The loader reads Kaggle's read-only mount directly — it accepts the dataset's
`valid` folder as the `val` split, so nothing is copied or renamed.

> **Untested path:** all local development was verified on Apple MPS. The
> CUDA + AMP branch is written but has never executed here, which is exactly
> why the notebook runs a 2-epoch shakedown before the long run.

**Fast local iteration.** `data.limit_per_class` caps train/val without touching
the test split or re-preparing anything:

```yaml
data:
  limit_per_class: 4000     # ~4 min/epoch on the M1 instead of ~30
```

## 5. Get the dataset locally (if you'd rather train on your Mac)

**"140k Real and Fake Faces"** (Kaggle). 70k real Flickr faces +
70k StyleGAN faces, already cropped to 256×256, ~4 GB. It is the right size for
an 8 GB M1 and needs no licence agreement.

```bash
pip install kaggle
# Put your kaggle.json in ~/.config/kaggle/ (Kaggle → Account → Create New API Token)
kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw --unzip
python -m src.data.prepare \
  --src data/raw/real_vs_fake/real-vs-fake \
  --dst data/processed
```

`prepare.py` hard-links rather than copies, so this does **not** double your disk
usage. Use `--symlink` when the source sits on another filesystem or a read-only
mount (hard links cannot cross devices), or `--copy` if the source may move. It
also recognises other layouts — a flat `real/` + `fake/` pair works with
`--split`, and `--face-crop` runs detection first if your images are full scenes
rather than crops.

**Start small.** Add `--limit-per-class 8000` for your first real run: an epoch
takes minutes instead of hours, and you will find your bugs before burning a
whole afternoon.

Other options worth naming in your report: **FaceForensics++** and **Celeb-DF**
(both need a signed request form, both are video — you would extract frames),
and **DFDC**. Using a GAN dataset for training and a *different* generator for
testing is the cross-dataset experiment that separates a good project from an
average one.

Expected layout after preparation:

```
data/processed/
  train/real/*.jpg   train/fake/*.jpg
  val/real/*.jpg     val/fake/*.jpg
  test/real/*.jpg    test/fake/*.jpg
```

---

## 6. Train (locally)

```bash
python -m src.train --config configs/default.yaml
```

Checkpoints and `history.json` land in `runs/dfnet/`. The best checkpoint is
selected on **validation AUC** (threshold-free, far less noisy than accuracy).
Training stops early after 7 epochs without improvement.

Rough timing on an 8 GB M1 Air at `img_size: 160`, 100k training images:
**~25–40 min per epoch**. Practical plan:

- First run: `--limit-per-class 8000` during prepare, ~4 min/epoch, 15 epochs.
- Final run: full dataset, 20–30 epochs, leave it overnight.

If you hit memory pressure, drop `batch_size` to 32 and `img_size` to 128.

## 7. Evaluate

```bash
python -m src.evaluate --checkpoint runs/dfnet/best.pt --split test
```

Prints accuracy / AUC / AP / EER / precision / recall / F1 and writes
`roc.png`, `confusion_matrix.png`, `metrics.json` and raw `predictions.npz`
into `runs/dfnet/eval_test/` — drop these straight into your report.

**Cross-dataset test** (the number your examiner will ask about):

```bash
python -m src.evaluate --checkpoint runs/dfnet/best.pt \
  --data-root data/other_dataset --split test
```

Expect a large drop versus same-dataset accuracy. That drop *is* the finding —
report it honestly rather than hiding it; generalisation across generators is
the open problem in this field.

## 8. Ablation for the report

```bash
python -m src.train --config configs/no_srm.yaml     # RGB stream only
python -m src.evaluate --checkpoint runs/dfnet_no_srm/best.pt --split test
```

Compare against the full model. Fill in a table like:

| Model | Accuracy | AUC | EER | Cross-dataset AUC |
|---|---|---|---|---|
| DFNet (RGB only) | | | | |
| DFNet (RGB + SRM) | | | | |

The interesting result is usually that SRM barely helps *within* a dataset but
helps noticeably *across* datasets — a genuinely reportable finding.

## 9. Run the web app

```bash
DFNET_CHECKPOINT=runs/dfnet/best.pt uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Drag in a face image and you get a REAL/FAKE verdict,
P(fake), an adjustable decision threshold, and a Grad-CAM overlay showing which
regions drove the decision.

API:

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Model status, device, validation metrics |
| `POST /api/predict?explain=true&threshold=0.5` | Multipart image upload → JSON verdict + base64 heatmap |

## 10. Project layout

```
src/
  config.py          Typed config, YAML-loadable
  train.py           Training loop: AdamW, cosine LR + warmup, early stopping
  evaluate.py        Metrics + ROC/confusion-matrix figures
  predict.py         Detector class (used by CLI and web app)
  explain.py         Grad-CAM
  models/
    layers.py        SRM filter bank, SE block, residual block
    dfnet.py         The network
  data/
    dataset.py       Dataset, split-name aliasing, balanced sampling
    transforms.py    Augmentation incl. JPEG/downscale/blur degradations
    face.py          Haar-cascade face detection and cropping
    prepare.py       Dataset organisation and splitting
  utils/
    common.py        Seeding, device, checkpoints
    metrics.py       Accuracy, AUC, EER, confusion matrix
app/
  main.py            FastAPI backend
  static/index.html  Web UI
notebooks/
  kaggle_train.ipynb Run the whole pipeline on a free Kaggle GPU
configs/
  default.yaml       Main run (local, M1/MPS)
  kaggle.yaml        Kaggle T4/P100: CUDA, AMP, 224px, batch 128
  quick.yaml         Fast sanity run
  no_srm.yaml        Ablation
```

## 11. Honest limitations — put these in your report

- **Images only.** No video temporal analysis, no audio.
- **Cropped faces only.** Full scenes fall back to a centre crop and the result
  is unreliable; the UI says so when this happens.
- **Haar cascade misses profiles**, heavy occlusion and dark skin tones at low
  contrast. It is a 2001 algorithm chosen because it ships with OpenCV and needs
  no pretrained deep model. Swapping in MTCNN or YuNet is a clean improvement,
  and worth naming as future work.
- **Generalisation is the weak point.** A detector trained on StyleGAN faces will
  underperform on diffusion-generated faces. Measure it, do not assume it.
- **Not a forensic tool.** Never present the output as proof about a real person.

## 12. Suggested next steps if you have time

1. **Frequency stream** — add a DCT/FFT branch; periodic upsampling peaks are a
   strong and well-cited deepfake cue.
2. **Test-time augmentation** — average predictions over flips and JPEG qualities.
3. **Train on a second generator** (diffusion faces) and report the confusion.
4. **Calibration** — temperature-scale the output so P(fake) means something.
