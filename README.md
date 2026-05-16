# EfficientNet on CUB-200-2011

Fine-tune an ImageNet-pretrained EfficientNet to identify the 200 bird species
in the Caltech-UCSD Birds-200-2011 dataset. This is a textbook transfer-learning
project — the same recipe used in real-world computer-vision work.

## What's in the box

| File | Purpose |
| --- | --- |
| `config.py` | All hyperparameters and paths in one place. Edit here or override on the CLI. |
| `download_data.py` | Downloads + extracts `CUB_200_2011.tgz` (~1.1 GB) into `./data/`. |
| `dataset.py` | PyTorch `Dataset` that reads the CUB metadata files; also returns ready-to-use train/val `DataLoader`s. |
| `train.py` | Builds an EfficientNet, swaps the head for 200 classes, runs the training loop, saves the best checkpoint. |
| `predict.py` | Loads the trained checkpoint and prints top-5 predictions for a single image. |

## Setup

```bash
cd cub_efficientnet
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

On Python 3.13 you may also need `pip install audioop-lts` (the same fix as the
`bird_id` project — though this project doesn't actually use Gradio so you can
skip it).

## Step 1: download the dataset

```bash
python download_data.py
```

This streams `CUB_200_2011.tgz` from `data.caltech.edu` (~1.1 GB compressed,
~1.6 GB extracted) into `./data/CUB_200_2011/`. Expect 5–15 minutes depending
on your connection. The script is resumable — if the .tgz is already there it
skips download; if the extracted folder is already there it skips extraction.

If the official URL is slow or blocked, two alternative sources are noted in
the script:

- Kaggle: `kaggle datasets download wenewone/cub2002011`
- Hugging Face: `datasets.load_dataset("Donghyun99/CUB-200-2011")`

## Step 2: train

```bash
python train.py
```

With the defaults in `config.py` (CPU-friendly):

- Model: `efficientnet_b0` (pretrained on ImageNet, ~5M params)
- Backbone **frozen** — we only train the new 200-class classifier head
  (~257K trainable params)
- 8 epochs, batch size 32, image size 224

Expected on CPU: **~10–20 minutes per epoch**, ~60–70% top-1 val accuracy
after 8 epochs. (Yes, the model can correctly name the species in 6 out of
10 random test photos after only training a single linear layer — that's the
power of transfer learning from a strong pretrained backbone.)

### Useful flags

```bash
python train.py --epochs 12              # train longer
python train.py --batch-size 16          # if you run out of RAM
python train.py --no-freeze --lr 1e-4    # full fine-tune (needs a GPU realistically)
python train.py --model efficientnet_b2  # bigger backbone
```

Checkpoints land in `./checkpoints/best.pt`. A CSV log of every epoch
(`training_log.csv`) lives there too — useful for plotting learning curves.

## Step 3: predict on a new photo

```bash
python predict.py path/to/some_bird.jpg
```

Prints something like:

```
some_bird.jpg
-------------
  Indigo Bunting                       72.41%  #####################
  Painted Bunting                      11.30%  ###
  Blue Grosbeak                         5.67%  ##
  Eastern Bluebird                      3.42%  #
  Lazuli Bunting                        1.98%
```

## How the training works (short version)

1. **Load a pretrained EfficientNet-B0.** Its 5M parameters were trained on
   ImageNet (1.2M photos, 1000 everyday categories). The lower layers
   already know what edges, textures, feathers, and beaks look like.
2. **Chop off the original 1000-class output layer.** Replace it with a
   fresh `nn.Linear(1280, 200)` — the 1280 numbers coming out of the last
   conv block now flow into 200 species scores instead.
3. **Freeze the backbone.** Setting `requires_grad=False` on every parameter
   except the new head means only ~257K weights need gradient updates.
   Forward passes still flow through the full network, but the backward pass
   skips most of it — massively cheaper and good enough when your dataset
   is similar to ImageNet (which CUB is).
4. **Train with AdamW + cosine learning-rate decay.** Cross-entropy loss on
   the softmaxed logits vs. the integer class label. Every epoch we evaluate
   on the held-out test split and save the best checkpoint.

## Tips for higher accuracy

- **Unfreeze the backbone** (`--no-freeze`) and use a small learning rate
  (`--lr 1e-4`) for 20+ epochs. Best on a GPU. Targets ~80%+ top-1.
- **Use a bigger backbone:** `--model efficientnet_b2` or `b3`. Diminishing
  returns past `b3` for CUB.
- **Stronger augmentation:** edit `dataset.py` → `train_transforms()` to add
  `RandAugment()` or `AutoAugment(policy=AutoAugmentPolicy.IMAGENET)`.
- **Mixup / cutmix:** small library `timm.data.Mixup` plugs in cleanly.

## Troubleshooting

**`No matching distribution for torch`.** Some platforms (especially Windows
+ Python 3.13) don't have a default torch wheel yet. Try
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

**DataLoader hangs on Windows.** Set `--num-workers 0` (or edit
`config.NUM_WORKERS`) — Windows has historical issues with PyTorch
multi-process DataLoaders.

**Out of memory.** Drop `--batch-size`. `batch-size 8` is fine; the optimizer
doesn't care.

**Download stalls.** The Caltech mirror is occasionally slow. Use Ctrl+C and
re-run; the script will resume from the partial file if it exists. Otherwise
switch to the Kaggle or Hugging Face mirror noted above.

## What you'll have learned by the end

- How to read a real-world dataset whose layout is metadata files + a folder
  of JPEGs (the same pattern shows up everywhere — ImageNet, COCO, medical
  datasets, satellite imagery).
- Why transfer learning works and when to freeze vs. fine-tune.
- The full PyTorch training loop: dataset → loader → model → optimizer →
  scheduler → checkpointing.
- A working classifier you can plug into the `bird_id` Gradio app from the
  sister project — just edit `classifier.py` to load `best.pt` instead of
  the Hugging Face model.
