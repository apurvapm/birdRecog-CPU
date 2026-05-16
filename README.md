# EfficientNet on CUB-200-2011

Fine-tune an ImageNet-pretrained EfficientNet to identify the 200 bird species
in the Caltech-UCSD Birds-200-2011 dataset. This is a transfer-learning
project — used in real-world computer-vision work.

## Files

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


## Step 1: download the dataset

```bash
python download_data.py
```

Expect 5–15 minutes depending
on your connection.


## Step 2: train

```bash
python train.py
```

With the defaults in `config.py` (CPU):

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


