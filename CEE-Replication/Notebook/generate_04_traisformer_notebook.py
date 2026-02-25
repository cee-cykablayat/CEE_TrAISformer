#!/usr/bin/env python3
"""Generate a replication notebook that uses the real CEE_TrAISformer code path."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# 04 TrAISformer Exact Replication (Region 1)\n"
            "\n"
            "This notebook replicates the **actual** `CEE_TrAISformer/trAISformer.py` pipeline:\n"
            "- Build TrAISformer `*.pkl` trajectory files from `region_1_interpolated.parquet`\n"
            "- Train using `CEE_TrAISformer/models.py`, `datasets.py`, `trainers.py`\n"
            "- Inspect logits, CE loss components, and sampling behavior\n"
            "\n"
            "Important: `region_1_dense_embeddings.h5` is kept only for diagnostics and visualization. "
            "It is **not** used by the original TrAISformer training path."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os, sys, json, pickle, subprocess\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import torch\n"
            "from torch.utils.data import DataLoader\n"
            "\n"
            "ROOT = Path('/home/crimsondeepdarshak/Desktop/Deep_Darshak/References/TRAIS_former_paper_Work_')\n"
            "NOTEBOOK_DIR = ROOT / 'CEE-Replication' / 'Notebook'\n"
            "TRAISFORMER_DIR = ROOT / 'CEE_TrAISformer'\n"
            "REGION_DATA_DIR = ROOT / 'CEE-Replication' / 'traisformer_data' / 'region_1'\n"
            "\n"
            "assert TRAISFORMER_DIR.exists(), TRAISFORMER_DIR\n"
            "sys.path.insert(0, str(TRAISFORMER_DIR))\n"
            "print('Using repo:', ROOT)\n"
            "print('Using TrAISformer src:', TRAISFORMER_DIR)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1) Build TrAISformer-Compatible Dataset (`*.pkl`)\n"
            "\n"
            "This step converts Region 1 parquet rows into trajectory windows with the same shape expected by "
            "`AISDataset`:\n"
            "- each sample: `{'mmsi': int, 'traj': np.ndarray[N,6]}`\n"
            "- traj columns: `[lat_norm, lon_norm, sog_norm, cog_norm, unix_time, mmsi]`"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "builder = NOTEBOOK_DIR / 'build_region1_traisformer_dataset.py'\n"
            "assert builder.exists(), builder\n"
            "\n"
            "train_pkl = REGION_DATA_DIR / 'region_1_train.pkl'\n"
            "valid_pkl = REGION_DATA_DIR / 'region_1_valid.pkl'\n"
            "test_pkl = REGION_DATA_DIR / 'region_1_test.pkl'\n"
            "\n"
            "if not (train_pkl.exists() and valid_pkl.exists() and test_pkl.exists()):\n"
            "    cmd = [\n"
            "        sys.executable,\n"
            "        str(builder),\n"
            "        '--input', str(NOTEBOOK_DIR / 'region_1_interpolated.parquet'),\n"
            "        '--outdir', str(REGION_DATA_DIR),\n"
            "        '--max-seqlen-plus-one', '121',\n"
            "        '--min-seqlen', '37',\n"
            "        '--stride', '60',\n"
            "        '--max-sog', '30.0',\n"
            "        '--seed', '42',\n"
            "    ]\n"
            "    print('Running:', ' '.join(cmd))\n"
            "    subprocess.run(cmd, check=True)\n"
            "else:\n"
            "    print('Pickles already exist. Skipping rebuild.')\n"
            "\n"
            "meta = json.loads((REGION_DATA_DIR / 'metadata.json').read_text())\n"
            "print(json.dumps(meta['splits'], indent=2))\n"
            "print('Normalization:', json.dumps(meta['normalization'], indent=2))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "with open(train_pkl, 'rb') as f:\n"
            "    train_raw = pickle.load(f)\n"
            "with open(valid_pkl, 'rb') as f:\n"
            "    valid_raw = pickle.load(f)\n"
            "with open(test_pkl, 'rb') as f:\n"
            "    test_raw = pickle.load(f)\n"
            "\n"
            "print('train/valid/test windows:', len(train_raw), len(valid_raw), len(test_raw))\n"
            "ex = train_raw[0]\n"
            "print('sample keys:', ex.keys())\n"
            "print('sample traj shape:', ex['traj'].shape)\n"
            "print('first row [lat,lon,sog,cog,t,mmsi]:', ex['traj'][0])"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2) Optional H5 Diagnostics\n"
            "\n"
            "The generated H5 stores dense vectors (`[N,256]`) from a separate embedding notebook.\n"
            "We inspect it here, but **do not feed it into the exact TrAISformer model**."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import h5py\n"
            "h5_path = NOTEBOOK_DIR / 'region_1_dense_embeddings.h5'\n"
            "if h5_path.exists():\n"
            "    with h5py.File(h5_path, 'r') as f:\n"
            "        keys = list(f.keys())\n"
            "        print('H5 keys:', keys)\n"
            "        d = f[keys[0]]\n"
            "        print('H5 shape:', d.shape, 'dtype:', d.dtype)\n"
            "        print('first vector first 8 dims:', d[0, :8])\n"
            "else:\n"
            "    print('H5 file not found:', h5_path)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3) Load Original TrAISformer Modules and Config"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import datasets, models, trainers\n"
            "\n"
            "class RegionConfig:\n"
            "    # runtime\n"
            "    retrain = True\n"
            "    tb_log = False\n"
            "    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n"
            "\n"
            "    # optimization (same style as original config)\n"
            "    max_epochs = 10\n"
            "    batch_size = 32\n"
            "    n_samples = 8\n"
            "    learning_rate = 6e-4\n"
            "    betas = (0.9, 0.95)\n"
            "    grad_norm_clip = 1.0\n"
            "    weight_decay = 0.1\n"
            "    lr_decay = True\n"
            "    warmup_tokens = 512 * 20\n"
            "    final_tokens = 260e9\n"
            "    num_workers = 0\n"
            "\n"
            "    # sequence settings\n"
            "    init_seqlen = 18\n"
            "    max_seqlen = 120\n"
            "    min_seqlen = 36\n"
            "\n"
            "    # model/sampling flags\n"
            "    mode = 'pos'\n"
            "    sample_mode = 'pos_vicinity'\n"
            "    top_k = 10\n"
            "    r_vicinity = 40\n"
            "\n"
            "    # blur regularization (this is the modified loss in code)\n"
            "    blur = True\n"
            "    blur_learnable = False\n"
            "    blur_loss_w = 1.0\n"
            "    blur_n = 2\n"
            "\n"
            "    # region normalization bounds from metadata\n"
            "    lat_min = meta['normalization']['lat_min']\n"
            "    lat_max = meta['normalization']['lat_max']\n"
            "    lon_min = meta['normalization']['lon_min']\n"
            "    lon_max = meta['normalization']['lon_max']\n"
            "\n"
            "    # classification sizes (keep original TrAISformer defaults)\n"
            "    lat_size = 250\n"
            "    lon_size = 270\n"
            "    sog_size = 30\n"
            "    cog_size = 72\n"
            "\n"
            "    # embedding dims / transformer depth\n"
            "    n_lat_embd = 256\n"
            "    n_lon_embd = 256\n"
            "    n_sog_embd = 128\n"
            "    n_cog_embd = 128\n"
            "    n_head = 8\n"
            "    n_layer = 8\n"
            "    embd_pdrop = 0.1\n"
            "    resid_pdrop = 0.1\n"
            "    attn_pdrop = 0.1\n"
            "\n"
            "    dataset_name = 'region_1'\n"
            "    full_size = lat_size + lon_size + sog_size + cog_size\n"
            "    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd\n"
            "\n"
            "    savedir = str(ROOT / 'CEE-Replication' / 'results' / 'region_1_trAISformer') + '/'\n"
            "    ckpt_path = os.path.join(savedir, 'model.pt')\n"
            "\n"
            "cf = RegionConfig()\n"
            "os.makedirs(cf.savedir, exist_ok=True)\n"
            "print('Device:', cf.device)\n"
            "print('Checkpoint:', cf.ckpt_path)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## 4) Build AISDataset and Visualize Input")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "train_ds = datasets.AISDataset(train_raw, max_seqlen=cf.max_seqlen + 1, device=cf.device)\n"
            "valid_ds = datasets.AISDataset(valid_raw, max_seqlen=cf.max_seqlen + 1, device=cf.device)\n"
            "test_ds = datasets.AISDataset(test_raw, max_seqlen=cf.max_seqlen + 1, device=cf.device)\n"
            "print('Dataset lens:', len(train_ds), len(valid_ds), len(test_ds))\n"
            "\n"
            "seq, mask, seqlen, mmsi, t0 = train_ds[0]\n"
            "print('seq shape:', seq.shape, 'mask shape:', mask.shape, 'seqlen:', int(seqlen))\n"
            "print('mmsi:', int(mmsi), 'time_start_unix:', int(t0))\n"
            "print('first 3 normalized rows:\\n', seq[:3])"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(10, 3))\n"
            "plt.subplot(1, 2, 1)\n"
            "plt.plot(seq[:int(seqlen), 0].numpy(), label='lat_norm')\n"
            "plt.plot(seq[:int(seqlen), 1].numpy(), label='lon_norm')\n"
            "plt.legend()\n"
            "plt.title('Normalized Position Sequence')\n"
            "\n"
            "plt.subplot(1, 2, 2)\n"
            "plt.plot(seq[:int(seqlen), 2].numpy(), label='sog_norm')\n"
            "plt.plot(seq[:int(seqlen), 3].numpy(), label='cog_norm')\n"
            "plt.legend()\n"
            "plt.title('Normalized Kinematics Sequence')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## 5) Instantiate Real TrAISformer and Inspect Forward + Loss")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "model = models.TrAISformer(cf, partition_model=None).to(cf.device)\n"
            "print('n_params:', sum(p.numel() for p in model.parameters()))\n"
            "\n"
            "loader = DataLoader(train_ds, batch_size=cf.batch_size, shuffle=True)\n"
            "seqs, masks, seqlens, mmsis, tstarts = next(iter(loader))\n"
            "seqs = seqs.to(cf.device)\n"
            "masks = masks[:, :-1].to(cf.device)\n"
            "\n"
            "logits, loss, loss_tuple = model(seqs, masks=masks, with_targets=True, return_loss_tuple=True)\n"
            "print('logits shape:', tuple(logits.shape))\n"
            "print('loss:', float(loss))\n"
            "print('loss tuple shapes:', [tuple(x.shape) for x in loss_tuple])\n"
            "print('mean per-head CE:')\n"
            "print('  lat:', float(loss_tuple[0].mean()))\n"
            "print('  lon:', float(loss_tuple[1].mean()))\n"
            "print('  sog:', float(loss_tuple[2].mean()))\n"
            "print('  cog:', float(loss_tuple[3].mean()))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "lat_logits, lon_logits, sog_logits, cog_logits = torch.split(\n"
            "    logits[:, :, :],\n"
            "    (cf.lat_size, cf.lon_size, cf.sog_size, cf.cog_size),\n"
            "    dim=-1\n"
            ")\n"
            "last_lat_probs = torch.softmax(lat_logits[0, -1], dim=-1).detach().cpu().numpy()\n"
            "plt.figure(figsize=(8, 3))\n"
            "plt.plot(last_lat_probs)\n"
            "plt.title('Latitude head probabilities at final timestep (sample 0)')\n"
            "plt.xlabel('lat bin index')\n"
            "plt.ylabel('probability')\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 6) Train with Original Trainer (same path as `trAISformer.py`)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "trainer = trainers.Trainer(\n"
            "    model=model,\n"
            "    train_dataset=train_ds,\n"
            "    test_dataset=valid_ds,\n"
            "    config=cf,\n"
            "    savedir=cf.savedir,\n"
            "    device=cf.device,\n"
            "    aisdls={},\n"
            "    INIT_SEQLEN=cf.init_seqlen,\n"
            ")\n"
            "\n"
            "RUN_TRAIN = False  # change to True to run full training\n"
            "if RUN_TRAIN:\n"
            "    trainer.train()\n"
            "    print('Training completed.')\n"
            "else:\n"
            "    print('Training skipped. Set RUN_TRAIN=True to train.')"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 7) Sampling Demo (Autoregressive Rollout)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "model.eval()\n"
            "demo_batch = next(iter(DataLoader(test_ds, batch_size=4, shuffle=True)))\n"
            "demo_seqs, demo_masks, demo_seqlens, demo_mmsis, demo_ts = demo_batch\n"
            "demo_init = demo_seqs[:, :cf.init_seqlen, :].to(cf.device)\n"
            "\n"
            "with torch.no_grad():\n"
            "    preds = trainers.sample(\n"
            "        model,\n"
            "        demo_init,\n"
            "        steps=12,\n"
            "        temperature=1.0,\n"
            "        sample=True,\n"
            "        sample_mode=cf.sample_mode,\n"
            "        r_vicinity=cf.r_vicinity,\n"
            "        top_k=cf.top_k,\n"
            "    )\n"
            "\n"
            "print('init shape:', tuple(demo_init.shape), 'pred shape:', tuple(preds.shape))\n"
            "print('first trajectory last predicted point [lat,lon,sog,cog]:', preds[0, -1].detach().cpu().numpy())"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "p = preds[0].detach().cpu().numpy()\n"
            "plt.figure(figsize=(5, 5))\n"
            "plt.plot(p[:, 1], p[:, 0], marker='o', ms=2)\n"
            "plt.title('Sampled normalized trajectory (lon vs lat)')\n"
            "plt.xlabel('lon_norm')\n"
            "plt.ylabel('lat_norm')\n"
            "plt.grid(True)\n"
            "plt.show()"
        )
    )

    nb["cells"] = cells

    out_path = Path("CEE-Replication/Notebook/04_TRAISformer_01.ipynb")
    out_path.write_text(nbf.writes(nb))
    print(f"Wrote notebook: {out_path}")


if __name__ == "__main__":
    main()
