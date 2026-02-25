#!/usr/bin/env python3
"""Generate 04 notebook with modular preprocessing + MLflow + resume training."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# 04 TrAISformer MLOps Replication (Region 1)\n"
            "\n"
            "This notebook provides:\n"
            "- Class-wise modular AIS preprocessing (paper-style filters)\n"
            "- TrAISformer dataset export (`*.pkl`) compatible with `CEE_TrAISformer`\n"
            "- MLflow tracking (loss, accuracy, params, artifacts, logs)\n"
            "- Crash-safe resume (`last_checkpoint.pt`) and best model save\n"
            "- Early stopping with max 50 epochs\n"
            "\n"
            "Environment note:\n"
            "Run from your target env before opening Jupyter:\n"
            "`source ~/Desktop/Deepdarshk/code_source/env.bin/activate`"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os, sys, json, pickle\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import torch\n"
            "from torch.utils.data import DataLoader\n"
            "\n"
            "ROOT = Path('/home/crimsondeepdarshak/Desktop/Deep_Darshak/References/TRAIS_former_paper_Work_')\n"
            "NOTEBOOK_DIR = ROOT / 'CEE-Replication' / 'Notebook'\n"
            "TRAISFORMER_DIR = ROOT / 'CEE_TrAISformer'\n"
            "REGION_DATA_DIR = ROOT / 'CEE-Replication' / 'traisformer_data' / 'region_1'\n"
            "RUN_DIR = ROOT / 'CEE-Replication' / 'results' / 'region_1_trAISformer_mlops'\n"
            "\n"
            "assert TRAISFORMER_DIR.exists(), TRAISFORMER_DIR\n"
            "sys.path.insert(0, str(TRAISFORMER_DIR))\n"
            "sys.path.insert(0, str(NOTEBOOK_DIR))\n"
            "print('Using root:', ROOT)\n"
            "print('Using src:', TRAISFORMER_DIR)\n"
            "print('Run dir:', RUN_DIR)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import datasets, models, trainers\n"
            "from traisformer_mlops import (\n"
            "    AISPreprocessor,\n"
            "    PreprocessingConfig,\n"
            "    ExportConfig,\n"
            "    TrAISformerDataExporter,\n"
            "    EarlyStoppingConfig,\n"
            "    TrAISformerMLflowTrainer,\n"
            ")\n"
            "\n"
            "try:\n"
            "    import mlflow\n"
            "    print('mlflow version:', mlflow.__version__)\n"
            "except Exception as e:\n"
            "    print('mlflow not available. Install with: pip install mlflow')\n"
            "    raise"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1) Preprocessing + Export (paper criteria)\n"
            "\n"
            "Implemented filters:\n"
            "- remove near coastline (`<1 nm`) if coastline polygons + shapely available\n"
            "- split non-contiguous voyages (`max gap = 2h`)\n"
            "- remove short voyages (`<20 msgs` or duration `<4h`)\n"
            "- remove abnormal empirical speed (`>40 knots`)\n"
            "- remove unrealistic SOG (`>=30 knots`) and moored/anchor-like rows\n"
            "- downsample to 10 minutes\n"
            "- split long voyages to max 20 hours"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "RUN_PREPROCESS = False  # heavy; set True for full preprocessing from parquet\n"
            "parquet_path = NOTEBOOK_DIR / 'region_1_interpolated.parquet'\n"
            "coastline_pickle = TRAISFORMER_DIR / 'data' / 'ct_dma' / 'dma_coastline_polygons.pkl'\n"
            "\n"
            "if RUN_PREPROCESS:\n"
            "    pre_cfg = PreprocessingConfig(\n"
            "        max_gap_hours=2.0,\n"
            "        min_voyage_messages=20,\n"
            "        min_voyage_hours=4.0,\n"
            "        max_empirical_speed_knots=40.0,\n"
            "        max_sog_knots=30.0,\n"
            "        downsample_minutes=10,\n"
            "        max_seq_hours=20.0,\n"
            "        keep_only_underway=True,\n"
            "        coastline_min_nm=1.0,\n"
            "    )\n"
            "    pre = AISPreprocessor(pre_cfg)\n"
            "    df_clean = pre.run(parquet_path=parquet_path, coastline_pickle=coastline_pickle)\n"
            "\n"
            "    exporter = TrAISformerDataExporter(ExportConfig(max_sog=30.0, train_ratio=0.8, valid_ratio=0.1, seed=42))\n"
            "    meta = exporter.export(df_clean, REGION_DATA_DIR)\n"
            "    print('Exported metadata:', json.dumps(meta, indent=2))\n"
            "else:\n"
            "    print('Skipping heavy preprocessing. Using existing exported pickles.')\n"
            "    meta = json.loads((REGION_DATA_DIR / 'metadata.json').read_text())\n"
            "    print(json.dumps(meta, indent=2))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "train_pkl = REGION_DATA_DIR / 'region_1_train.pkl'\n"
            "valid_pkl = REGION_DATA_DIR / 'region_1_valid.pkl'\n"
            "test_pkl = REGION_DATA_DIR / 'region_1_test.pkl'\n"
            "for p in [train_pkl, valid_pkl, test_pkl]:\n"
            "    assert p.exists(), f'Missing file: {p}'\n"
            "\n"
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
            "print('sample first row [lat,lon,sog,cog,time,mmsi]:', ex['traj'][0])"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## 2) Config and Datasets")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "class RegionConfig:\n"
            "    retrain = True\n"
            "    tb_log = False\n"
            "    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n"
            "\n"
            "    # training\n"
            "    max_epochs = 50\n"
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
            "    # sequence\n"
            "    init_seqlen = 18\n"
            "    max_seqlen = 120\n"
            "    min_seqlen = 36\n"
            "\n"
            "    # model behavior\n"
            "    mode = 'pos'\n"
            "    sample_mode = 'pos_vicinity'\n"
            "    top_k = 10\n"
            "    r_vicinity = 40\n"
            "    blur = True\n"
            "    blur_learnable = False\n"
            "    blur_loss_w = 1.0\n"
            "    blur_n = 2\n"
            "\n"
            "    lat_min = meta['normalization']['lat_min']\n"
            "    lat_max = meta['normalization']['lat_max']\n"
            "    lon_min = meta['normalization']['lon_min']\n"
            "    lon_max = meta['normalization']['lon_max']\n"
            "\n"
            "    lat_size = 250\n"
            "    lon_size = 270\n"
            "    sog_size = 30\n"
            "    cog_size = 72\n"
            "    n_lat_embd = 256\n"
            "    n_lon_embd = 256\n"
            "    n_sog_embd = 128\n"
            "    n_cog_embd = 128\n"
            "    n_head = 8\n"
            "    n_layer = 8\n"
            "    embd_pdrop = 0.1\n"
            "    resid_pdrop = 0.1\n"
            "    attn_pdrop = 0.1\n"
            "    full_size = lat_size + lon_size + sog_size + cog_size\n"
            "    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd\n"
            "    savedir = str(RUN_DIR) + '/'\n"
            "    ckpt_path = str(RUN_DIR / 'model.pt')\n"
            "\n"
            "cf = RegionConfig()\n"
            "RUN_DIR.mkdir(parents=True, exist_ok=True)\n"
            "print('Device:', cf.device)\n"
            "print('Max epochs:', cf.max_epochs)\n"
            "print('Checkpoint path:', cf.ckpt_path)"
        )
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
            "print('mmsi:', int(mmsi), 'time_start_unix:', int(t0))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "plt.figure(figsize=(10, 3))\n"
            "plt.subplot(1, 2, 1)\n"
            "plt.plot(seq[:int(seqlen), 0].numpy(), label='lat_norm')\n"
            "plt.plot(seq[:int(seqlen), 1].numpy(), label='lon_norm')\n"
            "plt.legend()\n"
            "plt.title('Normalized position')\n"
            "\n"
            "plt.subplot(1, 2, 2)\n"
            "plt.plot(seq[:int(seqlen), 2].numpy(), label='sog_norm')\n"
            "plt.plot(seq[:int(seqlen), 3].numpy(), label='cog_norm')\n"
            "plt.legend()\n"
            "plt.title('Normalized kinematics')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## 3) Model Forward Sanity Check")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "model = models.TrAISformer(cf, partition_model=None).to(cf.device)\n"
            "print('n_params:', sum(p.numel() for p in model.parameters()))\n"
            "loader = DataLoader(train_ds, batch_size=cf.batch_size, shuffle=True)\n"
            "seqs, masks, *_ = next(iter(loader))\n"
            "seqs = seqs.to(cf.device)\n"
            "masks = masks[:, :-1].to(cf.device)\n"
            "logits, loss, loss_tuple = model(seqs, masks=masks, with_targets=True, return_loss_tuple=True)\n"
            "print('logits shape:', tuple(logits.shape))\n"
            "print('loss:', float(loss))\n"
            "print('head losses:', [float(x.mean()) for x in loss_tuple])"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## 4) MLflow + Early Stopping + Resume Training")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# Launch MLflow UI from terminal if needed:\n"
            "# mlflow ui --host 0.0.0.0 --port 5000\n"
            "\n"
            "early_cfg = EarlyStoppingConfig(max_epochs=50, patience=8, min_delta=1e-4)\n"
            "trainer = TrAISformerMLflowTrainer(\n"
            "    model=model,\n"
            "    train_dataset=train_ds,\n"
            "    valid_dataset=valid_ds,\n"
            "    test_dataset=test_ds,\n"
            "    config=cf,\n"
            "    sample_fn=trainers.sample,\n"
            "    run_dir=RUN_DIR,\n"
            "    early_stop=early_cfg,\n"
            "    experiment_name='trAISformer_region1',\n"
            ")\n"
            "\n"
            "RUN_TRAIN = True\n"
            "if RUN_TRAIN:\n"
            "    trainer.train()\n"
            "    print('Training complete. Best checkpoint:', RUN_DIR / 'best_model.pt')\n"
            "else:\n"
            "    print('Skipped training.')"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## 5) Post-Training Inference Sample")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "if (RUN_DIR / 'best_model.pt').exists():\n"
            "    payload = torch.load(RUN_DIR / 'best_model.pt', map_location=cf.device)\n"
            "    model.load_state_dict(payload['model_state'])\n"
            "model.eval()\n"
            "demo_batch = next(iter(DataLoader(test_ds, batch_size=4, shuffle=True)))\n"
            "demo_seqs, *_ = demo_batch\n"
            "demo_init = demo_seqs[:, :cf.init_seqlen, :].to(cf.device)\n"
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
            "print('pred shape:', tuple(preds.shape))\n"
            "print('first pred last token:', preds[0, -1].detach().cpu().numpy())"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "p = preds[0].detach().cpu().numpy()\n"
            "plt.figure(figsize=(5, 5))\n"
            "plt.plot(p[:, 1], p[:, 0], marker='o', ms=2)\n"
            "plt.title('Sampled normalized trajectory')\n"
            "plt.xlabel('lon_norm')\n"
            "plt.ylabel('lat_norm')\n"
            "plt.grid(True)\n"
            "plt.show()"
        )
    )

    nb["cells"] = cells

    out = Path("CEE-Replication/Notebook/04_TRAISformer_01.ipynb")
    out.write_text(nbf.writes(nb))
    print(f"Wrote notebook: {out}")


if __name__ == "__main__":
    main()
