#!/usr/bin/env python3
"""Append exact TrAISformer export steps to 03_embedding_pipeline.ipynb."""

from __future__ import annotations

from pathlib import Path
import json


MARKER = "## 5) Exact TrAISformer Export (for main.py replication)"


def main() -> None:
    nb_path = Path("CEE-Replication/Notebook/03_embedding_pipeline.ipynb")
    nb = json.loads(nb_path.read_text())

    existing = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )
    if MARKER in existing:
        print("Marker already present. Skipping update.")
        return

    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"{MARKER}\n",
            "\n",
            "To replicate `CEE_TrAISformer/trAISformer.py` exactly, export Region 1 into\n",
            "TrAISformer-compatible pickle files (`region_1_train.pkl`, `region_1_valid.pkl`, `region_1_test.pkl`).\n",
            "\n",
            "Note: `region_1_dense_embeddings.h5` is optional for diagnostics and not part of\n",
            "the original TrAISformer training path.\n",
        ],
    }

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import subprocess, sys\n",
            "\n",
            "root = Path('/home/crimsondeepdarshak/Desktop/Deep_Darshak/References/TRAIS_former_paper_Work_')\n",
            "script = root / 'CEE-Replication' / 'Notebook' / 'build_region1_traisformer_dataset.py'\n",
            "outdir = root / 'CEE-Replication' / 'traisformer_data' / 'region_1'\n",
            "cmd = [\n",
            "    sys.executable, str(script),\n",
            "    '--input', str(root / 'CEE-Replication' / 'Notebook' / 'region_1_interpolated.parquet'),\n",
            "    '--outdir', str(outdir),\n",
            "    '--max-seqlen-plus-one', '121',\n",
            "    '--min-seqlen', '37',\n",
            "    '--stride', '60',\n",
            "    '--max-sog', '30.0',\n",
            "    '--seed', '42',\n",
            "]\n",
            "print('Running:', ' '.join(cmd))\n",
            "subprocess.run(cmd, check=True)\n",
            "print('Done. Files in', outdir)\n",
            "for p in sorted(outdir.glob('*')):\n",
            "    print('-', p.name)\n",
        ],
    }

    nb["cells"].append(md_cell)
    nb["cells"].append(code_cell)
    nb_path.write_text(json.dumps(nb, indent=1))
    print(f"Updated notebook: {nb_path}")


if __name__ == "__main__":
    main()
