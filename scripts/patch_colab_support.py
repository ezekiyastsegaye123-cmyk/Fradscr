#!/usr/bin/env python3
"""
Patch model-2.ipynb and model-1.ipynb with 1-click Google Colab Bootstrap
and self-healing out-of-order execution guards.
"""

import json
from pathlib import Path

COLAB_BOOTSTRAP_CELL_SOURCE = [
    "# ==============================================================================\n",
    "# 🚀 STEP 1: GOOGLE COLAB 1-CLICK BOOTSTRAP (RUN THIS CELL FIRST IF ON COLAB)\n",
    "# ==============================================================================\n",
    "import sys\n",
    "import os\n",
    "import subprocess\n",
    "from pathlib import Path\n",
    "\n",
    "# Detect Google Colab environment\n",
    "IN_COLAB = \"google.colab\" in sys.modules or os.path.exists(\"/content\")\n",
    "\n",
    "if IN_COLAB:\n",
    "    print(\"⚡ Google Colab environment detected. Initializing Fradscr...\")\n",
    "    repo_dir = Path(\"/content/Fradscr\")\n",
    "    \n",
    "    # Clone repository if missing or incomplete\n",
    "    if not (repo_dir / \"africa\" / \"eth002.rwl\").exists():\n",
    "        if repo_dir.exists():\n",
    "            import shutil\n",
    "            shutil.rmtree(repo_dir, ignore_errors=True)\n",
    "        print(\"📥 Cloning Fradscr repository from GitHub into /content/Fradscr...\")\n",
    "        subprocess.run([\"git\", \"clone\", \"https://github.com/ezekiyastsegaye123-cmyk/Fradscr.git\", str(repo_dir)], check=True)\n",
    "    \n",
    "    # Change working directory permanently in IPython kernel\n",
    "    try:\n",
    "        get_ipython().run_line_magic(\"cd\", str(repo_dir))\n",
    "    except Exception:\n",
    "        os.chdir(str(repo_dir))\n",
    "        \n",
    "    if str(repo_dir) not in sys.path:\n",
    "        sys.path.insert(0, str(repo_dir))\n",
    "        \n",
    "    # Ensure netCDF4 is available for climate data reading\n",
    "    try:\n",
    "        import netCDF4\n",
    "    except ImportError:\n",
    "        print(\"📦 Installing netCDF4 for SPEI climate dataset ingestion...\")\n",
    "        subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"netCDF4\"], check=True)\n",
    "        \n",
    "    PROJECT_ROOT = repo_dir\n",
    "    print(f\"✅ Google Colab initialized! Working directory: {os.getcwd()}\")\n",
    "else:\n",
    "    PROJECT_ROOT = Path(\".\").resolve()\n",
    "    if str(PROJECT_ROOT) not in sys.path:\n",
    "        sys.path.insert(0, str(PROJECT_ROOT))\n",
    "    print(f\"✅ Local environment active. Working directory: {PROJECT_ROOT}\")\n",
]

CELL_4_SELF_HEALING_PREFIX = [
    "# Self-healing environment guard (enables single-cell execution on Google Colab)\n",
    "if \"PROJECT_ROOT\" not in globals() or not (PROJECT_ROOT / \"africa\" / \"eth002.rwl\").exists():\n",
    "    import sys, os, subprocess\n",
    "    from pathlib import Path\n",
    "    for cand in [Path(\"/content/Fradscr\"), Path(\".\").resolve(), Path(\"..\").resolve()]:\n",
    "        if (cand / \"africa\" / \"eth002.rwl\").exists():\n",
    "            PROJECT_ROOT = cand\n",
    "            break\n",
    "    else:\n",
    "        if os.path.exists(\"/content\"):\n",
    "            subprocess.run([\"git\", \"clone\", \"https://github.com/ezekiyastsegaye123-cmyk/Fradscr.git\", \"/content/Fradscr\"], check=True)\n",
    "            PROJECT_ROOT = Path(\"/content/Fradscr\")\n",
    "        else:\n",
    "            PROJECT_ROOT = Path(\".\").resolve()\n",
    "    if str(PROJECT_ROOT) not in sys.path:\n",
    "        sys.path.insert(0, str(PROJECT_ROOT))\n",
    "\n",
    "if \"process_multiple_rwl\" not in globals():\n",
    "    from treering.pipeline import process_multiple_rwl\n",
    "if \"plt\" not in globals():\n",
    "    import matplotlib.pyplot as plt\n",
    "\n",
]

def patch_model_2():
    path = Path("model-2.ipynb")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # 1. Update Cell 0 markdown with clear instruction
    c0_text = "".join(cells[0]["source"])
    instruction = (
        "\n> ⚡ **Google Colab Quickstart**: If viewing this notebook in Google Colab, "
        "simply click and run the **Step 1: Google Colab 1-Click Bootstrap** code cell below! "
        "It will clone the repository, mount all datasets (`africa/`, `data/`, `models/`), and "
        "configure the environment in under 5 seconds.\n"
    )
    if "Google Colab Quickstart" not in c0_text:
        # Add after Colab badge
        if "[![Open In Colab]" in c0_text:
            parts = c0_text.split("[![Open In Colab]", 1)
            subparts = parts[1].split(")\n\n", 1)
            if len(subparts) == 2:
                c0_text = parts[0] + "[![Open In Colab]" + subparts[0] + ")\n\n" + instruction + "\n" + subparts[1]
                cells[0]["source"] = [line + "\n" for line in c0_text.split("\n")[:-1]] + [c0_text.split("\n")[-1]]

    # 2. Check if Bootstrap cell already exists
    has_bootstrap = any("STEP 1: GOOGLE COLAB 1-CLICK BOOTSTRAP" in "".join(c["source"]) for c in cells)
    if not has_bootstrap:
        bootstrap_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": COLAB_BOOTSTRAP_CELL_SOURCE,
        }
        # Insert right after Cell 0 (or Cell 1)
        cells.insert(1, bootstrap_cell)
        print("Inserted Colab bootstrap cell at index 1 in model-2.ipynb")

    # Find the cell that ingests regional chronologies
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if "regional_rwl_paths = [PROJECT_ROOT" in src:
            if "Self-healing environment guard" not in src:
                c["source"] = CELL_4_SELF_HEALING_PREFIX + c["source"]
                print(f"Added self-healing guard to cell {i} (regional RWL ingestion)")
            break

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully patched model-2.ipynb")

def patch_model_1():
    path = Path("model-1.ipynb")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    has_bootstrap = any("STEP 1: GOOGLE COLAB 1-CLICK BOOTSTRAP" in "".join(c["source"]) for c in cells)
    if not has_bootstrap:
        bootstrap_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": COLAB_BOOTSTRAP_CELL_SOURCE,
        }
        cells.insert(1, bootstrap_cell)
        print("Inserted Colab bootstrap cell at index 1 in model-1.ipynb")

    # Add self-healing to Cell 4 (candidate discovery)
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if 'rwl_files = sorted(list((PROJECT_ROOT / "africa").glob("eth*.rwl")))' in src:
            if "Self-healing environment guard" not in src:
                guard = [
                    "# Self-healing environment guard\n",
                    "if \"PROJECT_ROOT\" not in globals() or not (PROJECT_ROOT / \"africa\").exists():\n",
                    "    import sys, os, subprocess\n",
                    "    from pathlib import Path\n",
                    "    for cand in [Path(\"/content/Fradscr\"), Path(\".\").resolve(), Path(\"..\").resolve()]:\n",
                    "        if (cand / \"africa\").exists():\n",
                    "            PROJECT_ROOT = cand\n",
                    "            break\n",
                    "    else:\n",
                    "        if os.path.exists(\"/content\"):\n",
                    "            subprocess.run([\"git\", \"clone\", \"https://github.com/ezekiyastsegaye123-cmyk/Fradscr.git\", \"/content/Fradscr\"], check=True)\n",
                    "            PROJECT_ROOT = Path(\"/content/Fradscr\")\n",
                    "        else:\n",
                    "            PROJECT_ROOT = Path(\".\").resolve()\n",
                    "    if str(PROJECT_ROOT) not in sys.path:\n",
                    "        sys.path.insert(0, str(PROJECT_ROOT))\n",
                    "\n",
                ]
                c["source"] = guard + c["source"]
                print(f"Added self-healing guard to model-1 cell {i}")
            break

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully patched model-1.ipynb")

if __name__ == "__main__":
    patch_model_2()
    patch_model_1()
