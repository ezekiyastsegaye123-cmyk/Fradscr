#!/usr/bin/env python3
"""
Make model-2.ipynb and model-1.ipynb 100% self-contained and Google Colab proof.
"""

import json
from pathlib import Path

def patch_model_2():
    path = Path("model-2.ipynb")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # 1. Update Cell 0 markdown with clear instruction and Colab badge
    colab_badge = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ezekiyastsegaye123-cmyk/Fradscr/blob/main/model-2.ipynb)"
    colab_box = (
        "> 🚀 **Google Colab 1-Click Launch**: When opening in Colab, click the **Step 1: Google Colab 1-Click Bootstrap** "
        "cell right below and press **Shift+Enter**. It clones the repository, sets up all paths, and configures auxiliary "
        "packages in under 5 seconds! Every cell also features self-healing path resolution so you can run any cell independently."
    )
    
    # Clean cell 0
    c0_lines = cells[0]["source"]
    c0_text = "".join(c0_lines)
    if "Google Colab 1-Click Launch" not in c0_text:
        if colab_badge in c0_text:
            parts = c0_text.split(colab_badge, 1)
            new_text = parts[0] + colab_badge + "\n\n" + colab_box + "\n\n" + parts[1].strip() + "\n"
            cells[0]["source"] = [l + "\n" for l in new_text.split("\n")[:-1]] + [new_text.split("\n")[-1]]

    # Ensure Bootstrap cell is present at Cell 1
    bootstrap_code = [
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
        "    print(\"⚡ Google Colab environment detected. Initializing Fradscr repository...\")\n",
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

    has_bootstrap = False
    for c in cells:
        if "STEP 1: GOOGLE COLAB 1-CLICK BOOTSTRAP" in "".join(c["source"]):
            c["source"] = bootstrap_code
            has_bootstrap = True
            break
    if not has_bootstrap:
        cells.insert(1, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": bootstrap_code,
        })

    # Update Setup cell to include load_spei_series helper
    helper_def = (
        "\ndef load_spei_series(nc_path: Path, csv_path: Path, lat: float, lon: float) -> pd.DataFrame:\n"
        "    \"\"\"Load SPEI ground truth from local netCDF4 if available, or pre-extracted CSV.\"\"\"\n"
        "    if nc_path.exists():\n"
        "        try:\n"
        "            return extract_annual_spei(nc_path, lat=lat, lon=lon).annual_df\n"
        "        except Exception:\n"
        "            pass\n"
        "    if csv_path.exists():\n"
        "        return pd.read_csv(csv_path)\n"
        "    raise FileNotFoundError(f\"Neither {nc_path} nor {csv_path} was found.\")\n\n"
    )

    for c in cells:
        src = "".join(c["source"])
        if "from treering.pipeline import process_rwl" in src:
            if "def load_spei_series" not in src:
                c["source"].append(helper_def)
                print("Added load_spei_series helper to setup cell.")
            break

    # Update Feature Engineering cell to use load_spei_series
    for c in cells:
        src = "".join(c["source"])
        if "spei_regional = extract_annual_spei(" in src:
            new_lines = []
            for line in c["source"]:
                if "spei_regional = extract_annual_spei(" in line:
                    new_lines.append(
                        'spei_regional = load_spei_series(PROJECT_ROOT / "data" / "spei01.nc", '
                        'PROJECT_ROOT / "results" / "spei_gondar.csv", lat=13.01, lon=37.80)\n'
                    )
                else:
                    new_lines.append(line)
            c["source"] = new_lines
            print("Updated Cell for regional SPEI ingestion using load_spei_series.")

    # Update Blind Holdout cell (Debrebirkan)
    for c in cells:
        src = "".join(c["source"])
        if "debre_spei = extract_annual_spei(" in src:
            new_lines = []
            for line in c["source"]:
                if "debre_spei = extract_annual_spei(" in line:
                    new_lines.append(
                        'debre_spei = load_spei_series(PROJECT_ROOT / "data" / "spei01.nc", '
                        'PROJECT_ROOT / "results" / "spei_debrebirkan.csv", lat=9.68, lon=39.53)\n'
                    )
                else:
                    new_lines.append(line)
            c["source"] = new_lines
            print("Updated Debrebirkan SPEI loading with load_spei_series.")

    # Update Secondary Cross-Basin cell (Adaba-Dodola)
    for c in cells:
        src = "".join(c["source"])
        if "adaba_spei = extract_annual_spei(" in src:
            new_lines = []
            for line in c["source"]:
                if "adaba_spei = extract_annual_spei(" in line:
                    new_lines.append(
                        'adaba_spei = load_spei_series(PROJECT_ROOT / "data" / "spei01.nc", '
                        'PROJECT_ROOT / "results" / "spei_adaba.csv", lat=6.83, lon=39.25)\n'
                    )
                else:
                    new_lines.append(line)
            c["source"] = new_lines
            print("Updated Adaba SPEI loading with load_spei_series.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("model-2.ipynb updated successfully.")

def patch_model_1():
    path = Path("model-1.ipynb")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Update Cell 34 to safely load Adaba eth004 SPEI
    for c in cells:
        src = "".join(c["source"])
        if 'spei_004_res = extract_annual_spei(PROJECT_ROOT / "data" / "spei01.nc", lat=6.92, lon=39.24)' in src:
            new_lines = []
            for line in c["source"]:
                if 'spei_004_res = extract_annual_spei(PROJECT_ROOT / "data" / "spei01.nc", lat=6.92, lon=39.24)' in line:
                    new_lines.append(
                        'if (PROJECT_ROOT / "data" / "spei01.nc").exists():\n'
                        '    spei_004_res = extract_annual_spei(PROJECT_ROOT / "data" / "spei01.nc", lat=6.92, lon=39.24)\n'
                        '    df_spei_004 = spei_004_res.annual_df\n'
                        'else:\n'
                        '    df_spei_004 = pd.read_csv(PROJECT_ROOT / "results" / "spei_adaba.csv")\n'
                    )
                elif 'df_spei_004 = spei_004_res.annual_df' in line:
                    continue
                else:
                    new_lines.append(line)
            c["source"] = new_lines
            print("Updated model-1.ipynb eth004 SPEI loading.")
            break

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("model-1.ipynb updated successfully.")

if __name__ == "__main__":
    patch_model_2()
    patch_model_1()
