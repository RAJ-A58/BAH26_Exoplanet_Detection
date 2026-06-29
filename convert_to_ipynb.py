import json
import os

with open('generate_final_pdf.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Make it Jupyter friendly!
code = code.replace('matplotlib.use("Agg")', '# matplotlib.use("Agg")  # Disabled for Jupyter inline plotting')

# Inject plt.show() so the graphs display directly inside the notebook cells!
code = code.replace('print("    Cover done.")', 'print("    Cover done.")\nplt.show()')
code = code.replace('print("    Evaluation page done.")', 'print("    Evaluation page done.")\nplt.show()')
code = code.replace('print("    Light curve page done (REAL data).")', 'print("    Light curve page done (REAL data).")\nplt.show()')
code = code.replace('print("    WARNING: No demo targets loaded — placeholder page used.")', 'print("    WARNING: No demo targets loaded — placeholder page used.")\nplt.show()')
code = code.replace('print("    Benchmark table done.")', 'print("    Benchmark table done.")\nplt.show()')
code = code.replace('print("    Architecture diagram done.")', 'print("    Architecture diagram done.")\nplt.show()')

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    lines = [line + "\n" for line in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

def add_code(text):
    lines = [line + "\n" for line in text.rstrip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    notebook['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

# Markdown Title
add_md("""# Exoplanet Detection Pipeline — Final Demo

**AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves**

This notebook runs the complete pipeline end-to-end, displays the charts inline, and exports a PDF report.

**Sections:**
1. Setup & model loading
2. Synthetic model evaluation (accuracy, PR-AUC, ROC-AUC, confusion matrix)
3. Real NASA light curve demos — phase-folded plots with confidence overlay
4. Benchmark table (15 targets — planets + false positives)
5. Export to PDF""")

# Import block
import_code = code[code.find('import warnings'):code.find('# ════════════════')]
add_code(import_code.strip())

# Blocks splitting logic
blocks = code.split('# ══════════════════════════════════════════════════════════════════════════════')
for block in blocks:
    if '# STEP 1' in block:
        add_md("## 1. Load trained model and synthetic validation data")
        add_code(block.split('holdout\n')[-1].strip())
    elif '# STEP 2' in block:
        add_md("## 2. Load real Kepler light curves from local cache")
        add_code(block.split('cache\n')[-1].strip())
    elif '# STEP 3' in block:
        add_md("## 3. Build benchmark table data")
        add_code(block.split('data\n')[-1].strip())
    elif '# STEP 4' in block:
        add_md("## 4. Build all figures")
        # Split STEP 4 into sub-cells for each figure so they render beautifully
        step4_code = block.split('figures\n')[-1].strip()
        
        # Cell for Cover
        cover_code = step4_code[:step4_code.find('# ── PAGE 2')]
        add_md("### 4.1 Title & Architecture Summary")
        add_code(cover_code.strip())
        
        # Cell for Eval
        eval_code = step4_code[step4_code.find('# ── PAGE 2'):step4_code.find('# ── PAGE 3')]
        add_md("### 4.2 Synthetic Holdout Evaluation")
        add_code(eval_code.strip())
        
        # Cell for Light curves
        lc_code = step4_code[step4_code.find('# ── PAGE 3'):step4_code.find('# ── PAGE 4')]
        add_md("### 4.3 Real NASA Kepler Data")
        add_code(lc_code.strip())
        
        # Cell for Benchmark
        bench_code = step4_code[step4_code.find('# ── PAGE 4'):step4_code.find('# ── PAGE 5')]
        add_md("### 4.4 Final Benchmark Results (15/15)")
        add_code(bench_code.strip())
        
        # Cell for Arch
        arch_code = step4_code[step4_code.find('# ── PAGE 5'):]
        add_md("### 4.5 Full Architecture Diagram")
        add_code(arch_code.strip())
        
    elif '# STEP 5' in block:
        add_md("## 5. Write PDF")
        add_code(block.split('PDF\n')[-1].strip())

with open('exoplanet_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
