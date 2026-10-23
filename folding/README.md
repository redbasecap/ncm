# NCM for protein folding (prototype)

An experiment: can Next-Concept Modeling predict a protein's 3D backbone from its amino-acid sequence? It is a small proof of concept trained on one Apple M3 Max, and it is nowhere near AlphaFold.

**Short answer:** the model learns local structure clearly better than simple baselines (angle error 31.5° vs 47.1°, CA lDDT 0.35 vs 0.27). It does not learn the global fold: TM-score stays at the level of a random structure (0.06).

## Setup

- **Data:** CATH 4.2 chains from Ingraham et al. 2019 (`chain_set.jsonl`, 517 MB), with the official split by CATH topology. Test proteins have folds that never occur in training. Only resolved segments without chain breaks are used.
- **Structure representation:** one (θ, τ) pair per residue: the virtual CA bond angle and CA dihedral. With a fixed 3.806 Å CA-CA distance, this rebuilds real chains to 0.1–0.4 Å RMSD. φ/ψ with ideal bond geometry drifted by 4–14 Å even from the true angles, so it was dropped.
- **Tokens:** k-means of (cos θ, sin θ, cos τ, sin τ) into 256 clusters, plus TERM (chain ends) and PAD.
- **Stage 1, VQ-VAE** (6.2M params): multiscale residual VQ over 128-residue windows, 13 levels (1 … 128), 256 codes per level. It reaches 99.99% reconstruction on validation after 1,000 steps. 64 codes per level was too little (6 bits for 256 tokens), and without dead-code restart the codebooks collapsed.
- **Stage 2, NCM generator** (22.7M params): the sequence enters through the `ctx_encoder` prefix, and the levels are generated coarse to fine. Training ran 8,000 steps at batch 64 (about 4.5 h on `mps`). Validation loss fell from 5.70 to 1.83, still slightly decreasing at the end.
- **Overfit gate:** after memorizing 8 chains, `generate()` reproduces their codes exactly (100%), so the training inputs match the inference path.

## Results

Test split: 690 resolved segments of 40–128 residues, folds unseen in training. Decoding is greedy.

| Method | Token acc | Angle MAE | CA RMSD | TM-score¹ | CA lDDT |
|---|---|---|---|---|---|
| Most frequent token | 0.008 | 57.0° | 76.4 Å | 0.014 | 0.159 |
| Most frequent token per amino acid | 0.013 | 47.1° | 20.2 Å | 0.063 | 0.268 |
| **NCM** | **0.017** | **31.5°** | 26.8 Å | 0.061 | **0.353** |
| Oracle (true tokens; ceiling of the representation) | 1.000 | 1.6° | 2.8 Å | 0.684 | 0.755 |

¹ TM-score under the RMSD-optimal (Kabsch) superposition, a lower bound on the TM-align score. Values below about 0.17 are what unrelated structures get.

Validation gives the same picture (NCM angle MAE 33.9° vs 48.4°, lDDT 0.355 vs 0.276).

**Interpretation:**
- **Local structure:** NCM gets much closer than the per-amino-acid baseline, judging by the angle error and lDDT, which compares local CA distances.
- **Global fold:** angle errors add up along the chain, so the overall 3D shape is still wrong (RMSD, TM-score).
- **Token accuracy:** exact-token accuracy says little, because many of the 256 clusters sit close together in the helix region.

## What would likely help

- Predict structure without accumulating error: pairwise distances or frames instead of chained angles.
- Finer tokens, since the 256-cluster ceiling is TM 0.68.
- Much more training (the paper's generators train for 50k–350k steps at batch 512–2,048).
- Evolutionary input such as MSAs or protein language-model embeddings, which is what makes AlphaFold-class models work.

## Watch it fold live

```bash
python -m folding.live      # then open http://127.0.0.1:8770
```

NCM folds test proteins one after another on this machine and streams each of the 13 levels to the browser as `generate()` produces it. The page shows the prediction against the true structure, per-level lDDT/TM/RMSD and the measured model time (about 6–8 ms per level on an M3 Max). You can also paste your own sequence of 10–128 residues. `python -m folding.visualize` exports the same per-level traces to JSON.

## Run it

```bash
# data: chain_set.jsonl and chain_set_splits.json into data/cath/ (URLs in folding/data.py)
python -m folding.train prepare            # angle tokens, cached in data/cath/
python -m folding.train vqvae --steps 1000 # stage 1
python -m folding.train overfit            # sanity gate: generate() must reproduce memorized chains
python -m folding.train ncm --steps 8000   # stage 2
python -m folding.train evaluate --split test
python -m pytest tests/test_folding_geometry.py
```

Training on `mps` needs `utils.device.contiguous_grad`, which works around an mps convolution backward abort ("Weights tensor and ndArray input channel mismatch") in the level-0 down-sampler.
