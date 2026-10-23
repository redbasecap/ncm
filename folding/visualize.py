"""Export coarse-to-fine NCM folds for the web viewer: true CA trace plus the CA trace decoded after
every generation level, superposed onto the truth.

  python -m folding.visualize --split test --out runs/folding/viz.json
"""
import argparse
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from folding.data import AA, AA_PAD
from folding.geometry import kabsch, lddt_ca, rmsd, tm_score_kabsch
from folding.tokens import AngleTokenizer
from folding.train import DEV, Windows, build_ncm, generate_codes, load_vqvae, out_dir, prepare


@torch.no_grad()
def level_structs(vq, codes):
    """Angle tokens decoded from the cumulative reconstruction after each level: (n_levels, B, T)."""
    # Mirrors MultiscaleResidualVectorQuantizer.idx_to_zhats, which accumulates in place and so returns
    # the same tensor for every level; here each level's cumulative reconstruction is kept separately.
    q = vq.vq
    z_hat = torch.zeros(codes.shape[0], q.levels[-1], q.embed_dim, device=codes.device)
    out, acc = [], 0
    for i, n in enumerate(q.levels):
        e = q.embeddings[i](codes[:, acc:acc + n])
        e = q._up_to_full(e, i, q.levels[-1]) if i != len(q.levels) - 1 else e
        z_hat = z_hat + q._apply_quant_resi(e, i)
        out.append(vq.decoder(z_hat).argmax(-1))
        acc += n
    return torch.stack(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/folding/cath128.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="runs/folding/viz.json")
    ap.add_argument("--best", type=int, default=4)
    ap.add_argument("--typical", type=int, default=2)
    a = ap.parse_args()
    cfg = OmegaConf.load(a.config)
    P = prepare(cfg)
    tok = AngleTokenizer(P["tokenizer"]["centroids"])
    pad, T = int(cfg.angle_tokens) + 1, int(cfg.context_length)
    vq = load_vqvae(cfg)
    net = build_ncm(cfg, vq)
    net.load_state_dict(torch.load(f"{out_dir(cfg)}/generator.pt", map_location="cpu")["model"])
    net.eval()
    segs = [s for s in P["splits"][a.split] if int(cfg.data.eval_min_len) <= len(s["seq"]) <= T]
    W = Windows(segs, T, pad)

    def to_ca(t, true_t):
        t = t.copy()
        t[true_t == tok.TERM] = tok.TERM
        return tok.decode(np.where(t >= tok.K, tok.TERM, t))

    scored = []
    for b in range(0, len(segs), 64):
        chunk = segs[b:b + 64]
        seq, _, lens = map(np.stack, zip(*[W.crop(s) for s in chunk]))
        levels = level_structs(vq, generate_codes(net, torch.from_numpy(seq).to(DEV)))
        for i, (s, L) in enumerate(zip(chunk, lens)):
            scored.append((lddt_ca(to_ca(levels[-1, i, :L], s["struct"][:L]), s["ca"]), s, levels[:, i, :L]))
    scored.sort(key=lambda x: -x[0])
    picks = [("best", x) for x in scored[:a.best // 2]]
    picks += [("best long", x) for x in [x for x in scored if len(x[1]["seq"]) >= 80][:a.best - a.best // 2]]
    mid = len(scored) // 2
    picks += [("typical", x) for x in scored[mid:mid + a.typical]]

    proteins = []
    for kind, (_, s, levels) in picks:
        true = s["ca"] - s["ca"].mean(0)
        frames = []
        for k, lv in enumerate(levels):
            X = kabsch(to_ca(lv, s["struct"]), true)
            frames.append({"ca": np.round(X, 2).tolist(), "lddt": round(lddt_ca(X, true), 3),
                           "tm": round(tm_score_kabsch(X, true), 3), "rmsd": round(rmsd(X, true), 2)})
        proteins.append({"name": s["name"], "kind": kind, "seq": "".join(AA[i] if i < len(AA) else "X" for i in s["seq"]),
                         "true": np.round(true, 2).tolist(), "frames": frames})
        print(f"{kind:10s} {s['name']:8s} L={len(true):3d} level lDDT: " +
              " ".join(f"{f['lddt']:.2f}" for f in frames))
    json.dump({"levels": list(vq.vq.levels), "split": a.split, "n_ranked": len(scored), "proteins": proteins},
              open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
