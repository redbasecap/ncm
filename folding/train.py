"""Train the protein folding prototype: stage 1 VQ-VAE over angle tokens, stage 2 NCM generator.

  python -m folding.train prepare  --config config/folding/cath128.yaml
  python -m folding.train vqvae    --config config/folding/cath128.yaml
  python -m folding.train overfit  --config config/folding/cath128.yaml   # generate() must reproduce 8 chains
  python -m folding.train ncm      --config config/folding/cath128.yaml
  python -m folding.train evaluate --config config/folding/cath128.yaml --split test
"""
import argparse
import os
import pickle
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from folding.data import AA_PAD, load_cath
from folding.geometry import ca_internal, lddt_ca, rmsd, tm_score_kabsch
from folding.tokens import AngleTokenizer
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from utils.device import get_device

DEV = get_device()


def out_dir(cfg):
    d = os.path.join("runs", "folding", cfg.task)
    os.makedirs(d, exist_ok=True)
    return d


def prepare(cfg):
    path = os.path.join(cfg.data.root, f"tokens_k{cfg.angle_tokens}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    d = load_cath(cfg.data.root, cfg.data.min_len)
    tok = AngleTokenizer.fit([x["ca"] for x in d["train"]], int(cfg.angle_tokens), iters=25)
    for split in d.values():
        for x in split:
            x["struct"] = tok.encode(x["ca"])
    out = {"tokenizer": tok.state_dict(), "splits": d}
    pickle.dump(out, open(path, "wb"))
    return out


class Windows:
    """Random length-T crops of segments; seq padded with AA_PAD, struct with PAD."""

    def __init__(self, segs, T, pad, seed=0):
        self.segs, self.T, self.pad = segs, T, pad
        w = np.array([len(s["seq"]) for s in segs], dtype=np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def crop(self, s, start=0):
        n = min(len(s["seq"]), self.T)
        seq = np.full(self.T, AA_PAD, dtype=np.int64)
        st = np.full(self.T, self.pad, dtype=np.int64)
        seq[:n], st[:n] = s["seq"][start:start + n], s["struct"][start:start + n]
        return seq, st, n

    def batch(self, B):
        out = []
        for i in self.rng.choice(len(self.segs), B, p=self.p):
            s = self.segs[i]
            start = self.rng.integers(0, max(len(s["seq"]) - self.T, 0) + 1)
            out.append(self.crop(s, start)[:2])
        seq, st = map(np.stack, zip(*out))
        return torch.from_numpy(seq).to(DEV), torch.from_numpy(st).to(DEV)


def build_vqvae(cfg):
    return VQVAE(**cfg.tokenizer.vqvae).to(DEV)


def load_vqvae(cfg):
    vq = build_vqvae(cfg)
    vq.load_state_dict(torch.load(os.path.join(out_dir(cfg), "tokenizer.pt"), map_location="cpu")["model"])
    return vq.eval()


@torch.no_grad()
def vqvae_accuracy(vq, data, pad, n_batches=8, B=128):
    vq.eval()
    hit = tot = 0
    for _ in range(n_batches):
        _, st = data.batch(B)
        logits = vq.decode(vq.encode(st))
        m = st != pad
        hit += (logits.argmax(-1) == st)[m].sum().item()
        tot += m.sum().item()
    return hit / tot


def train_vqvae(cfg, P):
    pad = int(cfg.angle_tokens) + 1
    T = int(cfg.context_length)
    tr = Windows(P["splits"]["train"], T, pad)
    va = Windows(P["splits"]["validation"], T, pad, seed=1)
    vq = build_vqvae(cfg)
    tc = cfg.tokenizer.train
    opt = torch.optim.AdamW(vq.parameters(), lr=float(tc.lr), weight_decay=0.01)
    t0 = time.time()
    for step in range(1, int(tc.steps) + 1):
        vq.train()
        for g in opt.param_groups:
            g["lr"] = float(tc.lr) * min(1.0, step / 200) * 0.5 * (1 + np.cos(np.pi * step / int(tc.steps)))
        _, st = tr.batch(int(tc.batch_size))
        logits, vq_loss, util, gutil, _ = vq(st)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), st.reshape(-1), ignore_index=pad)
        loss = ce + vq_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vq.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == 1:
            acc = vqvae_accuracy(vq, va, pad)
            print(f"[vqvae] step {step} ce {ce.item():.4f} vq {float(vq_loss.detach()):.4f} codebook {gutil:.2f} "
                  f"min-level {min(util):.2f} val-recon-acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
            for i in range(len(vq.vq.levels)):
                getattr(vq.vq, f"codebook_hit_{i}").zero_()
            torch.save({"model": vq.state_dict()}, os.path.join(out_dir(cfg), "tokenizer.pt"))
    return vq


def build_ncm(cfg, vq):
    return NextConceptTransformer(vqvae_local=vq, config=cfg.model).to(DEV)


def ncm_inputs(net, vq, st):
    """Teacher forcing: target codes (B, L) and NCM inputs cat(level-0 queries, next-concept inputs)."""
    idx = vq.encode(st)
    x = torch.cat([net.build_l0_query_input(st.shape[0]), vq.vq.idx_to_ncm_input(idx)], dim=1)
    return idx, x


def ncm_loss(net, vq, seq, st):
    idx, x = ncm_inputs(net, vq, st)
    logits = net(x, prefix_ids=seq, prefix_pad_mask=seq == AA_PAD)
    assert logits.shape[1] == idx.shape[1] == sum(vq.vq.levels), (logits.shape, idx.shape)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1)), logits, idx


@torch.no_grad()
def generate_codes(net, seq, temperature=0.0):
    return net.generate(None, seq, temperature=max(temperature, 1e-8), top_k=1 if temperature == 0 else 0,
                        top_p=0.0 if temperature == 0 else 1.0)


@torch.no_grad()
def codes_to_struct(vq, codes):
    _, zl = vq.vq.idx_to_zhats(codes, return_zlevels=True)
    return vq.decoder(zl[-1]).argmax(-1)


def train_ncm(cfg, P, overfit=0):
    pad = int(cfg.angle_tokens) + 1
    T = int(cfg.context_length)
    vq = load_vqvae(cfg)
    for p in vq.parameters():
        p.requires_grad = False
    if overfit:
        cfg.model.dropout = 0.0
    net = build_ncm(cfg, vq)
    mc = cfg.model.train
    tr = Windows(P["splits"]["train"], T, pad)
    if overfit:
        fixed = tr.batch(overfit)
    steps = 600 if overfit else int(mc.steps)
    lr = 1e-3 if overfit else float(mc.lr)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0 if overfit else 0.05)
    va = Windows(P["splits"]["validation"], T, pad, seed=1)
    t0 = time.time()
    for step in range(1, steps + 1):
        net.train()
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / 300) * (1.0 if overfit else 0.5 * (1 + np.cos(np.pi * step / steps)))
        seq, st = fixed if overfit else tr.batch(int(mc.batch_size))
        loss, _, _ = ncm_loss(net, vq, seq, st)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % (100 if overfit else 500) == 0 or step == 1:
            msg = f"[ncm] step {step} loss {loss.item():.4f}"
            if not overfit:
                net.eval()
                with torch.no_grad():
                    vl = np.mean([ncm_loss(net, vq, *va.batch(64))[0].item() for _ in range(4)])
                msg += f" val {vl:.4f}"
                torch.save({"model": net.state_dict()}, os.path.join(out_dir(cfg), "generator.pt"))
            print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
    if overfit:
        net.eval()
        seq, st = fixed
        idx, _ = ncm_inputs(net, vq, st)
        g = generate_codes(net, seq)
        codes_ok = (g == idx).float().mean().item()
        rec = codes_to_struct(vq, g)
        m = st != pad
        print(f"[overfit] generated codes == teacher-forcing targets: {codes_ok:.4f}; "
              f"decoded angle tokens correct: {(rec == st)[m].float().mean().item():.4f}", flush=True)
        assert codes_ok > 0.99, "generate() does not reproduce memorized codes: train/inference mismatch"
    return net


def evaluate(cfg, P, split="test", n=0, temperature=0.0):
    pad = int(cfg.angle_tokens) + 1
    T = int(cfg.context_length)
    tok = AngleTokenizer(P["tokenizer"]["centroids"])
    vq = load_vqvae(cfg)
    net = build_ncm(cfg, vq)
    net.load_state_dict(torch.load(os.path.join(out_dir(cfg), "generator.pt"), map_location="cpu")["model"])
    net.eval()
    segs = [s for s in P["splits"][split] if int(cfg.data.eval_min_len) <= len(s["seq"]) <= T]
    segs = segs[:n] if n else segs
    W = Windows(segs, T, pad)
    rows = {k: [] for k in ("ncm", "oracle", "per_aa", "majority")}
    acc = {k: [] for k in rows}
    mae = {k: [] for k in rows}
    train = P["splits"]["train"]
    # Baselines: most frequent angle token overall, and per amino acid.
    aa_all = np.concatenate([s["seq"] for s in train])
    st_all = np.concatenate([s["struct"] for s in train])
    angle = st_all < tok.K
    majority = np.bincount(st_all[angle], minlength=tok.K).argmax()
    per_aa = np.array([np.bincount(st_all[angle & (aa_all == a)], minlength=tok.K).argmax()
                       if (angle & (aa_all == a)).any() else majority for a in range(22)])
    for b in range(0, len(segs), 64):
        chunk = segs[b:b + 64]
        seq, st, lens = map(np.stack, zip(*[W.crop(s) for s in chunk]))
        pred = codes_to_struct(vq, generate_codes(net, torch.from_numpy(seq).to(DEV), temperature)).cpu().numpy()
        for s, p, L in zip(chunk, pred, lens):
            true_t = s["struct"][:L]
            cands = {"ncm": p[:L], "oracle": true_t, "per_aa": per_aa[s["seq"][:L]],
                     "majority": np.full(L, majority)}
            for k, t in cands.items():
                t = t.copy()
                t[true_t == tok.TERM] = tok.TERM  # chain ends are known from the sequence length
                X = tok.decode(np.where(t >= tok.K, tok.TERM, t))
                rows[k].append((rmsd(X, s["ca"]), tm_score_kabsch(X, s["ca"]), lddt_ca(X, s["ca"])))
                m = true_t < tok.K
                acc[k].append((t[m] == true_t[m]).mean())
                th, ta = ca_internal(s["ca"])
                ok = m & (t < tok.K) & ~np.isnan(th) & ~np.isnan(ta)
                circ = lambda a, b: np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
                mae[k].append(np.rad2deg(np.concatenate([circ(tok.theta[t[ok]], th[ok]), circ(tok.tau[t[ok]], ta[ok])])).mean())
    print(f"\n{split}: {len(segs)} resolved segments, {cfg.data.eval_min_len}-{T} residues, "
          f"CATH-topology split (folds unseen in training)\n")
    print(f"{'method':>10} {'token acc':>10} {'angle MAE':>10} {'CA RMSD':>9} {'TM (Kabsch)':>12} {'CA lDDT':>8}")
    for k in ("majority", "per_aa", "ncm", "oracle"):
        r = np.array(rows[k])
        print(f"{k:>10} {np.mean(acc[k]):10.3f} {np.mean(mae[k]):9.1f}° {r[:, 0].mean():9.2f} {r[:, 1].mean():12.3f} "
              f"{r[:, 2].mean():8.3f}")
    return rows, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "vqvae", "overfit", "ncm", "evaluate"])
    ap.add_argument("--config", default="config/folding/cath128.yaml")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="override training steps")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. tokenizer.train.lr=5e-4")
    ap.add_argument("--run", default="", help="run name under runs/folding/ (default: task)")
    a = ap.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_dotlist(a.set))
    if a.run:
        cfg.task = a.run
    if a.steps:
        cfg.tokenizer.train.steps = cfg.model.train.steps = a.steps
    torch.manual_seed(0)
    print(f"[folding] device {DEV}", flush=True)
    P = prepare(cfg)
    if a.stage == "vqvae":
        train_vqvae(cfg, P)
    elif a.stage == "overfit":
        train_ncm(cfg, P, overfit=8)
    elif a.stage == "ncm":
        train_ncm(cfg, P)
    elif a.stage == "evaluate":
        evaluate(cfg, P, a.split, a.n)


if __name__ == "__main__":
    main()
