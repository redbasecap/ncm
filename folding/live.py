"""Live folding viewer: NCM folds proteins on this machine and streams every generation level to a browser.

  python -m folding.live            # then open http://127.0.0.1:8770
  python -m folding.live --pace 0.6 # seconds shown per level (generation itself takes milliseconds)

Each level is captured from NextConceptTransformer.generate() itself (via _sample_level), decoded to a CA
trace from the cumulative reconstruction so far, and sent as a server-sent event.
"""
import argparse
import json
import os
import queue
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from omegaconf import OmegaConf

from folding.data import AA, AA_PAD, encode_seq
from folding.geometry import kabsch, lddt_ca, rmsd, tm_score_kabsch
from folding.tokens import AngleTokenizer
from folding.train import DEV, build_ncm, load_vqvae, out_dir, prepare

HERE = os.path.dirname(os.path.abspath(__file__))


class LiveFolder:
    def __init__(self, cfg):
        P = prepare(cfg)
        self.tok = AngleTokenizer(P["tokenizer"]["centroids"])
        self.T = int(cfg.context_length)
        self.vq = load_vqvae(cfg)
        self.net = build_ncm(cfg, self.vq)
        self.net.load_state_dict(torch.load(f"{out_dir(cfg)}/generator.pt", map_location="cpu")["model"])
        self.net.eval()
        lo = int(cfg.data.eval_min_len)
        self.test = [s for s in P["splits"]["test"] if lo <= len(s["seq"]) <= self.T]
        self.order = []
        self.lock = threading.Lock()  # one fold at a time on the GPU
        self._orig_sample = self.net._sample_level

    def next_test(self):
        if not self.order:
            self.order = random.sample(range(len(self.test)), len(self.test))
        return self.test[self.order.pop()]

    def _decode(self, z_hat, L):
        t = self.vq.decoder(z_hat).argmax(-1)[0, :L].cpu().numpy()
        t[[0, L - 2, L - 1]] = self.tok.TERM  # chain ends: angles undefined, known from the length
        return self.tok.decode(np.where(t >= self.tok.K, self.tok.TERM, t))

    def fold(self, seq_ids, true_ca=None):
        """Yields one dict per generation level as the model produces it.

        Everything that touches the GPU (generation and decoding each level) runs in one worker thread,
        because mps hangs when two threads use it at once. The caller only formats and paces the output.
        """
        L = len(seq_ids)
        prefix = np.full(self.T, AA_PAD, dtype=np.int64)
        prefix[:L] = seq_ids
        q = queue.Queue()
        vqm = self.vq.vq
        st = {"k": 0, "z": torch.zeros(1, vqm.levels[-1], vqm.embed_dim, device=DEV), "prev": None, "t": None}

        def sync():
            if DEV == "mps":
                torch.mps.synchronize()
            elif DEV == "cuda":
                torch.cuda.synchronize()

        def capture(logits, *a):
            pred = self._orig_sample(logits, *a)
            sync()
            t_ready = time.perf_counter()
            k = st["k"]
            e = vqm.embeddings[k](pred)
            e = vqm._up_to_full(e, k, vqm.levels[-1]) if k != len(vqm.levels) - 1 else e
            st["z"] = st["z"] + vqm._apply_quant_resi(e, k)
            X = self._decode(st["z"], L)
            ref = true_ca if true_ca is not None else st["prev"]
            X = kabsch(X, ref) if ref is not None else X - X.mean(0)
            st["prev"] = X
            out = {"k": k, "n": int(vqm.levels[k]), "ms": round((t_ready - st["t"]) * 1000, 1), "ca": np.round(X, 2).tolist()}
            if true_ca is not None:
                out.update(lddt=round(lddt_ca(X, true_ca), 3), tm=round(tm_score_kabsch(X, true_ca), 3),
                           rmsd=round(rmsd(X, true_ca), 2))
            q.put(out)
            st["k"] += 1
            sync()
            st["t"] = time.perf_counter()  # decoding for the viewer is not counted as model time
            return pred

        def run():
            try:
                with torch.no_grad():
                    st["t"] = time.perf_counter()
                    self.net.generate(None, torch.from_numpy(prefix)[None].to(DEV), temperature=1e-8, top_k=1, top_p=0.0)
            except Exception as e:  # surface errors to the stream
                q.put(e)
            finally:
                q.put(None)

        with self.lock:
            self.net._sample_level = capture
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            try:
                while (item := q.get()) is not None:
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                worker.join()
                self.net._sample_level = self._orig_sample


def make_handler(folder, pace):
    page = open(os.path.join(HERE, "live.html"), encoding="utf-8").read()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if u.path != "/fold":
                return self._send(404, "not found", "text/plain")
            qs = parse_qs(u.query)
            raw = "".join(c for c in qs.get("seq", [""])[0].upper() if c.isalpha())
            if raw:
                if not 10 <= len(raw) <= folder.T:
                    return self._send(400, json.dumps({"error": f"Die Sequenz braucht 10 bis {folder.T} Aminosäuren, sie hat {len(raw)}."}), "application/json")
                seq_ids, true_ca, name, kind = encode_seq(raw), None, "eigene Sequenz", "own"
            else:
                s = folder.next_test()
                seq_ids, true_ca, name, kind = s["seq"], s["ca"] - s["ca"].mean(0), s["name"], "test"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(event, obj):
                self.wfile.write(f"event: {event}\ndata: {json.dumps(obj, separators=(',', ':'))}\n\n".encode())
                self.wfile.flush()

            try:
                emit("start", {"name": name, "kind": kind, "seq": "".join(AA[i] if i < len(AA) else "X" for i in seq_ids),
                               "levels": list(folder.vq.vq.levels),
                               "true": None if true_ca is None else np.round(true_ca, 2).tolist(), "device": DEV})
                total = 0.0
                for ev in folder.fold(seq_ids, true_ca):
                    total += ev["ms"]
                    emit("level", ev)
                    time.sleep(pace)
                emit("done", {"ms": round(total, 1)})
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/folding/cath128.yaml")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--pace", type=float, default=0.45, help="seconds each level stays on screen")
    a = ap.parse_args()
    folder = LiveFolder(OmegaConf.load(a.config))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(folder, a.pace))
    print(f"[live] {len(folder.test)} test proteins on {DEV}; open http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
