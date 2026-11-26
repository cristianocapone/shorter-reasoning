"""Demos -> program proposer: a small decoder-only transformer over BARC Python, conditioned on the grids.

Architecture (deliberately the smallest thing that can write code):
  * grid encoder — the emlang `PairEncoder` (conv + axial attention over a 30x30 canvas) applied to each
    (input, output) demo pair, then a transformer over the pair vectors -> K context tokens.
  * decoder — causal transformer over a byte-level-ish BPE of the program text, cross-attending nothing:
    the context tokens are simply prefixed to the sequence (prefix-LM), the same trick that lets the TRM
    executor read a message.
Sampling at test time gives K candidate programs; `synth/solve.py` executes and verifies them.

Tokenizer: a word-level vocabulary built from the corpus (identifiers, punctuation, numbers, indentation
runs), capped at `--vocab`. Rare tokens fall back to characters, so every program is representable.

    python synth/proposer.py train --data verified.jsonl --out run/ --steps 200000
    python synth/proposer.py sample --ckpt run/latest.pt --task task.json --k 32
"""
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("EMLANG_CANVAS", "30")
from emlang.model import PairEncoder  # noqa: E402
from emlang.data import pad_grid      # noqa: E402

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
# Lossless: newline+indent runs, a leading space glued to the next word/number/symbol (GPT-style), other
# space runs, then single non-space chars. "".join(tokenize(t)) == t for every string.
TOKEN_RE = re.compile(r"\n[ ]*| ?[A-Za-z_][A-Za-z_0-9]*| ?\d+| ?[^\sA-Za-z_0-9]|[ \t]+|[^\n]")


def tokenize(text):
    return TOKEN_RE.findall(text)


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {s: i for i, s in enumerate(itos)}

    @classmethod
    def build(cls, texts, size=8192, min_count=20):
        from collections import Counter
        c = Counter()
        for t in texts:
            c.update(tokenize(t))
        chars = {ch for t in texts[:2000] for ch in t}
        keep = [w for w, n in c.most_common() if n >= min_count][: size - len(SPECIAL) - len(chars)]
        return cls(SPECIAL + sorted(chars) + keep)

    def encode(self, text, max_len=None):
        out = [BOS]
        for w in tokenize(text):
            i = self.stoi.get(w)
            if i is not None:
                out.append(i)
            else:
                out.extend(self.stoi.get(ch, UNK) for ch in w)
            if max_len and len(out) >= max_len - 1:
                break
        if max_len:
            out = out[: max_len - 1]
        out.append(EOS)
        return out

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i >= len(SPECIAL))

    def save(self, path):
        json.dump(self.itos, open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))


class Proposer(nn.Module):
    def __init__(self, vocab, d=512, n_layer=8, n_head=8, n_ctx=8, d_grid=128, max_len=1024):
        super().__init__()
        self.d, self.n_ctx, self.max_len = d, n_ctx, max_len
        self.pair = PairEncoder(d_grid)
        layer = nn.TransformerEncoderLayer(d_grid, 4, 4 * d_grid, dropout=0.0, batch_first=True, norm_first=True)
        self.demo_ctx = nn.TransformerEncoder(layer, 2)
        self.to_ctx = nn.Linear(d_grid, n_ctx * d)
        self.tok = nn.Embedding(len(vocab.itos), d)
        nn.init.normal_(self.tok.weight, std=0.02)       # tied with the output head: unit-variance init gives loss ~190
        self.pos = nn.Parameter(torch.randn(1, max_len + n_ctx, d) * 0.02)
        dec = nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout=0.0, batch_first=True, norm_first=True,
                                         activation="gelu")
        self.dec = nn.TransformerEncoder(dec, n_layer)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(vocab.itos), bias=False)
        self.head.weight = self.tok.weight

    def context(self, din, dout, dmask):
        B, K = dmask.shape
        v = self.pair(din.flatten(0, 1), dout.flatten(0, 1)).view(B, K, -1)
        v = self.demo_ctx(v, src_key_padding_mask=~dmask)
        m = dmask.float().unsqueeze(-1)
        pooled = (v * m).sum(1) / m.sum(1).clamp(min=1)
        return self.to_ctx(pooled).view(B, self.n_ctx, self.d)

    def hidden(self, ctx, ids):
        x = torch.cat([ctx, self.tok(ids)], 1)
        x = x + self.pos[:, : x.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        mask[:, : self.n_ctx] = 0.0                       # every position may attend to the demo context
        return self.norm(self.dec(x, mask=mask, is_causal=False))

    def forward(self, ctx, ids):
        """Teacher-forced logits aligned with `ids`: output t predicts ids[t] from ctx + ids[<t]."""
        return self.head(self.hidden(ctx, ids))[:, self.n_ctx - 1 : -1]

    def next_logits(self, ctx, ids):
        """Distribution of the token AFTER the last one in `ids` (the last position of the full sequence).
        Using forward(...)[:, -1] here would re-predict ids[-1] and duplicate every token."""
        return self.head(self.hidden(ctx, ids)[:, -1])

    # ---------------------------------------------------------------- cached incremental decoding
    @staticmethod
    def _attn(layer, x, cache, mask):
        """Pre-norm self-attention of an nn.TransformerEncoderLayer with a growing (k, v) cache.
        x: [B, T, d] new positions; cache: dict with 'k','v' [B, H, S, dh] or empty; mask: [T, S+T] or None."""
        sa = layer.self_attn
        B, T, d = x.shape
        H = sa.num_heads; dh = d // H
        q, kk, v = F.linear(x, sa.in_proj_weight, sa.in_proj_bias).chunk(3, -1)
        q, kk, v = (t.view(B, T, H, dh).transpose(1, 2) for t in (q, kk, v))
        if cache:
            kk = torch.cat([cache["k"], kk], 2); v = torch.cat([cache["v"], v], 2)
        cache["k"], cache["v"] = kk, v
        y = F.scaled_dot_product_attention(q, kk, v, attn_mask=mask)
        return sa.out_proj(y.transpose(1, 2).reshape(B, T, d))

    def _step(self, x, caches, mask):
        for layer, cache in zip(self.dec.layers, caches):
            x = x + self._attn(layer, layer.norm1(x), cache, mask)
            x = x + layer.linear2(F.gelu(layer.linear1(layer.norm2(x))))
        return self.head(self.norm(x[:, -1]))

    @torch.no_grad()
    def sample(self, ctx, k, max_len, temperature=0.8, top_p=0.95, device="cuda", greedy=False):
        """KV-cached sampling: O(L) per token instead of recomputing the whole prefix (O(L^2))."""
        ctx = ctx.expand(k, -1, -1)
        ids = torch.full((k, 1), BOS, dtype=torch.long, device=device)
        done = torch.zeros(k, dtype=torch.bool, device=device)
        caches = [dict() for _ in self.dec.layers]
        x = torch.cat([ctx, self.tok(ids)], 1) + self.pos[:, : self.n_ctx + 1]
        P = x.shape[1]
        mask = torch.full((P, P), float("-inf"), device=device).triu(1)
        mask[:, : self.n_ctx] = 0.0                                   # same prefix mask as hidden()
        logits = self._step(x, caches, mask)
        for t in range(max_len - 1):
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                p = F.softmax(logits / max(temperature, 1e-5), -1)
                sp, si = p.sort(-1, descending=True)
                cut = (sp.cumsum(-1) - sp) > top_p
                sp[cut] = 0
                nxt = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
            nxt[done] = PAD
            ids = torch.cat([ids, nxt], 1)
            done |= nxt.squeeze(1).eq(EOS)
            if bool(done.all()) or t == max_len - 2:
                break
            x = self.tok(nxt) + self.pos[:, self.n_ctx + ids.shape[1] - 1 : self.n_ctx + ids.shape[1]]
            logits = self._step(x, caches, None)                     # new position sees everything cached
        return ids

    @torch.no_grad()
    def sample_nocache(self, ctx, k, max_len, device="cuda"):
        """Reference greedy decoder without cache, for the regression test only."""
        ctx = ctx.expand(k, -1, -1)
        ids = torch.full((k, 1), BOS, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            nxt = self.next_logits(ctx, ids).argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
        return ids


def collate(batch, vocab, max_len, max_demos=5):
    B = len(batch)
    din = np.full((B, max_demos, 30, 30), 10, np.int64)
    dout = np.full((B, max_demos, 30, 30), 10, np.int64)
    dmask = np.zeros((B, max_demos), np.bool_)
    ids = np.zeros((B, max_len), np.int64)
    for b, (pairs, src) in enumerate(batch):
        for k, (a, o) in enumerate(pairs[:max_demos]):
            din[b, k] = pad_grid(np.asarray(a)); dout[b, k] = pad_grid(np.asarray(o)); dmask[b, k] = True
        e = vocab.encode(src, max_len)
        ids[b, : len(e)] = e
    return (torch.from_numpy(din), torch.from_numpy(dout), torch.from_numpy(dmask), torch.from_numpy(ids))


def load_data(path, limit=0):
    data = []
    for line in open(path):
        r = json.loads(line)
        data.append((r["pairs"], r["source"]))
        if limit and len(data) >= limit:
            break
    return data


def train(a):
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    data = load_data(a.data, a.limit)
    rng = np.random.default_rng(0); idx = rng.permutation(len(data))
    hold = set(idx[: a.n_heldout].tolist())
    train_d = [d for i, d in enumerate(data) if i not in hold]
    hold_d = [d for i, d in enumerate(data) if i in hold]
    vpath = os.path.join(a.out, "vocab.json")
    vocab = Vocab.load(vpath) if os.path.exists(vpath) else Vocab.build([s for _, s in train_d], a.vocab)
    vocab.save(vpath)
    lens = [len(vocab.encode(s, 4096)) for _, s in train_d[:2000]]
    print(f"[data] {len(train_d)} train / {len(hold_d)} held-out programs | vocab {len(vocab.itos)} | "
          f"tokens per program: median {int(np.median(lens))} p95 {int(np.percentile(lens,95))} | max_len {a.max_len}", flush=True)
    model = Proposer(vocab, a.d, a.n_layer, a.n_head, a.n_ctx, max_len=a.max_len).to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"[model] {n/1e6:.1f}M params, d={a.d} layers={a.n_layer} on {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    step0 = 0
    ck = os.path.join(a.out, "latest.pt")
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step0 = s["step"]
        print(f"[resume] step {step0}", flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    rng = np.random.default_rng(a.seed)
    t0 = time.time(); ema = None; best_hl = float("inf")
    for step in range(step0, a.steps + 1):
        for g in opt.param_groups:
            warm = min(1.0, (step + 1) / a.warmup)
            g["lr"] = a.lr * warm * (0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
        sel = rng.integers(0, len(train_d), a.bs)
        batch = []
        for i in sel:
            pairs, src = train_d[i]
            take = rng.permutation(len(pairs))[: int(rng.integers(3, 6))]
            batch.append(([pairs[j] for j in take], src))
        din, dout, dmask, ids = (t.to(dev) for t in collate(batch, vocab, a.max_len))
        ctx = model.context(din, dout, dmask)
        logits = model(ctx, ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), ignore_index=PAD)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 100 == 0:
            rec = dict(kind="train", step=step, loss=ema, gn=float(gn), sec=time.time() - t0)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[{step}] loss {ema:.3f} gn {float(gn):.2f} ({(time.time()-t0)/max(step-step0,1):.2f}s/step)", flush=True)
        if step % a.eval_every == 0 and step > step0:
            model.eval(); hl = []
            with torch.no_grad():
                for i in range(0, min(len(hold_d), 256), a.bs):
                    b = hold_d[i : i + a.bs]
                    din, dout, dmask, ids = (t.to(dev) for t in collate(b, vocab, a.max_len))
                    lg = model(model.context(din, dout, dmask), ids)
                    hl.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids.reshape(-1), ignore_index=PAD).item())
            model.train()
            rec = dict(kind="eval", step=step, heldout_loss=float(np.mean(hl)))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[eval {step}] held-out loss {np.mean(hl):.3f}", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(a)}, ck)
            if np.mean(hl) < best_hl:                    # keep the best held-out checkpoint (loss overfits after ~80k)
                best_hl = float(np.mean(hl))
                torch.save({"model": model.state_dict(), "step": step, "args": vars(a), "heldout_loss": best_hl},
                           os.path.join(a.out, "best.pt"))
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": a.steps, "args": vars(a)}, ck)
    print("[done]", flush=True)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200000); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--d", type=int, default=512); ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=8); ap.add_argument("--n_ctx", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=8192); ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--n_heldout", type=int, default=2000); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None); ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = get_args()
    train(a)
