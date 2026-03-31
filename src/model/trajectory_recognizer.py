from typing import List, Optional, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Charset:
    def __init__(self, symbols: Optional[List[str]] = None, add_space: bool = True):
        if symbols is None:
            base = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            punct = list(".,;:!?-_'\"/\\()[]{}@#$%^&*+=|~`")
            symbols = base + punct
        if add_space and " " not in symbols:
            symbols = symbols + [" "]
        self.blank = 0
        self.sym2idx = {"<blank>": self.blank}
        self.idx2sym = {self.blank: "<blank>"}
        for i, s in enumerate(symbols, start=1):
            self.sym2idx[s] = i
            self.idx2sym[i] = s

    @property
    def vocab_size(self):
        return len(self.idx2sym)

    def encode(self, text: str) -> List[int]:
        return [self.sym2idx.get(ch, self.sym2idx.get(" ", 0)) for ch in text]

    def decode_greedy(self, indices: List[int]) -> str:
        prev, out = None, []
        for t in indices:
            if t != self.blank and t != prev:
                out.append(self.idx2sym.get(t, ""))
            prev = t
        return "".join(out)


try:
    from src.model.prev_seq_encoder import PositionalEncoding as CASHGPosEnc
except Exception:
    class CASHGPosEnc(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(1))
        def forward(self, x):
            return x + self.pe[:x.size(0)]


class Traj1DBackbone(nn.Module):
    def __init__(self, in_ch=5, base=128):
        super().__init__()
        def blk(cin, cout, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, s, p, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True)
            )
        self.net = nn.Sequential(
            blk(in_ch, base, 7, 1, 3),
            nn.MaxPool1d(2),
            blk(base, base, 3, 1, 1),
            blk(base, base*2, 3, 1, 1),
            nn.MaxPool1d(2),
            blk(base*2, base*2, 3, 1, 1),
            blk(base*2, base*4, 3, 1, 1),
            nn.MaxPool1d(2),
            blk(base*4, base*4, 3, 1, 1),
        )
        self.out_ch = base*4
    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        f = self.net(traj)
        return f.permute(2,0,1)


class TrajGRUHead(nn.Module):
    def __init__(self, input_size: int, hidden=256, layers=2, bidir=True, vocab=128, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden, num_layers=layers,
                          dropout=dropout, bidirectional=bidir)
        out = hidden * (2 if bidir else 1)
        self.fc = nn.Linear(out, vocab)
    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y)

class TrajTransformerHead(nn.Module):
    def __init__(self, input_size: int, d_model=256, nhead=4, layers=4, dim_ff=512, dropout=0.1, vocab=128):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_ff, dropout=dropout, batch_first=False)
        self.pe = CASHGPosEnc(d_model)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc = nn.Linear(d_model, vocab)
    def forward(self, x):
        z = self.pe(self.proj(x))
        return self.fc(self.enc(z))


class CTCLossWrapper(nn.Module):
    def __init__(self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    def forward(self, logits, targets, target_lengths):
        logp = F.log_softmax(logits, dim=-1)
        T, B, V = logp.shape
        flat_targets = torch.tensor([t for seq in targets for t in seq],
                                    dtype=torch.int32, device=logp.device)
        input_lengths = torch.full((B,), T, dtype=torch.int32, device=logp.device)
        return self.ctc(logp, flat_targets, input_lengths, target_lengths), logp

def content_score_from_logprobs(log_probs: torch.Tensor) -> torch.Tensor:

    probs = log_probs.exp()
    nb = probs[..., 1:]
    m, _ = nb.max(dim=-1)
    return m.mean(dim=1)


class TrajectoryRecognizer(nn.Module):
    def __init__(self, charset: Charset, arch="gru", **kwargs):
        super().__init__()
        self.charset = charset
        self.backbone = Traj1DBackbone(in_ch=5, base=kwargs.pop("base", 128))

        if arch.lower() == "gru":
            self.head = TrajGRUHead(
                self.backbone.out_ch,
                hidden=kwargs.pop("hidden", 256),
                layers=kwargs.pop("layers", 2),
                bidir=kwargs.pop("bidir", True),
                vocab=charset.vocab_size,
                dropout=kwargs.pop("dropout", 0.1),
            )

        elif arch.lower() in ("tr", "transformer"):
            self.head = TrajTransformerHead(
                self.backbone.out_ch,
                d_model=kwargs.pop("d_model", 256),
                nhead=kwargs.pop("nhead", 4),
                layers=kwargs.pop("layers", 4),
                dim_ff=kwargs.pop("dim_ff", 512),
                dropout=kwargs.pop("dropout", 0.1),
                vocab=charset.vocab_size,
            )

        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.crit = CTCLossWrapper(blank=self.charset.blank)

    @torch.no_grad()
    def freeze(self):
        for p in self.parameters(): p.requires_grad_(False)
        self.eval(); return self

    def forward(self, traj, targets=None, target_lengths=None):
        f = self.backbone(traj)
        logits = self.head(f)
        logits = logits.permute(1, 0, 2).contiguous()
        out = {"logits": logits}
        logp = F.log_softmax(logits, dim=-1)
        out["score"] = content_score_from_logprobs(logp)

        return out

    @torch.no_grad()
    def decode_greedy(self, logits_bt):
        pred = logits_bt.argmax(dim=-1)
        res = []
        for b in range(pred.size(0)):
            seq = pred[b].tolist()
            res.append(self.charset.decode_greedy(seq))
        return res
