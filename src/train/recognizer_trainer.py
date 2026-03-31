import os, glob, pickle, argparse, time
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model.trajectory_recognizer import Charset, TrajectoryRecognizer


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model

def save_checkpoint(path: str,
                    model: nn.Module,
                    charset_sym2idx: dict,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler = None,
                    epoch: int = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = unwrap_model(model)
    ckpt = {"model": m.state_dict(), "charset": charset_sym2idx}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    torch.save(ckpt, path)

def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer=None,
                    scheduler=None,
                    map_location="cuda",
                    load_optim_sched: bool = True):
    ckpt = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", 0)
    charset = ckpt.get("charset", None)
    if load_optim_sched:
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
    return epoch, charset


class SentenceTrajDataset(Dataset):
    def __init__(self, sent_pkl_root: str):
        self.paths = sorted(glob.glob(os.path.join(sent_pkl_root, "*_sent.pkl")))
        self.items = []
        self._index()
        syms = set(ch for _,_,lab in self.items for ch in lab)
        if " " not in syms: syms.add(" ")
        self.charset = Charset(symbols=sorted(list(syms)), add_space=False)

    def _index(self):
        for p in self.paths:
            data = pickle.load(open(p,"rb"))
            for sid, items in data.get("sentences", {}).items():
                label = "".join([it.get("character","") for it in items])
                if label:
                    self.items.append((p, sid, label))

    def __len__(self): return len(self.items)

    def _concat_sentence_traj(self, items):
        pts = []
        for it in items:
            if it.get("character","") == " ":
                continue
            tr = it.get("trajectory", None)
            if tr is None or len(tr)==0:
                continue
            ox = float(it.get("orig_min_x", 0.0))
            x = tr[:,0] + ox
            y = tr[:,1]
            pts.append(np.stack([x,y,tr[:,2],tr[:,3],tr[:,4]], axis=1))
        return np.concatenate(pts, axis=0).astype(np.float32) if pts else np.zeros((0,5),dtype=np.float32)

    def __getitem__(self, idx):
        p,sid,label = self.items[idx]
        items = pickle.load(open(p,"rb"))["sentences"][sid]
        traj = self._concat_sentence_traj(items)
        traj = torch.from_numpy(traj.T).float()
        return traj, label, os.path.basename(p), sid


def collate_batch(batch, charset: Charset):
    tr_list, labels, pkls, sids = zip(*batch)
    T_max = max(t.shape[1] for t in tr_list)
    B = len(tr_list)
    X = torch.zeros(B, 5, T_max)
    for i,t in enumerate(tr_list):
        X[i,:, :t.shape[1]] = t
    targets = [charset.encode(lab) for lab in labels]
    tlen = torch.IntTensor([len(t) for t in targets])
    return X, targets, tlen, list(labels), list(pkls), list(sids)


def _lev(a,b):
    m,n=len(a),len(b)
    dp=list(range(n+1))
    for i in range(1,m+1):
        prev,dp[0]=dp[0],i
        for j in range(1,n+1):
            cur=dp[j]
            cost=0 if a[i-1]==b[j-1] else 1
            dp[j]=min(dp[j]+1, dp[j-1]+1, prev+cost)
            prev=cur
    return dp[n]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_cer,n=0.0,0
    for X, targets, tlen, labels, _, _ in loader:
        X = X.to(device, non_blocking=True)
        logits_b = model(X)["logits"]
        pred_txt = unwrap_model(model).decode_greedy(logits_b)
        for gt, hyp in zip(labels, pred_txt):
            tot_cer += _lev(list(gt), list(hyp)) / max(1, len(gt))
            n += 1
    return {"CER": tot_cer/max(1,n)}


def train(args):
    device = args.device

    ds = SentenceTrajDataset(args.sent_pkl_root)
    charset = ds.charset

    n=len(ds); idx=np.arange(n); np.random.seed(0); np.random.shuffle(idx)
    split=int(n*(1-args.val_ratio)); tr_idx,va_idx=idx[:split],idx[split:]

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=True, persistent_workers=(args.num_workers>0))
    tr_loader=DataLoader(torch.utils.data.Subset(ds,tr_idx), shuffle=True,
                         collate_fn=lambda b: collate_batch(b,charset), **loader_kwargs)
    va_loader=DataLoader(torch.utils.data.Subset(ds,va_idx), shuffle=False,
                         collate_fn=lambda b: collate_batch(b,charset), **loader_kwargs)


    model=TrajectoryRecognizer(charset,arch=args.arch,base=args.base,hidden=args.hidden,
        layers=args.layers,bidir=not args.unidirectional,
        d_model=args.d_model,nhead=args.nhead,dim_ff=args.dim_ff).to(device)


    if torch.cuda.device_count() > 1 and "cuda" in device:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)


    opt=torch.optim.AdamW(unwrap_model(model).parameters(),lr=args.lr,weight_decay=args.wd)

    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,args.epochs))


    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] loading from {args.resume}")
        last_epoch, charset_from_ckpt = load_checkpoint(
            args.resume, model, opt, sched, map_location=device,
            load_optim_sched=(not args.reset_optim)
        )
        start_epoch = int(last_epoch) + 1
        print(f"[Resume] resumed at epoch {last_epoch}. Next epoch = {start_epoch}")

        if args.reset_optim:
            print("[Resume] reset optimizer & scheduler (fine-tune mode)")
            opt = torch.optim.AdamW(unwrap_model(model).parameters(),
                                    lr=args.lr, weight_decay=args.wd)
            T_max = (args.epochs_add or (args.epochs - last_epoch))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, T_max))


    if args.epochs_add is not None:
        end_epoch = start_epoch + args.epochs_add - 1
    else:
        end_epoch = args.epochs
    print(f"[Train] epoch range: {start_epoch}..{end_epoch}")

    os.makedirs(args.out_dir,exist_ok=True)
    writer=SummaryWriter(os.path.join(args.out_dir,"tb"))
    best=1e9
    crit = unwrap_model(model).crit


    n_train, n_val = len(tr_idx), len(va_idx)
    b_train = (n_train + args.batch_size - 1) // args.batch_size
    b_val   = (n_val   + args.batch_size - 1) // args.batch_size
    print(f"[INFO] train: {n_train} samples → {b_train} batches | val: {n_val} → {b_val}")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and ("cuda" in device))

    for epoch in range(start_epoch, end_epoch+1):
        model.train(); tot=0.0; t0=time.time()
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{end_epoch}", ncols=100)

        for X, targets, tlen, labels, _, _ in pbar:
            X = X.to(device, non_blocking=True)


            nonzero_X = (X.abs().sum(dim=(1,2)) > 0).detach().cpu()
            valid_len = (tlen > 0)
            valid_mask = (nonzero_X & valid_len)
            if not valid_mask.all().item():
                idxs = valid_mask.nonzero(as_tuple=False).squeeze(1).tolist()
                if len(idxs) == 0:
                    continue
                X = X[idxs]
                tlen = tlen[idxs]
                targets = [targets[i] for i in idxs]

            with torch.amp.autocast('cuda', enabled=args.amp and ("cuda" in device)):
                out = model(X)
                logits_b = out["logits"]
                logits = logits_b.permute(1, 0, 2).contiguous()
                logp = F.log_softmax(logits, dim=-1)
                T, B, V = logp.shape
                flat_targets = torch.tensor([t for seq in targets for t in seq],
                                            dtype=torch.int32, device=device)
                input_lengths = torch.full((B,), T, dtype=torch.int32, device=device)
                loss = crit.ctc(logp, flat_targets, input_lengths, tlen.to(device))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            tot += float(loss)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        sched.step()
        cer = evaluate(model, va_loader, device)["CER"]

        avg_loss = tot/max(1,len(tr_loader))
        print(f"[E{epoch}] train_loss={avg_loss:.4f} val_CER={cer:.4f} "
              f"lr={sched.get_last_lr()[0]:.2e} time={time.time()-t0:.1f}s")

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("val/cer", cer, epoch)

        save_checkpoint(os.path.join(args.out_dir,"last.pth"), model, charset.sym2idx,
                        optimizer=opt, scheduler=sched, epoch=epoch)
        if cer<best:
            best=cer
            save_checkpoint(os.path.join(args.out_dir,"best_cer.pth"), model, charset.sym2idx,
                            optimizer=opt, scheduler=sched, epoch=epoch)
            print(f"  ↳ New best CER {best:.4f}")


        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                for X, targets, tlen, labels, pkls, sids in va_loader:
                    X = X.to(device, non_blocking=True)
                    pred = unwrap_model(model).decode_greedy(model(X)["logits"])
                    for i in range(min(3, len(labels))):
                        print(f"[SAMPLE] GT: {labels[i][:60]} | PRED: {pred[i][:60]}")
                    break

    writer.close()


def load_recognizer_for_cashg(ckpt_path, device="cuda"):
    state=torch.load(ckpt_path,map_location=device)
    inv=sorted([(i,s) for s,i in state.get("charset",{}).items() if s!="<blank>"], key=lambda t:t[0])
    symbols=[s for _,s in inv if s!="<blank>"]
    charset=Charset(symbols=symbols,add_space=(" " in symbols))
    model=TrajectoryRecognizer(charset,arch="gru").to(device)
    model.load_state_dict(state["model"])
    return model.freeze().to(device)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sent_pkl_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--arch", type=str, default="gru", choices=["gru","tr"])
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--unidirectional", action="store_true")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--dim_ff", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint (e.g., best_cer.pth) to resume from")
    ap.add_argument("--reset_optim", action="store_true",
                    help="ignore optimizer/scheduler state in checkpoint (fine-tune)")
    ap.add_argument("--epochs_add", type=int, default=None,
                    help="train for N more epochs from the resumed epoch. If set, overrides --epochs as total.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="enable mixed-precision training")
    args=ap.parse_args()
    train(args)

if __name__=="__main__":
    main()
