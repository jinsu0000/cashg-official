#!/usr/bin/env python

import argparse
import os
import random
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from src.config.config_parser import load_config
from src.model.style_identifier import StyleIdentifier
from src.model.font_encoder import FontEncoder
from src.model.handwriting_generator import HandwritingGenerator
from src.model.full_model import FullModel
from src.data.handwriting_generator_dataset import SentenceGenDataset
from src.utils.train_util import load_checkpoint
from src.utils.logger import print_once, set_trace_log
from src.data.data_utils import (
    delta_to_abs_norm,
    denormalize_xy_abs_symmetric,
    denorm_img_tensor,
)
from src.utils.tb_util import render_trajectory_image


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(cli: str) -> torch.device:
    if cli == "cpu":
        return torch.device("cpu")
    if cli == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ids_to_chars(ids: torch.Tensor, idx2char: Dict[int, str]) -> List[str]:
    ids = ids.detach().cpu().tolist()
    return [idx2char.get(int(i), "?") for i in ids]


def stitch_horizontally(images: List[Image.Image], gap: int = 4, bg=(255, 255, 255)) -> Image.Image:
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    total_w = sum(widths) + gap * (len(images) - 1)
    H = max(heights) if heights else 0
    canvas = Image.new("RGB", (total_w, H), bg)
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def save_sentence_panels(out_dir: str, chars: List[str], font_imgs: torch.Tensor, pred_list: List[np.ndarray], img_size: int, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    panels = []
    for s, ch in enumerate(chars):
        cimg = font_imgs[s] if (font_imgs is not None and s < font_imgs.shape[0]) else None
        pred = pred_list[s] if s < len(pred_list) else None
        panel = render_trajectory_image(pred, image_size=img_size, char=ch, thickness=2)
        if cimg is not None:
            font_pil = to_pil_image(denorm_img_tensor(cimg).squeeze(0).cpu()).resize((img_size, img_size))
        else:
            font_pil = Image.new("RGB", (img_size, img_size), (255,255,255))
        blank_gt = Image.new("RGB", (img_size, img_size), (255,255,255))
        Htot = img_size*3 + 4*2
        panel_full = Image.new("RGB", (img_size, Htot), (255,255,255))
        panel_full.paste(font_pil, (0,0))
        panel_full.paste(blank_gt, (0,img_size+4))
        panel_full.paste(panel, (0,(img_size+4)*2))
        panel_full.save(os.path.join(out_dir, f"{stem}_{s:02d}_{ch}.png"))
        panels.append(panel_full)
    strip = stitch_horizontally(panels, gap=6)
    strip.save(os.path.join(out_dir, f"{stem}_strip.png"))


class HandwritingGenerationTester:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.device = pick_device(args.device)
        print_once(f"[Tester] Device: {self.device}")


        self.sent_val_ds = SentenceGenDataset(
            writer_pickles_root = cfg.ENV.SENT_DATASET_PATH,
            writer_pickles_root_for_style = cfg.ENV.STYLE_DATASET_PATH,
            content_font_img_pkl = cfg.ENV.FONT_DATASET_IMG_PATH,
            character_dict_pkl   = cfg.ENV.FONT_DATASET_DICT_PATH,
            writer_ids = None,
            num_style_refs = cfg.VALID.NUM_REFERENCE_IMGS,
            use_rep_pickle_root = cfg.ENV.CHAR_DATASET_PATH,
            img_size=(cfg.ENV.IMG_H, cfg.ENV.IMG_W),
        )
        W = getattr(cfg.ENV, 'DATALOADER_WORKERS', 0)
        self.val_loader = DataLoader(
            self.sent_val_ds, batch_size=1, shuffle=False,
            num_workers=W, pin_memory=(self.device.type=="cuda"),
            collate_fn=self.sent_val_ds.collate_fn,
        )
        print_once(f"[Tester] valid(sent)={len(self.sent_val_ds)}")


        style_identifier = StyleIdentifier(
            style_dim=cfg.MODEL.STYLE_DIM,
            encoder_type=cfg.MODEL.ENCODER_TYPE,
            img_size=(cfg.ENV.IMG_H, cfg.ENV.IMG_W),
            base_layers=cfg.MODEL.BASE_LAYERS,
            base_nhead=cfg.MODEL.BASE_NHEAD,
            head_layers=cfg.MODEL.HEAD_LAYERS,
            head_nhead=cfg.MODEL.HEAD_NHEAD,
            patch_size=cfg.MODEL.PATCH_SIZE,
        ).to(self.device)
        font_encoder = FontEncoder(
            d_model=cfg.MODEL.FONT_DIM,
            base_nhead=cfg.MODEL.BASE_NHEAD,
            head_layers=cfg.MODEL.FONT_HEAD_LAYERS,
        ).to(self.device)
        hw_generator = HandwritingGenerator(
            d_model=cfg.MODEL.HWGEN_DIM,
            nhead=cfg.MODEL.BASE_NHEAD,
            writer_layers=cfg.MODEL.HWGEN_WRITER_LAYERS,
            glyph_layers=cfg.MODEL.HWGEN_GLYPH_LAYERS,
        ).to(self.device)
        self.full_model = FullModel(
            style_identifier=style_identifier,
            font_encoder=font_encoder,
            handwriting_generator=hw_generator,
        ).to(self.device)


        if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            self.full_model = torch.nn.DataParallel(self.full_model)


        resume_path = args.checkpoint
        step, best = load_checkpoint(resume_path, self.full_model, self.device, optimizer=None, strict=False)
        print_once(f"[Tester] Loaded checkpoint from {resume_path} (step={step}, best={best})")


        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_root = os.path.join(args.out, ts)
        os.makedirs(self.out_root, exist_ok=True)
        self.max_len = args.max_len


    @torch.no_grad()
    def run(self, num_batches: int = 1):
        self.full_model.eval()
        n_done = 0
        for batch in self.val_loader:
            if n_done >= num_batches:
                break

            style_imgs   = batch['style_reference'].to(self.device)
            writer_ids_t = batch['style_labels'].to(self.device)
            font_ref     = batch['font_reference'].to(self.device)
            sent_ids     = batch['sentence_char_ids'][0]
            idx2char     = self.sent_val_ds.idx2char
            chars_str    = ids_to_chars(sent_ids, idx2char)


            char_list = [[{'character': ch} for ch in chars_str]]

            model_for_infer = self.full_model.module if isinstance(self.full_model, torch.nn.DataParallel) else self.full_model
            results = model_for_infer.inference(
                style_imgs=style_imgs,
                char_list=char_list,
                char_imgs=font_ref,
                writer_ids=writer_ids_t,
                global_iter=0,
                max_len=self.max_len,
                output_dim=123,
            )


            img_size = int(self.cfg.ENV.IMG_H)
            pred_list_px: List[np.ndarray] = []


            batch_out = results[0] if isinstance(results, list) else results


            if not (isinstance(batch_out, dict) and ('seq' in batch_out)):
                raise RuntimeError("FullModel.inference must return {'gmm': [...], 'seq': [...]}")

            pred_seqs = batch_out['seq']

            for seq_t in pred_seqs:

                try:
                    dx, dy = seq_t[:,0], seq_t[:,1]
                    pm = seq_t[:,2]
                    print_once(f"[dbg] mean|Δ|={torch.mean(torch.sqrt(dx*dx+dy*dy)).item():.4f} p_move_mean={pm.mean().item():.3f}")
                except Exception:
                    pass

                coords = seq_t.detach().cpu().numpy().astype(np.float32)
                coords = delta_to_abs_norm(coords)
                coords = denormalize_xy_abs_symmetric(coords, img_size)
                pred_list_px.append(coords)


            font_imgs = font_ref[0].detach().cpu()
            stem = f"sample{n_done:03d}"
            out_dir = os.path.join(self.out_root, stem)
            save_sentence_panels(out_dir, chars_str, font_imgs, pred_list_px, img_size, stem)
            print_once(f"[Tester] Saved panels to: {out_dir}")

            n_done += 1
        print_once(f"[Tester] DONE. Outputs under: {self.out_root}")


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out', type=str, default='./infer_out')
    parser.add_argument('--num', type=int, default=1, help='How many sentences to sample (B=1 each)')
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cuda','cpu'])
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)

    tester = HandwritingGenerationTester(cfg, args)
    tester.run(num_batches=args.num)


if __name__ == '__main__':
    main()
