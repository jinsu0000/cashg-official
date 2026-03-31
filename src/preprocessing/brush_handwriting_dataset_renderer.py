from glob import glob
import os
import pickle
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import argparse


TRAJ_INDEX = {"X":0, "Y":1, "PEN_MOVE":2, "EOS":3, "EOC":4}
TRAJ_DIM = len(TRAJ_INDEX)


COLOR_MAP = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (255, 0, 255),
    4: (0, 200, 0),
    5: (0, 255, 255),
    6: (255, 255, 0),
    7: (128, 0, 128),
}


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)

def ndarray_gray_to_rgba(img_gray: np.ndarray) -> Image.Image:
    img_rgb = np.stack([img_gray]*3, axis=-1).astype(np.uint8)
    alpha = np.full(img_gray.shape, 255, dtype=np.uint8)
    img_rgba = np.concatenate([img_rgb, alpha[..., None]], axis=-1)
    return Image.fromarray(img_rgba, mode='RGBA')

def draw_point_by_penstate(traj: np.ndarray, draw: ImageDraw.ImageDraw):
    for pt in traj:
        move = 1 if pt[TRAJ_INDEX["PEN_MOVE"]] >= 0.5 else 0
        eos  = 1 if pt[TRAJ_INDEX["EOS"]]       >= 0.5 else 0
        eoc  = 1 if pt[TRAJ_INDEX["EOC"]]       >= 0.5 else 0
        key = (move) | (eos << 1) | (eoc << 2)
        r, g, b = COLOR_MAP.get(key, (0, 0, 0))
        x, y = pt[TRAJ_INDEX["X"]], pt[TRAJ_INDEX["Y"]]
        draw.point((x, y), fill=(r, g, b, 255))

def char_dir_name(ch: str) -> str:
    if ch == ' ':
        return "SPACE"
    if ch.isupper():
        return f"{ch}_U"
    elif ch.islower():
        return f"{ch}_L"
    elif ch.isdigit():
        return f"{ch}_D"
    else:
        return f"MARK_{ord(ch)}"

def safe_token(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ['-', '_', '.']:
            out.append(ch)
        elif ch == ' ':
            out.append("SPACE")
        else:
            out.append(f"U{ord(ch):04X}")
    return "".join(out)

def _tight_canvas_from_points(points: np.ndarray, margin: int = 4):
    if points.size == 0:
        img = Image.new("L", (2*margin+1, 2*margin+1), 255)
        def _tr(p): return (p[0] + margin, p[1] + margin)
        return img, _tr
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    w = int(np.ceil(mx[0] - mn[0])) + 2*margin + 1
    h = int(np.ceil(mx[1] - mn[1])) + 2*margin + 1
    w = max(w, 2*margin+1); h = max(h, 2*margin+1)
    img = Image.new("L", (w, h), 255)
    def _tr(p):
        return (int(round(p[0] - mn[0])) + margin,
                int(round(p[1] - mn[1])) + margin)
    return img, _tr


def save_character_image_from_sample(sample, save_path, draw_points=False):
    img_gray = sample.get('image', None)
    if img_gray is None:
        return

    img = ndarray_gray_to_rgba(img_gray)
    traj = sample.get('trajectory', None)

    if isinstance(traj, np.ndarray) and traj.shape[0] > 1:
        draw = ImageDraw.Draw(img)


        t = traj.copy()
        x_min = float(np.min(t[:, TRAJ_INDEX["X"]]))
        t[:, TRAJ_INDEX["X"]] = t[:, TRAJ_INDEX["X"]] - x_min

        x  = t[:, TRAJ_INDEX["X"]]
        y  = t[:, TRAJ_INDEX["Y"]]
        pm = (t[:, TRAJ_INDEX["PEN_MOVE"]] >= 0.5).astype(np.uint8)
        es = (t[:, TRAJ_INDEX["EOS"]]       >= 0.5).astype(np.uint8)

        for i in range(1, len(x)):
            if pm[i-1] and not es[i-1]:
                draw.line((x[i-1], y[i-1], x[i], y[i]), fill=(0, 0, 0, 255), width=2)

        if draw_points:
            draw_point_by_penstate(t, draw)

    ensure_dir(os.path.dirname(save_path))
    img.save(save_path)

def render_character_pickles(char_root: str, save_root: str, draw_points: bool=False) -> int:
    if not char_root or not os.path.isdir(char_root):
        return 0
    count = 0
    for writer_file in tqdm(sorted(os.listdir(char_root)), desc="Character Rendering"):
        if not writer_file.endswith("_rep.pkl"):
            continue
        writer_id = writer_file[:-8]
        with open(os.path.join(char_root, writer_file), "rb") as f:
            char_map = pickle.load(f)

        for ch, samples in char_map.items():
            if ch == ' ':
                continue
            if isinstance(samples, dict):
                samples = [samples]
            elif not isinstance(samples, (list, tuple)):
                continue

            cdir = char_dir_name(ch)
            for i, sample in enumerate(samples):
                conn = sample.get("connection","unknown")
                sid  = sample.get("sentence_id", f"unknown{i}")
                save_dir = os.path.join(save_root, "char", writer_id, cdir, conn)
                save_path = os.path.join(save_dir, f"{safe_token(sid)}_{i}.png")
                save_character_image_from_sample(sample, save_path, draw_points=draw_points)
                count += 1
    return count


def render_style_pickles(style_root: str, save_root: str, draw_points: bool=False) -> int:
    if not style_root or not os.path.isdir(style_root):
        return 0
    count = 0
    for writer_file in tqdm(sorted(os.listdir(style_root)), desc="Style Rendering"):
        if not writer_file.endswith(".pkl"):
            continue
        writer_id = writer_file[:-4]
        with open(os.path.join(style_root, writer_file), "rb") as f:
            char_map = pickle.load(f)
        for ch, samples in char_map.items():
            if ch == ' ':
                continue
            cdir = char_dir_name(ch)
            for i, sample in enumerate(samples):
                conn = sample.get("connection","unknown")
                sid  = sample.get("sentence_id", f"unknown{i}")
                save_dir = os.path.join(save_root, "style", writer_id, cdir, conn)
                save_path = os.path.join(save_dir, f"{safe_token(sid)}_{i}.png")
                save_character_image_from_sample(sample, save_path, draw_points=draw_points)
                count += 1
    return count


def render_sentence_pickles(
    sent_root: str, save_root: str, draw_points=False,
    img_size=(64,64), ligature_lines=True, margin: int = 4, line_width: int = 1
) -> int:
    if not sent_root or not os.path.isdir(sent_root):
        return 0
    ensure_dir(save_root)
    total = 0

    for writer_file in tqdm(sorted(os.listdir(sent_root)), desc="Sentence Rendering"):
        if not (writer_file.endswith(".pkl") and writer_file.endswith("_sent.pkl")):
            continue
        writer_id = writer_file[:-9]
        with open(os.path.join(sent_root, writer_file), "rb") as f:
            sent_obj = pickle.load(f)
        sentences = sent_obj.get("sentences", {})

        for sid, chars in sentences.items():
            chars_sorted = sorted(chars, key=lambda x: x.get('cursor', 0))

            pts_blocks, pm_blocks, eos_last = [], [], []
            labels = []
            for it in chars_sorted:
                ch = it.get("character","")
                labels.append(ch)
                if ch == ' ':
                    continue
                traj = it.get("trajectory", None)
                if not isinstance(traj, np.ndarray) or traj.shape[0] == 0:
                    continue
                pts_blocks.append(traj[:, [TRAJ_INDEX["X"], TRAJ_INDEX["Y"]]].astype(np.float32))
                pm_blocks.append(traj[:, TRAJ_INDEX["PEN_MOVE"]].astype(np.float32))
                eos_last.append(float(traj[-1, TRAJ_INDEX["EOS"]]))


            if not pts_blocks:
                img = Image.new("L", (16,16), 255)
                out_dir = os.path.join(save_root, "sent", writer_id)
                ensure_dir(out_dir)
                out_path = os.path.join(out_dir, f"{safe_token(writer_id)}_{safe_token(str(sid))}.png")
                img.save(out_path)
                total += 1
                continue

            pts_all = np.concatenate(pts_blocks, axis=0)
            pm_all  = np.concatenate(pm_blocks, axis=0)


            img_gray, to_px = _tight_canvas_from_points(pts_all, margin=margin)
            drw = ImageDraw.Draw(img_gray)


            offset = 0
            for block_pts, block_pm in zip(pts_blocks, pm_blocks):
                if len(block_pts) >= 2:
                    for i in range(1, len(block_pts)):
                        if block_pm[i-1] >= 0.5:
                            a = to_px(block_pts[i-1])
                            b = to_px(block_pts[i])
                            drw.line([a, b], fill=0, width=line_width)


            if ligature_lines:
                for i in range(1, len(pts_blocks)):
                    if eos_last[i-1] < 0.5:
                        a = to_px(pts_blocks[i-1][-1])
                        b = to_px(pts_blocks[i][0])
                        drw.line([a, b], fill=0, width=line_width)


            if draw_points:
                for block_pts, block_pm in zip(pts_blocks, pm_blocks):
                    t = np.zeros((len(block_pts), 5), dtype=np.float32)
                    t[:,0:2] = block_pts
                    t[:,2] = block_pm

                    tmp = t.copy()

                    tmp[:,0] = tmp[:,0]

                    for p in block_pts:
                        px = to_px(p)
                        img_gray.putpixel(px, 0)


            out_dir = os.path.join(save_root, "sent", writer_id)
            ensure_dir(out_dir)
            out_path = os.path.join(out_dir, f"{safe_token(writer_id)}_{safe_token(str(sid))}.png")
            img_gray.convert("RGB").save(out_path)
            total += 1

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render RGB images from CASHG pickles (char/sent/style)")
    parser.add_argument("--char_root", type=str, default=None, help="Directory of char pickles: {writer}.pkl or *_rep.pkl")
    parser.add_argument("--sent_root", type=str, default=None, help="Directory of sent pickles: {writer}_sent.pkl")
    parser.add_argument("--style_root", type=str, default=None, help="Directory of style pickles: {writer}.pkl")
    parser.add_argument("--save_root", type=str, required=True, help="Where to save rendered images")
    parser.add_argument("--img_size", type=int, nargs=2, default=[64, 64], help="Image size (H W) (char/style preview만 영향)")
    parser.add_argument("--draw_points", action="store_true", help="Draw per-point dots (debug)")
    parser.add_argument("--modes", nargs="+", default=["all"],
                        choices=["char","sentence","style","all"],
                        help="What to render")
    parser.add_argument("--ligature_lines", action="store_true",
                        help="Draw connecting lines between adjacent glyphs when prev EOS==0")

    args = parser.parse_args()
    H, W = tuple(args.img_size)

    char_count = style_count = sent_count = 0

    if "all" in args.modes or "char" in args.modes:
        char_count = render_character_pickles(args.char_root, args.save_root, draw_points=args.draw_points)
    if "all" in args.modes or "style" in args.modes:
        style_count = render_style_pickles(args.style_root, args.save_root, draw_points=args.draw_points)
    if "all" in args.modes or "sentence" in args.modes:
        sent_count = render_sentence_pickles(
            args.sent_root, args.save_root,
            draw_points=args.draw_points,
            img_size=(H, W),
            ligature_lines=args.ligature_lines
        )

    if char_count > 0:
        print(f"Rendered {char_count} character images.")
    if style_count > 0:
        print(f"Rendered {style_count} style character images.")
    if sent_count > 0:
        print(f"Rendered {sent_count} sentence images.")
