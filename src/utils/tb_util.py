from torchvision.utils import make_grid
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
from typing import Optional, List
import numpy as np
import torch
import cv2

from src.utils.logger import print_once, debug_coord_stats
from src.loss.pen_loss import get_pen_loss
from src.config.constants import TRAJ_INDEX_EXPANDED, PEN_CLASS


PM_FLAG = 1
PU_FLAG = 2
CURSIVE_EOC_FLAG = 4
EOC_FLAG = 8

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
FADED_COLOR = (200, 200, 200)

def _to_np(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.array(t)

def visualize_input_images_tb(tag, tb_writer, style_imgs, writer_ids, char_tokens, step, nrow=16):

    B, N2, C, H, W = style_imgs.shape
    imgs = style_imgs.reshape(B * N2, C, H, W).cpu()
    imgs_rgb = imgs.repeat(1, 3, 1, 1)

    labels = [
        f"w{writer_ids[idx // N2].item()}\n{char_tokens[idx]}"
        for idx in range(B * N2)
    ]


    colors = [(0, 0, 255)] * (N2 // 2) + [(255, 0, 0)] * (N2 - N2 // 2)
    colors = colors * B

    grid = make_grid(imgs_rgb, nrow=nrow, padding=4)
    grid_np = TF.to_pil_image(grid)

    draw = ImageDraw.Draw(grid_np)
    font = ImageFont.load_default()

    for idx, label in enumerate(labels):
        x = (idx % nrow) * (W + 4) + 2
        y = (idx // nrow) * (H + 4) + 2
        draw.text((x, y), label, font=font, fill=colors[idx])

    tb_writer.add_image(tag, TF.to_tensor(grid_np), step)


def render_trajectory_image(traj, image_size=64, char=None,
    fit_to_canvas=False, thickness=1, stop_at_eoc=False,
    shift_to_min=False, margin=2):
    img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)


    if traj is None or len(traj) == 0:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()

        if char is not None and char != "":
            if len(char) >= 2:
                label = f"{char[0]}{char[1]}(0/0)"
            else:
                label = f"{char}(0/0)"
        else:
            label = "0/0"
        draw.text((2, 2), label, fill=(255, 0, 0), font=font)
        return img

    T = traj.shape[0]
    traj = np.array(traj, dtype=np.float32)


    x = traj[:, TRAJ_INDEX_EXPANDED["X"]]
    y = traj[:, TRAJ_INDEX_EXPANDED["Y"]]
    if shift_to_min and traj is not None and len(traj) > 0:
        min_x, min_y = float(x.min()), float(y.min())
        x = x - min_x + margin
        y = y - min_y + margin
        x = np.clip(x, 0, image_size - 1)
        y = np.clip(y, 0, image_size - 1)


    pen_pm = traj[:, TRAJ_INDEX_EXPANDED["PM"]] > 0.5
    pen_pu = traj[:, TRAJ_INDEX_EXPANDED["PU"]] > 0.5
    pen_cursive_eoc = traj[:, TRAJ_INDEX_EXPANDED["CURSIVE_EOC"]] > 0.5
    pen_eoc = traj[:, TRAJ_INDEX_EXPANDED["EOC"]] > 0.5
    pen_state = (
        pen_pm.astype(np.uint8) * PM_FLAG +
        pen_pu.astype(np.uint8) * PU_FLAG +
        pen_cursive_eoc.astype(np.uint8) * CURSIVE_EOC_FLAG +
        pen_eoc.astype(np.uint8) * EOC_FLAG
    )


    if fit_to_canvas and not shift_to_min:
        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
        if max_x > min_x:
            x = (x - min_x) / (max_x - min_x) * (image_size - 1)
        if max_y > min_y:
            y = (y - min_y) / (max_y - min_y) * (image_size - 1)


    draw.ellipse((x[0]-1, y[0]-1, x[0]+1, y[0]+1), fill=(0, 0, 0), outline=(0, 0, 0))
    eoc_index = -1


    for i in range(0, len(x)):
        if pen_state[i] & (EOC_FLAG | CURSIVE_EOC_FLAG):
            eoc_index = i
    if eoc_index == -1:
        color = FADED_COLOR
    else:
        color = (0,0,0)


    for i in range(len(x) - 1):
        if pen_state[i] & PM_FLAG:
            draw.line((x[i], y[i], x[i+1], y[i+1]), fill=color, width=thickness)

        if eoc_index == i:
            if stop_at_eoc:
                break
            color = FADED_COLOR


    for i in range(len(x)):

        if pen_state[i] & PM_FLAG:
            pt_color = (0, 0, 255)
        elif pen_state[i] & PU_FLAG:
            pt_color = (255, 0, 0)
        elif pen_state[i] & CURSIVE_EOC_FLAG:
            pt_color = (0, 200, 0)
        elif pen_state[i] & EOC_FLAG:
            pt_color = (0, 100, 0)
        else:
            pt_color = (128, 128, 128)

        draw.ellipse((x[i]-1, y[i]-1, x[i]+1, y[i]+1), fill=pt_color, outline=pt_color)

        if stop_at_eoc and eoc_index == i:
            break


    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()


    if eoc_index == -1:
        eoc_str = "0"
        label_color = (255, 0, 0)
    else:
        eoc_str = str(eoc_index + 1)
        label_color = (0, 0, 0)

    if char is not None and char != "":

        if len(char) >= 2:
            label = f"{char[0]}{char[1]}({eoc_str}/{T})"
        else:
            label = f"{char}({eoc_str}/{T})"
    else:
        label = f"{eoc_str}/{T}"
    draw.text((2, 2), label, fill=label_color, font=font)
    return img


def render_char_panel(
    char_img: Optional[torch.Tensor],
    gt_coords,
    pred_coords,
    char,
    image_size: int = 64,
    gap: int = 5,
    infer_coords=None,
):

    if char_img is not None:
        font_img = TF.to_pil_image(char_img.squeeze(0).cpu()).resize((image_size, image_size))
        font_img = font_img.convert("RGB")
    else:
        font_img = Image.new("RGB", (image_size, image_size), (255, 255, 255))

    draw = ImageDraw.Draw(font_img)


    if char == " ":
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font_large = ImageFont.load_default()
        draw.text((image_size//2 - 10, image_size//2 - 12), "[ ]", fill=(128, 128, 128), font=font_large)

    draw.rectangle([0, 0, image_size - 1, image_size - 1], outline=(0, 0, 0), width=1)


    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    if char and len(char) >= 2:

        label = f"{char[0]}{char[1]}"
        draw.text((2, image_size - 16), label, fill=(255, 0, 0), font=font)
    elif char:

        draw.text((2, image_size - 16), char, fill=(255, 0, 0), font=font)


    gt_img    = render_trajectory_image(gt_coords,    image_size=image_size, char=char, thickness=2, stop_at_eoc=True, shift_to_min=True)
    pred_img  = render_trajectory_image(pred_coords,  image_size=image_size, char=char, thickness=2, stop_at_eoc=False, shift_to_min=True)
    infer_img = render_trajectory_image(infer_coords, image_size=image_size, char=char, thickness=2, stop_at_eoc=False, shift_to_min=True) \
                if infer_coords is not None else Image.new("RGB", (image_size, image_size), (255, 255, 255))


    total_height = image_size * 4 + gap * 3
    panel = Image.new("RGB", (image_size, total_height), (255, 255, 255))
    panel.paste(font_img, (0, 0))
    panel.paste(gt_img, (0, image_size + gap))
    panel.paste(pred_img, (0, (image_size + gap) * 2))
    panel.paste(infer_img, (0, (image_size + gap) * 3))
    return panel

def visualize_character_level(
    char_imgs: List[Optional[torch.Tensor]],
    gt_coords_list,
    pred_coords_list,
    step: int,
    tb_writer,
    chars: Optional[List[str]] = None,
    tag_prefix: str = "train_char",
    image_size: int = 64,
    char_gap: int = 10,
    infer_coords_list=None,
):
    if chars is None:
        chars = [None] * len(char_imgs)
    if infer_coords_list is None:
        infer_coords_list = [None] * len(char_imgs)

    panels = [
        render_char_panel(
            c_img, gt_coords, pred_coords, char,
            image_size=image_size, infer_coords=inf_coords
        )
        for c_img, gt_coords, pred_coords, inf_coords, char
        in zip(char_imgs, gt_coords_list, pred_coords_list, infer_coords_list, chars)
    ]

    total_width = sum([p.width for p in panels]) + char_gap * (len(panels) - 1) if panels else image_size
    total_height = panels[0].height if panels else image_size
    out_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    curr_x = 0
    for p in panels:
        out_img.paste(p, (curr_x, 0))
        curr_x += p.width + char_gap

    tb_writer.add_image(f"{tag_prefix}/char_panel", TF.to_tensor(out_img), step)

def _to_np_coords(seq):
    if seq is None:
        return np.zeros((0, 6), dtype=np.float32)
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().float().numpy()
    seq = np.asarray(seq)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    if seq.shape[-1] < 6:

        pad = np.zeros((seq.shape[0], 6 - seq.shape[-1]), dtype=seq.dtype)
        seq = np.concatenate([seq, pad], axis=-1)
    return seq.astype(np.float32)

def _render_sentence_canvas(infer_coords_list, pad=2, line_width=1, bg=(255,255,255), fg=(0,0,0), sentence_chars=None):
    if len(infer_coords_list) == 0:
        return Image.new("RGB", (100, 50), bg)


    def has_eoc_or_cursive(a):
        if a.size == 0:
            return False
        eoc = a[:, TRAJ_INDEX_EXPANDED["EOC"]] > 0.5
        cursive_eoc = a[:, TRAJ_INDEX_EXPANDED["CURSIVE_EOC"]] > 0.5
        return eoc.any() or cursive_eoc.any()


    converted_coords = []
    for idx, arr in enumerate(infer_coords_list):

        if sentence_chars is not None and idx < len(sentence_chars) and sentence_chars[idx] == " ":
            continue
        a = _to_np_coords(arr)
        if a.size > 0 and has_eoc_or_cursive(a):
            converted_coords.append(a)

    if len(converted_coords) == 0:
        return Image.new("RGB", (100, 50), bg)


    all_coords = np.vstack(converted_coords)
    xmin = all_coords[:, TRAJ_INDEX_EXPANDED["X"]].min()
    xmax = all_coords[:, TRAJ_INDEX_EXPANDED["X"]].max()
    ymin = all_coords[:, TRAJ_INDEX_EXPANDED["Y"]].min()
    ymax = all_coords[:, TRAJ_INDEX_EXPANDED["Y"]].max()


    coord_w = float(max(1e-6, xmax - xmin))
    coord_h = float(max(1e-6, ymax - ymin))

    MIN_HEIGHT_PX = 64
    MIN_WIDTH_PX = 64
    MAX_WIDTH_PX = 2048
    MAX_HEIGHT_PX = 512


    scale = float(MIN_HEIGHT_PX / coord_h)


    w_candidate = int(np.ceil(coord_w * scale)) + 2 * pad
    if w_candidate > MAX_WIDTH_PX:
        scale = float((MAX_WIDTH_PX - 2 * pad) / coord_w)


    W = int(np.ceil(coord_w * scale)) + 2 * pad
    H = int(np.ceil(coord_h * scale)) + 2 * pad
    W = int(np.clip(W, MIN_WIDTH_PX, MAX_WIDTH_PX))
    H = int(np.clip(H, MIN_HEIGHT_PX, MAX_HEIGHT_PX))

    img = Image.new("RGB", (W, H), bg)
    drw = ImageDraw.Draw(img)

    prev_last_xy = None
    prev_is_cursive_eoc = False


    for a in converted_coords:

        xs = (a[:, TRAJ_INDEX_EXPANDED["X"]] - xmin) * scale + pad
        ys = (a[:, TRAJ_INDEX_EXPANDED["Y"]] - ymin) * scale + pad
        pm = a[:, TRAJ_INDEX_EXPANDED["PM"]] > 0.5
        cursive_eoc = a[:, TRAJ_INDEX_EXPANDED["CURSIVE_EOC"]] > 0.5
        eoc = a[:, TRAJ_INDEX_EXPANDED["EOC"]] > 0.5


        should_connect = prev_last_xy is not None and prev_is_cursive_eoc and pm[0]

        if should_connect:
            x0, y0 = prev_last_xy
            x1, y1 = float(xs[0]), float(ys[0])
            drw.line([(x0, y0), (x1, y1)], fill=fg, width=line_width)


        for t in range(len(xs) - 1):
            if pm[t]:
                drw.line([(float(xs[t]), float(ys[t])), (float(xs[t+1]), float(ys[t+1]))],
                         fill=fg, width=line_width)


        prev_last_xy = (float(xs[-1]), float(ys[-1]))
        prev_is_cursive_eoc = bool(cursive_eoc[-1])

    return img


def visualize_sentence_level(
    char_imgs: List[Optional[torch.Tensor]],
    gt_coords_list,
    pred_coords_list,
    step: int,
    tb_writer,
    sentence_chars,
    tag_prefix: str = "train",
    image_size: int = 64,
    sentence_gap: int = 2,
    infer_coords_list=None,
):
    if infer_coords_list is None:
        infer_coords_list = [None] * len(char_imgs)


    num_chars = len(sentence_chars)
    char_imgs = char_imgs[:num_chars]
    gt_coords_list = gt_coords_list[:num_chars]
    pred_coords_list = pred_coords_list[:num_chars]
    infer_coords_list = infer_coords_list[:num_chars]

    panels = [
        render_char_panel(
            c_img, gt_coords, pred_coords, char,
            image_size=image_size, infer_coords=inf_coords
        )
        for c_img, gt_coords, pred_coords, inf_coords, char
        in zip(char_imgs, gt_coords_list, pred_coords_list, infer_coords_list, sentence_chars)
    ]

    total_width = sum([p.width for p in panels]) + sentence_gap * (len(panels) - 1) if panels else image_size
    total_height = panels[0].height if panels else image_size
    sentence_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    curr_x = 0
    for p in panels:
        sentence_img.paste(p, (curr_x, 0))
        curr_x += p.width + sentence_gap


    try:
        gt_canvas = _render_sentence_canvas(gt_coords_list, pad=2, line_width=1, sentence_chars=sentence_chars)
    except Exception:
        gt_canvas = Image.new("RGB", (total_width // 2, max(1, total_height//4)), (255,255,255))


    try:
        infer_canvas = _render_sentence_canvas(infer_coords_list, pad=2, line_width=1, sentence_chars=sentence_chars)
    except Exception:
        infer_canvas = Image.new("RGB", (total_width // 2, max(1, total_height//4)), (255,255,255))


    max_canvas_h = max(gt_canvas.height, infer_canvas.height)


    canvas_gap = 10
    canvas_w = (total_width - canvas_gap) // 2


    if gt_canvas.width > canvas_w:
        scale = canvas_w / max(1, gt_canvas.width)
        new_h = max(1, int(round(gt_canvas.height * scale)))
        gt_canvas = gt_canvas.resize((canvas_w, new_h), Image.BILINEAR)


    if infer_canvas.width > canvas_w:
        scale = canvas_w / max(1, infer_canvas.width)
        new_h = max(1, int(round(infer_canvas.height * scale)))
        infer_canvas = infer_canvas.resize((canvas_w, new_h), Image.BILINEAR)


    max_canvas_h = max(gt_canvas.height, infer_canvas.height)


    label_height = 20
    comparison_h = label_height + max_canvas_h
    comparison_w = total_width
    comparison = Image.new("RGB", (comparison_w, comparison_h), (255, 255, 255))


    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()


    draw.text((canvas_w // 2 - 15, 2), "GT", fill=(255, 0, 0), font=font)


    draw.text((canvas_w + canvas_gap + canvas_w // 2 - 40, 2), "Inference", fill=(0, 0, 255), font=font)


    gt_x = (canvas_w - gt_canvas.width) // 2
    comparison.paste(gt_canvas, (gt_x, label_height))


    infer_x = canvas_w + canvas_gap + (canvas_w - infer_canvas.width) // 2
    comparison.paste(infer_canvas, (infer_x, label_height))


    gap_y = max(8, sentence_gap * 3)
    combo_h = total_height + gap_y + comparison_h
    combo = Image.new("RGB", (total_width, combo_h), (255,255,255))
    combo.paste(sentence_img, (0, 0))
    combo.paste(comparison, (0, total_height + gap_y))


    tb_writer.add_image(f"{tag_prefix}/sentence_panel", TF.to_tensor(combo), step)


def visualize_style_images_tb(tb_writer, style_imgs, writer_ids, step, tag="Stage1/style_imgs", nrow=16):
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as TF
    from PIL import ImageDraw, ImageFont

    B, N, C, H, W = style_imgs.shape
    imgs = style_imgs.view(B * N, C, H, W).cpu()


    imgs_inverted = 1.0 - imgs
    imgs_rgb = imgs_inverted.repeat(1, 3, 1, 1)


    grid = make_grid(imgs_rgb, nrow=nrow, padding=2, normalize=False)
    grid_np = TF.to_pil_image(grid)


    draw = ImageDraw.Draw(grid_np)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for idx in range(B * N):
        b = idx // N
        n = idx % N
        img_sample = imgs[idx]


        mean_val = img_sample.mean().item()
        std_val = img_sample.std().item()


        x = (idx % nrow) * (W + 2) + 2
        y = (idx // nrow) * (H + 2) + 2


        color = (255, 0, 0) if n < N // 2 else (0, 0, 255)


        label = f"w{writer_ids[b].item()}\nm={mean_val:.2f}"
        if font:
            draw.text((x, y), label, font=font, fill=color)


    grid_tensor = TF.to_tensor(grid_np)
    tb_writer.add_image(tag, grid_tensor, step)


    mean_all = imgs.mean().item()
    std_all = imgs.std().item()
    zero_count = (imgs.view(B*N, -1).sum(dim=1) == 0).sum().item()
    print(f"[StyleVis][{tag}] it={step} B={B}, N={N}, mean={mean_all:.3f}, std={std_all:.3f}, zero_imgs={zero_count}/{B*N}", flush=True)
