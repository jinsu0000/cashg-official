import os
import pickle
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm


SAVE_IMG_SIZE = 64

def check_font_support(font, char):
    try:
        mask = font.getmask(char)
        return mask.getbbox() is not None
    except Exception:
        return False

def render_font_char_img(char, font, image_size=64):
    img = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)

    bbox = font.getbbox(char)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]


    x = -bbox[0]
    y = (image_size - h) // 2 - bbox[1] + 4
    draw.text((x, y), char, font=font, fill=0)
    return np.array(img)

def save_preview_images(content_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    font = ImageFont.load_default()
    for char, img_arr in content_dict.items():
        img_rgb = np.stack([img_arr] * 3, axis=-1)
        img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img)
        draw.text((4, 4), f"{char}", fill=(255, 0, 0), font=font)
        fname = f"{char}.png" if char.isalnum() else f"MARK_{ord(char)}.png"
        img.save(os.path.join(save_dir, fname))

def make_content_font_and_dict(pickle_root, font_path, save_path_img, save_path_dict, preview_dir=None, image_size=64):
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_dict), exist_ok=True)

    char_counts = defaultdict(int)
    for writer_file in tqdm(os.listdir(pickle_root), desc="Counting chars"):
        if not writer_file.endswith(".pkl"):
            continue
        if writer_file.endswith("_rep.pkl"):
            continue
        writer_path = os.path.join(pickle_root, writer_file)
        with open(writer_path, 'rb') as f:
            writer_data = pickle.load(f)
        for char, samples in writer_data.items():
            if isinstance(samples, list):
                char_counts[char] += len(samples)
            elif isinstance(samples, dict):

                char_counts[char] += 1

    char_list = sorted(char_counts.keys())
    print(f"\n=== Unique char count: {len(char_list)} ===")
    print("Char frequencies (top 20):")
    for char in list(char_counts.keys())[:20]:
        print(f"'{char}' : {char_counts[char]}")
    print("...")

    font = ImageFont.truetype(font_path, image_size-18)
    content_dict = {}
    valid_chars = []
    for char in tqdm(char_list, desc="Rendering font content images"):
        if not check_font_support(font, char):
            print(f"[WARN] Font does not support: '{char}' (U+{ord(char):04X})")
            continue
        try:
            img_arr = render_font_char_img(char, font, image_size=image_size)
            content_dict[char] = img_arr
            valid_chars.append(char)
        except Exception as e:
            print(f"Error rendering '{char}': {e}")

    with open(save_path_img, 'wb') as f:
        pickle.dump(content_dict, f)

    char_dict = {char: idx for idx, char in enumerate(sorted(valid_chars))}
    with open(save_path_dict, 'wb') as f:
        pickle.dump(char_dict, f)

    print(f"\n===== Font Content Generation Done =====")
    print(f"Saved {len(content_dict)} content font images to: {save_path_img}")
    print(f"Saved character_dict with {len(char_dict)} entries to: {save_path_dict}")

    if preview_dir:
        print(f"\nSaving preview images to: {preview_dir}")
        save_preview_images(content_dict, preview_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_root', type=str, required=True, help='Path to char_pickles/')
    parser.add_argument('--font_path', type=str, required=True, help='Path to .ttf/.otf font file')
    parser.add_argument('--save_img', type=str, default='content_font_img.pkl')
    parser.add_argument('--save_dict', type=str, default='character_dict.pkl')
    parser.add_argument('--preview_dir', type=str, default=None, help='Optional: directory to save visualized images')
    parser.add_argument('--img_size', type=int, default=64, help='Image size (default=64)')
    args = parser.parse_args()

    make_content_font_and_dict(args.pickle_root, args.font_path, args.save_img, args.save_dict, args.preview_dir, args.img_size)
