import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

FONT_PATH = "/home/work/CASHG_data/font/NotoSansCJKsc-Regular.otf"

class EmbeddingMonitor:

    def __init__(self, tb_writer, max_chars_for_projector: int = 200):
        self.tb = tb_writer
        self.max_chars_for_projector = max_chars_for_projector


        self.char_embedding_cache: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.max_cache_per_char = 10

    def compute_similarity_metrics(
        self,
        content_embs: torch.Tensor,
        seq_chars: torch.Tensor
    ) -> Dict[str, float]:
        B, S, _, D = content_embs.shape


        embs = content_embs.squeeze(2)
        chars = seq_chars.cpu()


        char_to_embs: Dict[int, List[torch.Tensor]] = defaultdict(list)

        for b in range(B):
            for s in range(S):
                char_code = int(chars[b, s].item())
                if char_code > 0:
                    char_to_embs[char_code].append(embs[b, s].detach())


        same_char_sims = []
        for char_code, emb_list in char_to_embs.items():
            if len(emb_list) >= 2:
                emb_stack = torch.stack(emb_list)
                emb_norm = F.normalize(emb_stack, dim=-1)
                sim_matrix = torch.mm(emb_norm, emb_norm.t())

                n = sim_matrix.size(0)
                if n > 1:
                    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
                    same_char_sims.extend(sim_matrix[mask].tolist())


        diff_char_sims = []
        char_codes = list(char_to_embs.keys())
        if len(char_codes) >= 2:

            num_pairs = min(100, len(char_codes) * (len(char_codes) - 1) // 2)
            for _ in range(num_pairs):
                i, j = np.random.choice(len(char_codes), 2, replace=False)
                char_i, char_j = char_codes[i], char_codes[j]
                emb_i = char_to_embs[char_i][0]
                emb_j = char_to_embs[char_j][0]
                sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).item()
                diff_char_sims.append(sim)


        same_sim = np.mean(same_char_sims) if same_char_sims else 0.0
        diff_sim = np.mean(diff_char_sims) if diff_char_sims else 0.0
        separation = same_sim / max(diff_sim, 1e-6)

        return {
            "same_char_sim": same_sim,
            "diff_char_sim": diff_sim,
            "separation_ratio": separation,
            "num_unique_chars": len(char_codes),
            "num_same_pairs": len(same_char_sims),
        }

    def log_content_embedding_quality(
        self,
        content_embs: torch.Tensor,
        seq_chars: torch.Tensor,
        global_iter: int,
        prefix: str = "content_emb"
    ):
        with torch.no_grad():
            metrics = self.compute_similarity_metrics(content_embs, seq_chars)


        self.tb.add_scalar(f"{prefix}/same_char_similarity", metrics["same_char_sim"], global_iter)
        self.tb.add_scalar(f"{prefix}/diff_char_similarity", metrics["diff_char_sim"], global_iter)
        self.tb.add_scalar(f"{prefix}/separation_ratio", metrics["separation_ratio"], global_iter)
        self.tb.add_scalar(f"{prefix}/num_unique_chars", metrics["num_unique_chars"], global_iter)

        return metrics

    def log_embedding_projector(
        self,
        content_embs: torch.Tensor,
        seq_chars: torch.Tensor,
        global_iter: int,
        tag: str = "content_embeddings"
    ):
        B, S, _, D = content_embs.shape

        embs = content_embs.squeeze(2)
        chars = seq_chars.cpu()


        all_embs = []
        all_labels = []

        for b in range(B):
            for s in range(S):
                char_code = int(chars[b, s].item())
                if char_code > 0:
                    all_embs.append(embs[b, s].detach().cpu())

                    try:
                        char_str = chr(char_code)
                    except:
                        char_str = f"U+{char_code:04X}"
                    all_labels.append(f"{char_str} ({char_code})")

        if len(all_embs) == 0:
            return


        if len(all_embs) > self.max_chars_for_projector:
            indices = np.random.choice(len(all_embs), self.max_chars_for_projector, replace=False)
            all_embs = [all_embs[i] for i in indices]
            all_labels = [all_labels[i] for i in indices]

        emb_tensor = torch.stack(all_embs)


        self.tb.add_embedding(
            emb_tensor,
            metadata=all_labels,
            global_step=global_iter,
            tag=tag
        )

    def log_category_analysis(
        self,
        content_embs: torch.Tensor,
        seq_chars: torch.Tensor,
        global_iter: int,
        prefix: str = "content_emb"
    ):
        B, S, _, D = content_embs.shape

        embs = content_embs.squeeze(2)
        chars = seq_chars.cpu()


        category_embs: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for b in range(B):
            for s in range(S):
                char_code = int(chars[b, s].item())
                if char_code > 0:
                    cat = self._get_char_category(char_code)
                    category_embs[cat].append(embs[b, s].detach())


        for cat, emb_list in category_embs.items():
            if len(emb_list) >= 2:
                emb_stack = torch.stack(emb_list)
                emb_norm = F.normalize(emb_stack, dim=-1)
                sim_matrix = torch.mm(emb_norm, emb_norm.t())

                n = sim_matrix.size(0)
                mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
                intra_sim = sim_matrix[mask].mean().item()

                self.tb.add_scalar(f"{prefix}/category_{cat}_sim", intra_sim, global_iter)
                self.tb.add_scalar(f"{prefix}/category_{cat}_count", len(emb_list), global_iter)

    def _get_char_category(self, char_code: int) -> str:
        if 0x0041 <= char_code <= 0x005A:
            return "eng_upper"
        elif 0x0061 <= char_code <= 0x007A:
            return "eng_lower"
        elif 0x0030 <= char_code <= 0x0039:
            return "digit"
        elif 0xAC00 <= char_code <= 0xD7A3:
            return "korean"
        elif 0x4E00 <= char_code <= 0x9FFF:
            return "chinese"
        elif 0x3040 <= char_code <= 0x309F:
            return "hiragana"
        elif 0x30A0 <= char_code <= 0x30FF:
            return "katakana"
        elif char_code == 0x20:
            return "space"
        else:
            return "other"


def compute_content_embedding_quality(
    content_embs: torch.Tensor,
    seq_chars: torch.Tensor
) -> Dict[str, float]:
    monitor = EmbeddingMonitor(None)
    return monitor.compute_similarity_metrics(content_embs, seq_chars)


def visualize_content_vs_context(
    content_embs: torch.Tensor,
    context_embs: torch.Tensor,
    seq_chars: torch.Tensor,
    tb_writer,
    global_iter: int,
    max_points: int = 200,
    tag: str = "emb_comparison"
):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from sklearn.manifold import TSNE
    from io import BytesIO
    from PIL import Image
    import torchvision.transforms as T

    font_prop = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None
    if font_prop: plt.rcParams['font.family'] = font_prop.get_name()

    B, S, _, D = content_embs.shape


    content_list = []
    context_list = []
    char_labels = []
    char_codes = []

    content_flat = content_embs.squeeze(2).detach().cpu()
    context_flat = context_embs.squeeze(2).detach().cpu()
    chars = seq_chars.cpu()

    for b in range(B):
        for s in range(S):
            code = int(chars[b, s].item())
            if code > 0:
                content_list.append(content_flat[b, s])
                context_list.append(context_flat[b, s])
                char_codes.append(code)
                try:
                    char_labels.append(chr(code))
                except:
                    char_labels.append(f"U+{code:04X}")

    if len(content_list) < 5:
        return None


    if len(content_list) > max_points:
        indices = np.random.choice(len(content_list), max_points, replace=False)
        content_list = [content_list[i] for i in indices]
        context_list = [context_list[i] for i in indices]
        char_labels = [char_labels[i] for i in indices]
        char_codes = [char_codes[i] for i in indices]

    content_arr = torch.stack(content_list).numpy()
    context_arr = torch.stack(context_list).numpy()


    combined = np.vstack([content_arr, context_arr])
    perplexity = min(30, len(content_arr) - 1)

    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=500)
        combined_2d = tsne.fit_transform(combined)
    except Exception as e:
        print(f"[EmbeddingMonitor] t-SNE failed: {e}")
        return None

    n = len(content_arr)
    content_2d = combined_2d[:n]
    context_2d = combined_2d[n:]


    unique_codes = list(set(char_codes))
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_codes))))
    code_to_color = {code: colors[i % len(colors)] for i, code in enumerate(unique_codes)}
    point_colors = [code_to_color[c] for c in char_codes]


    fig, axes = plt.subplots(1, 2, figsize=(16, 7))


    ax1 = axes[0]
    for i, (x, y) in enumerate(content_2d):
        ax1.scatter(x, y, c=[point_colors[i]], s=60, alpha=0.7)
        ax1.annotate(char_labels[i], (x, y), fontsize=6, alpha=0.8, fontproperties=font_prop)
    ax1.set_title(f"Content Embedding (낱자 단위)\n같은 글자 = 비슷한 위치?", fontsize=12)
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.grid(True, alpha=0.3)


    ax2 = axes[1]
    for i, (x, y) in enumerate(context_2d):
        ax2.scatter(x, y, c=[point_colors[i]], s=60, alpha=0.7)
        ax2.annotate(char_labels[i], (x, y), fontsize=6, alpha=0.8, fontproperties=font_prop)
    ax2.set_title(f"Context Embedding (문장 단위)\n같은 글자도 문맥에 따라 다른 위치?", fontsize=12)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.grid(True, alpha=0.3)


    monitor = EmbeddingMonitor(None)

    content_metrics = monitor.compute_similarity_metrics(
        torch.stack(content_list).unsqueeze(0).unsqueeze(2),
        torch.tensor([char_codes]).long()
    )
    context_metrics = monitor.compute_similarity_metrics(
        torch.stack(context_list).unsqueeze(0).unsqueeze(2),
        torch.tensor([char_codes]).long()
    )


    text = (
        f"Content: same={content_metrics['same_char_sim']:.3f}, "
        f"diff={content_metrics['diff_char_sim']:.3f}, "
        f"ratio={content_metrics['separation_ratio']:.2f}\n"
        f"Context: same={context_metrics['same_char_sim']:.3f}, "
        f"diff={context_metrics['diff_char_sim']:.3f}, "
        f"ratio={context_metrics['separation_ratio']:.2f}"
    )
    fig.suptitle(f"Iter {global_iter} | {len(char_codes)} chars | {len(unique_codes)} unique\n{text}",
                  fontsize=10, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])


    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_tensor = T.ToTensor()(img)

    plt.close(fig)


    if tb_writer is not None:
        tb_writer.add_image(tag, img_tensor, global_iter)

    return {
        "content_metrics": content_metrics,
        "context_metrics": context_metrics,
    }


def visualize_content_vs_context_v2(
    content_embs: torch.Tensor,
    context_embs: torch.Tensor,
    seq_chars: torch.Tensor,
    tb_writer,
    global_iter: int,
    max_points: int = 100,
    tag: str = "emb_comparison_v2"
):
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from io import BytesIO
    from PIL import Image
    import torchvision.transforms as T

    font_prop = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None

    B, S, _, D = content_embs.shape


    content_list = []
    context_list = []
    char_labels = []
    char_codes = []

    content_flat = content_embs.squeeze(2).detach().cpu()
    context_flat = context_embs.squeeze(2).detach().cpu()
    chars = seq_chars.cpu()

    for b in range(B):
        for s in range(S):
            code = int(chars[b, s].item())
            if code > 0:
                content_list.append(content_flat[b, s])
                context_list.append(context_flat[b, s])
                char_codes.append(code)
                try:
                    char_labels.append(chr(code))
                except:
                    char_labels.append(f"U+{code:04X}")

    if len(content_list) < 5:
        return None


    if len(content_list) > max_points:
        indices = np.random.choice(len(content_list), max_points, replace=False)
        content_list = [content_list[i] for i in indices]
        context_list = [context_list[i] for i in indices]
        char_labels = [char_labels[i] for i in indices]
        char_codes = [char_codes[i] for i in indices]

    content_arr = torch.stack(content_list)
    context_arr = torch.stack(context_list)
    N = len(content_list)


    sorted_indices = sorted(range(N), key=lambda i: (char_codes[i], i))
    content_sorted = content_arr[sorted_indices]
    context_sorted = context_arr[sorted_indices]
    labels_sorted = [char_labels[i] for i in sorted_indices]
    codes_sorted = [char_codes[i] for i in sorted_indices]


    content_norm = F.normalize(content_sorted, dim=-1)
    context_norm = F.normalize(context_sorted, dim=-1)

    content_sim = torch.mm(content_norm, content_norm.t()).numpy()
    context_sim = torch.mm(context_norm, context_norm.t()).numpy()


    content_context_sim = F.cosine_similarity(content_sorted, context_sorted, dim=-1).numpy()


    fig = plt.figure(figsize=(18, 12))


    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(content_sim, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax1.set_title('Content Pairwise Similarity\n(같은 글자끼리 블록 패턴 = 좋음)', fontsize=10)
    ax1.set_xlabel('Char Index (sorted)')
    ax1.set_ylabel('Char Index (sorted)')
    plt.colorbar(im1, ax=ax1, fraction=0.046)


    prev_code = codes_sorted[0]
    for i, code in enumerate(codes_sorted):
        if code != prev_code:
            ax1.axhline(y=i-0.5, color='white', linewidth=0.5)
            ax1.axvline(x=i-0.5, color='white', linewidth=0.5)
            prev_code = code


    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(context_sim, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax2.set_title('Context Pairwise Similarity\n(같은 글자도 문맥에 따라 다르면 블록 흐려짐)', fontsize=10)
    ax2.set_xlabel('Char Index (sorted)')
    ax2.set_ylabel('Char Index (sorted)')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    prev_code = codes_sorted[0]
    for i, code in enumerate(codes_sorted):
        if code != prev_code:
            ax2.axhline(y=i-0.5, color='white', linewidth=0.5)
            ax2.axvline(x=i-0.5, color='white', linewidth=0.5)
            prev_code = code


    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(content_context_sim, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=np.mean(content_context_sim), color='red', linestyle='--',
                label=f'Mean: {np.mean(content_context_sim):.3f}')
    ax3.set_title('Content vs Context Similarity\n(같은 위치의 Content-Context 유사도)', fontsize=10)
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.set_xlim(0, 1)


    ax4 = fig.add_subplot(2, 3, 4)
    same_char_sims_content = []
    diff_char_sims_content = []
    for i in range(N):
        for j in range(i+1, N):
            if codes_sorted[i] == codes_sorted[j]:
                same_char_sims_content.append(content_sim[i, j])
            else:
                diff_char_sims_content.append(content_sim[i, j])

    if same_char_sims_content:
        ax4.hist(same_char_sims_content, bins=30, alpha=0.7, color='green',
                 label=f'Same char (n={len(same_char_sims_content)}, μ={np.mean(same_char_sims_content):.3f})')
    if diff_char_sims_content:

        if len(diff_char_sims_content) > 5000:
            diff_char_sims_content = np.random.choice(diff_char_sims_content, 5000, replace=False).tolist()
        ax4.hist(diff_char_sims_content, bins=30, alpha=0.5, color='red',
                 label=f'Diff char (μ={np.mean(diff_char_sims_content):.3f})')
    ax4.set_title('Content: Same vs Different Char\n(두 분포가 멀리 떨어져야 좋음)', fontsize=10)
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_ylabel('Count')
    ax4.legend()
    ax4.set_xlim(0, 1)


    ax5 = fig.add_subplot(2, 3, 5)
    same_char_sims_context = []
    diff_char_sims_context = []
    for i in range(N):
        for j in range(i+1, N):
            if codes_sorted[i] == codes_sorted[j]:
                same_char_sims_context.append(context_sim[i, j])
            else:
                diff_char_sims_context.append(context_sim[i, j])

    if same_char_sims_context:
        ax5.hist(same_char_sims_context, bins=30, alpha=0.7, color='green',
                 label=f'Same char (n={len(same_char_sims_context)}, μ={np.mean(same_char_sims_context):.3f})')
    if diff_char_sims_context:
        if len(diff_char_sims_context) > 5000:
            diff_char_sims_context = np.random.choice(diff_char_sims_context, 5000, replace=False).tolist()
        ax5.hist(diff_char_sims_context, bins=30, alpha=0.5, color='red',
                 label=f'Diff char (μ={np.mean(diff_char_sims_context):.3f})')
    ax5.set_title('Context: Same vs Different Char\n(Context는 같은 글자도 더 넓게 퍼져야 함)', fontsize=10)
    ax5.set_xlabel('Cosine Similarity')
    ax5.set_ylabel('Count')
    ax5.legend()
    ax5.set_xlim(0, 1)


    ax6 = fig.add_subplot(2, 3, 6)
    try:
        from umap import UMAP
        combined = np.vstack([content_sorted.numpy(), context_sorted.numpy()])
        umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        combined_2d = umap.fit_transform(combined)

        content_2d = combined_2d[:N]
        context_2d = combined_2d[N:]


        unique_codes = list(set(codes_sorted))
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_codes))))
        code_to_color = {code: colors[i % len(colors)] for i, code in enumerate(unique_codes)}

        for i in range(N):
            c = code_to_color[codes_sorted[i]]
            ax6.scatter(content_2d[i, 0], content_2d[i, 1], c=[c], marker='o', s=40, alpha=0.6)
            ax6.scatter(context_2d[i, 0], context_2d[i, 1], c=[c], marker='^', s=40, alpha=0.6)

        ax6.set_title('UMAP: Content(●) vs Context(▲)\n(같은 색 = 같은 글자)', fontsize=10)
        ax6.set_xlabel('UMAP 1')
        ax6.set_ylabel('UMAP 2')
    except ImportError:
        ax6.text(0.5, 0.5, 'UMAP not installed\npip install umap-learn',
                 ha='center', va='center', fontsize=12, transform=ax6.transAxes)
        ax6.set_title('UMAP (not available)', fontsize=10)


    unique_chars = len(set(char_codes))
    content_same_mean = np.mean(same_char_sims_content) if same_char_sims_content else 0
    content_diff_mean = np.mean(diff_char_sims_content) if diff_char_sims_content else 0
    context_same_mean = np.mean(same_char_sims_context) if same_char_sims_context else 0
    context_diff_mean = np.mean(diff_char_sims_context) if diff_char_sims_context else 0
    cc_mean = np.mean(content_context_sim)

    fig.suptitle(
        f'Iter {global_iter} | {N} samples | {unique_chars} unique chars\n'
        f'Content: same={content_same_mean:.3f}, diff={content_diff_mean:.3f}, gap={content_same_mean-content_diff_mean:.3f}\n'
        f'Context: same={context_same_mean:.3f}, diff={context_diff_mean:.3f}, gap={context_same_mean-context_diff_mean:.3f}\n'
        f'Content-Context sim: {cc_mean:.3f}',
        fontsize=11, y=1.02
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])


    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_tensor = T.ToTensor()(img)
    plt.close(fig)


    if tb_writer is not None:
        tb_writer.add_image(tag, img_tensor, global_iter)

    return {
        "content_same": content_same_mean,
        "content_diff": content_diff_mean,
        "context_same": context_same_mean,
        "context_diff": context_diff_mean,
        "content_context_sim": cc_mean,
    }


def visualize_content_vs_context_v3(
    content_embs: torch.Tensor,
    context_embs: torch.Tensor,
    seq_chars: torch.Tensor,
    tb_writer,
    global_iter: int,
    sentences: list = None,
    max_points: int = 100,
    tag: str = "emb_comparison_v3"
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from io import BytesIO
    from PIL import Image
    import torchvision.transforms as T

    font_prop = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None

    B, S, _, D = content_embs.shape


    content_list = []
    context_list = []
    char_labels = []
    char_codes = []

    content_flat = content_embs.squeeze(2).detach().cpu()
    context_flat = context_embs.squeeze(2).detach().cpu()
    chars = seq_chars.cpu()

    for b in range(B):
        for s in range(S):
            code = int(chars[b, s].item())
            if code > 0:
                content_list.append(content_flat[b, s])
                context_list.append(context_flat[b, s])
                char_codes.append(code)
                try:
                    char_labels.append(chr(code))
                except:
                    char_labels.append(f"U+{code:04X}")

    if len(content_list) < 5:
        return None


    N_original = len(content_list)
    if len(content_list) > max_points:
        indices = np.random.choice(len(content_list), max_points, replace=False)
        content_list = [content_list[i] for i in indices]
        context_list = [context_list[i] for i in indices]
        char_labels = [char_labels[i] for i in indices]
        char_codes = [char_codes[i] for i in indices]

    N = len(content_list)
    unique_codes = sorted(set(char_codes))


    sorted_indices = sorted(range(N), key=lambda i: (char_codes[i], i))
    content_sorted = torch.stack([content_list[i] for i in sorted_indices])
    context_sorted = torch.stack([context_list[i] for i in sorted_indices])
    labels_sorted = [char_labels[i] for i in sorted_indices]
    codes_sorted = [char_codes[i] for i in sorted_indices]


    content_norm = F.normalize(content_sorted, dim=-1)
    context_norm = F.normalize(context_sorted, dim=-1)
    content_sim = torch.mm(content_norm, content_norm.t()).numpy()
    context_sim = torch.mm(context_norm, context_norm.t()).numpy()


    content_unique = []
    content_unique_labels = []
    content_unique_codes = []
    seen_embeddings = {}


    DISTANCE_THRESHOLD = 1e-5

    for i, code in enumerate(char_codes):
        emb = content_list[i]
        label = chr(code) if code != 32 else 'sp'

        if code not in seen_embeddings:

            seen_embeddings[code] = [(emb, i)]
            content_unique.append(emb)
            content_unique_labels.append(label)
            content_unique_codes.append(code)
        else:

            is_duplicate = False
            for existing_emb, _ in seen_embeddings[code]:
                dist = (emb - existing_emb).norm().item()
                if dist < DISTANCE_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:

                seen_embeddings[code].append((emb, i))
                content_unique.append(emb)
                content_unique_labels.append(f"{label}*")
                content_unique_codes.append(code)

    content_unique = torch.stack(content_unique)
    n_unique_emb = len(content_unique)
    n_unique_char = len(unique_codes)


    if n_unique_emb > n_unique_char:
        print(f"[EmbeddingMonitor]  Warning: {n_unique_emb - n_unique_char} duplicate chars with different embeddings!")


    same_content, diff_content = [], []
    same_context, diff_context = [], []
    for i in range(N):
        for j in range(i+1, N):
            if codes_sorted[i] == codes_sorted[j]:
                same_content.append(content_sim[i, j])
                same_context.append(context_sim[i, j])
            else:
                diff_content.append(content_sim[i, j])
                diff_context.append(context_sim[i, j])

    content_same_mean = np.mean(same_content) if same_content else np.nan
    content_diff_mean = np.mean(diff_content) if diff_content else 0
    context_same_mean = np.mean(same_context) if same_context else np.nan
    context_diff_mean = np.mean(diff_context) if diff_context else 0
    content_gap = (content_same_mean - content_diff_mean) if same_content else np.nan
    context_gap = (context_same_mean - context_diff_mean) if same_context else np.nan

    def _fmt_metric(v: float) -> str:
        return "N/A" if np.isnan(v) else f"{v:.3f}"


    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_codes))))
    code_to_color = {code: colors[i % len(colors)] for i, code in enumerate(unique_codes)}

    fig = plt.figure(figsize=(18, 12))


    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(content_sim, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax1.set_title('Content Pairwise Similarity\n(Same char = red block)', fontsize=11)
    plt.colorbar(im1, ax=ax1, fraction=0.046)


    if N <= 60:
        ax1.set_xticks(range(N))
        ax1.set_yticks(range(N))
        ax1.set_xticklabels(labels_sorted, fontsize=6, rotation=90, fontproperties=font_prop)
        ax1.set_yticklabels(labels_sorted, fontsize=6, fontproperties=font_prop)
    ax1.set_xlabel('Character', fontsize=9)
    ax1.set_ylabel('Character', fontsize=9)


    prev_code = codes_sorted[0]
    for i, code in enumerate(codes_sorted):
        if code != prev_code:
            ax1.axhline(y=i-0.5, color='white', linewidth=1)
            ax1.axvline(x=i-0.5, color='white', linewidth=1)
            prev_code = code


    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(context_sim, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax2.set_title('Context Pairwise Similarity\n(Blocks blur = context varies)', fontsize=11)
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    if N <= 60:
        ax2.set_xticks(range(N))
        ax2.set_yticks(range(N))
        ax2.set_xticklabels(labels_sorted, fontsize=6, rotation=90, fontproperties=font_prop)
        ax2.set_yticklabels(labels_sorted, fontsize=6, fontproperties=font_prop)
    ax2.set_xlabel('Character', fontsize=9)
    ax2.set_ylabel('Character', fontsize=9)

    prev_code = codes_sorted[0]
    for i, code in enumerate(codes_sorted):
        if code != prev_code:
            ax2.axhline(y=i-0.5, color='white', linewidth=1)
            ax2.axvline(x=i-0.5, color='white', linewidth=1)
            prev_code = code


    try:
        from umap import UMAP


        ax3 = fig.add_subplot(2, 2, 3)
        umap_content = UMAP(n_components=2, n_neighbors=min(5, n_unique_emb-1),
                            min_dist=0.1, random_state=42)
        content_2d = umap_content.fit_transform(content_unique.numpy())

        for i in range(n_unique_emb):
            code = content_unique_codes[i]
            c = code_to_color[code]

            is_duplicate_diff = '*' in content_unique_labels[i]
            edgecolor = 'red' if is_duplicate_diff else 'black'
            linewidth = 2 if is_duplicate_diff else 0.5
            ax3.scatter(content_2d[i, 0], content_2d[i, 1], c=[c], marker='o',
                        s=150, alpha=0.9, edgecolors=edgecolor, linewidth=linewidth)
            ax3.annotate(content_unique_labels[i], (content_2d[i, 0], content_2d[i, 1]),
                        fontsize=10, ha='center', va='bottom', fontweight='bold',
                        color='red' if is_duplicate_diff else 'black', fontproperties=font_prop)


        warning_text = f" {n_unique_emb - n_unique_char} diff emb!" if n_unique_emb > n_unique_char else ""
        ax3.set_title(f'CONTENT UMAP\n{n_unique_char} chars, {n_unique_emb} unique emb{warning_text}', fontsize=11)
        ax3.set_xlabel('UMAP Dim 1')
        ax3.set_ylabel('UMAP Dim 2')
        ax3.grid(True, alpha=0.3)


        ax4 = fig.add_subplot(2, 2, 4)
        context_all = torch.stack(context_list)
        umap_context = UMAP(n_components=2, n_neighbors=min(15, N-1),
                            min_dist=0.1, random_state=42)
        context_2d = umap_context.fit_transform(context_all.numpy())

        for i in range(N):
            code = char_codes[i]
            c = code_to_color[code]
            ax4.scatter(context_2d[i, 0], context_2d[i, 1], c=[c], marker='^',
                        s=80, alpha=0.8, edgecolors='black', linewidth=0.3)
            ax4.annotate(char_labels[i], (context_2d[i, 0], context_2d[i, 1]),
                        fontsize=7, ha='center', va='bottom', fontproperties=font_prop)

        ax4.set_title(f'CONTEXT UMAP\n{N} chars (same char spreads)', fontsize=11)
        ax4.set_xlabel('UMAP Dim 1')
        ax4.set_ylabel('UMAP Dim 2')
        ax4.grid(True, alpha=0.3)

    except ImportError:
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.text(0.5, 0.5, 'UMAP not installed\npip install umap-learn',
                 ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.text(0.5, 0.5, 'UMAP not installed',
                 ha='center', va='center', fontsize=12, transform=ax4.transAxes)


    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=code_to_color[code],
                            label=chr(code) if code != 32 else 'space')
                       for code in unique_codes[:20]]
    leg = fig.legend(handles=legend_elements, loc='center right', ncol=1, fontsize=8,
                       title='Chars', bbox_to_anchor=(0.99, 0.5))
    if font_prop:
        for t in leg.get_texts(): t.set_fontproperties(font_prop)


    if sentences:
        sent_str = str(sentences[:3])
        if len(sentences) > 3:
            sent_str = sent_str[:-1] + ", ...]"
    else:
        sent_str = f"{B} sentences"

    fig.suptitle(
        f'Embedding Analysis | Iter {global_iter}\n'
        f'Input: {sent_str}\n'
        f'{N_original} chars | {len(unique_codes)} unique\n'
        f'Content: same={_fmt_metric(content_same_mean)}, diff={_fmt_metric(content_diff_mean)}, GAP={_fmt_metric(content_gap)} | '
        f'Context: same={_fmt_metric(context_same_mean)}, diff={_fmt_metric(context_diff_mean)}, GAP={_fmt_metric(context_gap)}',
        fontsize=10, y=0.99
    )

    plt.tight_layout(rect=[0, 0, 0.92, 0.91])


    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_tensor = T.ToTensor()(img)
    plt.close(fig)


    if tb_writer is not None:
        tb_writer.add_image(tag, img_tensor, global_iter)

    return {
        "content_same": content_same_mean,
        "content_diff": content_diff_mean,
        "content_gap": content_gap,
        "context_same": context_same_mean,
        "context_diff": context_diff_mean,
        "context_gap": context_gap,
    }
