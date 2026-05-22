"""
Pre-compute CLIP text embeddings for all LIBERO benchmark task descriptions.

Usage:
  python -m scripts.precompute_clip_embeddings \
      --benchmarks libero_90 libero_object libero_spatial libero_goal libero_10 \
      --out-dir data/clip_embeddings
"""

import argparse
import re
from pathlib import Path

import torch
from transformers import CLIPTextModel, CLIPTokenizer

CLIP_MODEL = "openai/clip-vit-base-patch32"
SCENE_PREFIX_RE = re.compile(r"^[A-Z]+_SCENE\d+_")


def task_name_to_description(name: str) -> str:
    """KITCHEN_SCENE3_turn_on_the_stove -> 'turn on the stove'"""
    desc = SCENE_PREFIX_RE.sub("", name)
    return desc.replace("_", " ")


@torch.no_grad()
def compute_embeddings(task_names: list, device: str = "cpu") -> dict:
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    model = CLIPTextModel.from_pretrained(CLIP_MODEL).to(device)
    model.eval()

    embeddings = {}
    for name in task_names:
        desc = task_name_to_description(name)
        inputs = tokenizer(desc, return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(device)
        output = model(**inputs)
        emb = output.pooler_output[0].cpu().float()  # (512,)
        embeddings[name] = emb
        print(f"  [{name[:60]}] → dim={emb.shape[0]}")

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+",
                        default=["libero_90", "libero_object", "libero_spatial",
                                 "libero_goal", "libero_10"])
    parser.add_argument("--out-dir", default="data/clip_embeddings")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero.benchmark import get_benchmark

    for bname in args.benchmarks:
        print(f"\n=== {bname} ===")
        benchmark = get_benchmark(bname)(task_order_index=0)
        task_names = benchmark.get_task_names()
        print(f"  {len(task_names)} tasks")

        embeddings = compute_embeddings(task_names, device=args.device)

        out_path = out_dir / f"{bname}.pt"
        torch.save(embeddings, str(out_path))
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
