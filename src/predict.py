"""
CLI inference tool for Visual Product Search.

Usage:
  python -m src.predict --text "Black skinny jeans, zip fly, 79% cotton" --category denim --top-k 5
  python -m src.predict --image path/to/query.jpg --text "Blue shirt" --category shirts
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.search_engine import ProductSearchEngine


def print_results(response):
    print(f"\nSearch Results [{response.pipeline}]")
    print(f"Category: {response.query_category} | Candidates: {response.n_gallery_candidates} | Latency: {response.latency_ms:.1f}ms")
    print("-" * 80)
    for r in response.results:
        print(f"#{r.rank:2d}  {r.item_id}")
        print(f"     category={r.category}  color={r.color}")
        print(f"     score={r.combined_score:.4f}  (text={r.text_score:.4f}, color={r.color_score:.4f})")
        print(f"     {r.description[:100]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Visual Product Search — CLI inference")
    parser.add_argument("--text", required=True, help="Product text description")
    parser.add_argument("--image", default=None, help="Path to query image (optional)")
    parser.add_argument("--category", default=None,
                        choices=ProductSearchEngine.CATEGORIES + [None],
                        help="Product category filter")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--w-text", type=float, default=0.8,
                        help="Text weight in scoring (0-1). Default 0.8")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("Loading search engine...")
    engine = ProductSearchEngine(w_text=args.w_text, device=args.device)
    engine.load_gallery()
    print(f"Gallery loaded: {engine.gallery_size} products")

    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert("RGB")
        response = engine.search_by_image_and_text(
            img=img, description=args.text, category=args.category, k=args.top_k
        )
    else:
        response = engine.search_by_text(
            description=args.text, category=args.category, k=args.top_k
        )

    print_results(response)
    perf = engine.get_performance_summary()
    print(f"\nModel performance (test set): R@1={perf['best_r1']:.3f}  R@5={perf['best_r5']:.3f}")


if __name__ == "__main__":
    main()
