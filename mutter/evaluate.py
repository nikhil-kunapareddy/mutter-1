"""Word error rate of a checkpoint on LibriSpeech test sets.

    uv run python -m mutter.evaluate runs/conformer-s/best.pt
    uv run python -m mutter.evaluate runs/conformer-s/best.pt --splits dev-clean --show 10
"""
import argparse

from .data import SpeechDataset, make_loader, manifest_path, read_manifest
from .decode import evaluate, load_model
from .model import pick_device


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--splits", nargs="+", default=["test-clean", "test-other"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--show", type=int, default=5, help="print this many worst-matching examples")
    args = p.parse_args()

    device = pick_device()
    model, tokenizer = load_model(args.checkpoint, device)
    for split in args.splits:
        entries = read_manifest(manifest_path(args.data_dir, split))
        loader = make_loader(SpeechDataset(entries, tokenizer, train=False, n_mels=model.cfg.n_mels),
                             max_seconds=400, shuffle=False, num_workers=4)
        wer, pairs = evaluate(model, loader, tokenizer, device)
        print(f"{split}: WER {wer:.2%} on {len(entries):,} utterances")
        wrong = [(r, h) for r, h in pairs if r != h]
        for ref, hyp in sorted(wrong, key=lambda rh: -abs(len(rh[0]) - len(rh[1])))[:args.show]:
            print(f"  ref: {ref}\n  hyp: {hyp}")


if __name__ == "__main__":
    main()
