"""Build JSONL manifests for every downloaded LibriSpeech split and train the tokenizer.

    uv run python -m mutter.prepare

Reads  data/LibriSpeech/<split>/<speaker>/<chapter>/*.trans.txt + *.flac
Writes data/manifests/<split>.jsonl and data/tokenizer/sp256.model
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from .data import manifest_path
from .features import num_frames
from .tokenizer import Tokenizer, train_tokenizer

SPLITS = ["train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"]


def scan_split(split_dir: Path) -> list[dict]:
    items = []
    for trans in sorted(split_dir.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt_id, text = line.split(" ", 1)
            items.append({"audio": str(trans.parent / f"{utt_id}.flac"), "text": text.lower()})

    def duration(item):
        info = sf.info(item["audio"])
        return round(info.frames / info.samplerate, 3)

    with ThreadPoolExecutor(8) as pool:
        for item, dur in zip(items, pool.map(duration, items)):
            item["duration"] = dur
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--tokenizer-split", default="train-clean-100",
                   help="split whose transcripts train the tokenizer")
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--subsample", type=int, default=8, help="model time reduction (for the CTC check)")
    args = p.parse_args()

    root = Path(args.data_dir) / "LibriSpeech"
    manifests = {}
    for split in SPLITS:
        if not (root / split / ".done").exists():
            print(f"[skip] {split}: not downloaded (run scripts/download_librispeech.sh)")
            continue
        items = scan_split(root / split)
        out = manifest_path(args.data_dir, split)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        hours = sum(i["duration"] for i in items) / 3600
        print(f"[manifest] {split}: {len(items):,} utterances, {hours:.1f} h -> {out}")
        manifests[split] = items

    if args.tokenizer_split not in manifests:
        print(f"[skip] tokenizer: {args.tokenizer_split} not prepared yet")
        return
    texts = [i["text"] for i in manifests[args.tokenizer_split]]
    model_path = train_tokenizer(texts, Path(args.data_dir) / "tokenizer" / f"sp{args.vocab_size}")
    tok = Tokenizer(model_path)
    print(f"[tokenizer] {len(tok) - 1} pieces + blank -> {model_path}")
    print(f"[tokenizer] example: {texts[0][:60]!r} -> "
          f"{[tok.sp.id_to_piece(i - 1) for i in tok.encode(texts[0][:60])]}")

    # CTC can only emit one token per output step, so each utterance needs at least as many
    # output steps (80 ms each) as tokens. Report how many training utterances violate that.
    items = manifests[args.tokenizer_split]
    too_fast = 0
    for item in items:
        steps = num_frames(int(item["duration"] * 16000)) // args.subsample
        if len(tok.encode(item["text"])) > steps:
            too_fast += 1
    print(f"[check] {too_fast} of {len(items):,} utterances have more tokens than output steps "
          f"({100 * too_fast / len(items):.2f}%; these are skipped by CTC's zero_infinity)")


if __name__ == "__main__":
    main()
