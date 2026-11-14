"""Train the Conformer-CTC model.

    uv run python -m mutter.train                       # full run (stops after --max-hours 6)
    uv run python -m mutter.train                       # same command again = resume
    uv run python -m mutter.train --overfit-steps 300   # pipeline check: memorize one batch

Every field in config.py is a flag: `--d-model 256`, `--max-hours 8`, `--no-resume`, ...
Outputs in --run-dir: last.pt (resume point), best.pt (lowest dev WER), tokenizer.model,
metrics.jsonl (one line per epoch), train.log.
"""
import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig
from .data import SpeechDataset, collate, make_loader, manifest_path, read_manifest
from .decode import evaluate, greedy_decode
from .model import ConformerCTC, count_params, pick_device
from .tokenizer import BLANK, Tokenizer


def parse_args() -> tuple[TrainConfig, ModelConfig]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for cls in (TrainConfig, ModelConfig):
        for f in fields(cls):
            flag = "--" + f.name.replace("_", "-")
            if isinstance(f.default, bool):
                p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
            elif isinstance(f.default, tuple):
                p.add_argument(flag, nargs="+", default=f.default)
            else:
                p.add_argument(flag, type=type(f.default), default=f.default)
    args = vars(p.parse_args())
    pick = lambda cls: cls(**{f.name: args[f.name] for f in fields(cls)})
    return pick(TrainConfig), pick(ModelConfig)


class Logger:
    def __init__(self, path: Path):
        self.file = open(path, "a")

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()


def ctc_loss(logits, out_lens, targets, target_lens):
    # CTC isn't implemented on Apple's GPU backend; computing it on the CPU costs ~10 ms/batch.
    log_probs = logits.float().log_softmax(-1).transpose(0, 1).cpu()     # (T, B, V)
    return F.ctc_loss(log_probs, targets, out_lens.cpu(), target_lens,
                      blank=BLANK, reduction="mean", zero_infinity=True)


def noam(step: int, warmup: int) -> float:
    """LR multiplier: linear warmup to 1.0, then decay proportional to 1/sqrt(step)."""
    step = max(step, 1)
    return min(step / warmup, math.sqrt(warmup / step))


def save(path: Path, model, opt, sched, epoch: int, step: int, best_wer: float,
         tcfg: TrainConfig, mcfg: ModelConfig) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "epoch": epoch, "step": step, "best_wer": best_wer,
        "model_config": asdict(mcfg), "train_config": asdict(tcfg),
    }, tmp)
    tmp.replace(path)   # atomic: a crash mid-save never corrupts the previous checkpoint


def overfit_check(model, loader, tokenizer, device, steps: int, log) -> None:
    """If the pipeline is correct, a model can memorize a single batch: loss -> ~0 and the
    decoded text matches the transcript. If this fails, full training will fail too."""
    batches = loader.batch_sampler.batches
    indices = batches[len(batches) // 2]          # a typical-length batch, not the shortest
    feats, feat_lens, targets, target_lens, texts = collate([loader.dataset[i] for i in indices])
    feats, feat_lens = feats.to(device), feat_lens.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))
    log(f"overfit check: {len(texts)} utterances, {steps} steps")
    for step in range(1, steps + 1):
        logits, out_lens = model(feats, feat_lens)
        loss = ctc_loss(logits, out_lens, targets, target_lens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                hyp = greedy_decode(*model(feats[:1], feat_lens[:1]), tokenizer)[0]
            model.train()
            log(f"  step {step:4d}  loss {loss.item():.3f}  hyp: {hyp[:70]!r}")
    log(f"  ref:                   {texts[0][:70]!r}")


def main():
    tcfg, mcfg = parse_args()
    run_dir = Path(tcfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "train.log")
    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    device = pick_device()

    # The run dir keeps its own tokenizer copy so checkpoints stay self-contained. A resumed run
    # keeps the copy it started with, even if data/tokenizer was retrained in the meantime.
    last = run_dir / "last.pt"
    if last.exists() and not tcfg.resume and not tcfg.overfit_steps:
        raise SystemExit(f"{last} exists. Resume (drop --no-resume) or pick a new --run-dir.")
    run_tokenizer = run_dir / "tokenizer.model"
    if not last.exists() or not run_tokenizer.exists():
        shutil.copy(tcfg.tokenizer, run_tokenizer)
    tokenizer = Tokenizer(run_tokenizer)
    mcfg.vocab_size = len(tokenizer)

    # ---- resume -------------------------------------------------------------------------
    state = None
    if last.exists() and not tcfg.overfit_steps:
        state = torch.load(last, map_location="cpu", weights_only=False)
        mcfg = ModelConfig(**state["model_config"])   # architecture must match the checkpoint

    model = ConformerCTC(mcfg).to(device)
    log(f"device {device} | Conformer-CTC {count_params(model) / 1e6:.1f}M params | {asdict(mcfg)}")

    # ---- data ---------------------------------------------------------------------------
    dev = read_manifest(manifest_path(tcfg.data_dir, tcfg.dev_split), limit=tcfg.dev_limit)
    dev_loader = make_loader(SpeechDataset(dev, tokenizer, train=False, n_mels=mcfg.n_mels),
                             tcfg.max_batch_seconds, shuffle=False, num_workers=tcfg.num_workers)

    if tcfg.overfit_steps:
        overfit_check(model, dev_loader, tokenizer, device, tcfg.overfit_steps, log)
        return

    train = [e for split in tcfg.train_splits
             for e in read_manifest(manifest_path(tcfg.data_dir, split), tcfg.max_utt_seconds)]
    train_set = SpeechDataset(train, tokenizer, train=True, n_mels=mcfg.n_mels)
    train_loader = make_loader(train_set, tcfg.max_batch_seconds, shuffle=True,
                               num_workers=tcfg.num_workers, seed=tcfg.seed)
    train_hours = sum(train_set.durations) / 3600
    log(f"train {len(train):,} utts ({train_hours:.1f} h, {len(train_loader)} batches/epoch) | "
        f"dev {len(dev):,} utts")

    # ---- optimizer ----------------------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.peak_lr, betas=(0.9, 0.98),
                            weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: noam(s, tcfg.warmup_steps))
    start_epoch, step, best_wer = 0, 0, float("inf")
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch, step, best_wer = state["epoch"], state["step"], state["best_wer"]
        log(f"resumed from {last}: epoch {start_epoch}, step {step}, best dev WER {best_wer:.2%}")
        del state

    # ---- training loop ------------------------------------------------------------------
    run_start = time.time()
    budget = tcfg.max_hours * 3600
    epoch_secs: list[float] = []
    epoch = start_epoch
    try:
        for epoch in range(start_epoch, tcfg.epochs):
            elapsed = time.time() - run_start
            if epoch_secs and elapsed + epoch_secs[-1] > budget:
                log(f"time budget: {elapsed / 3600:.1f} h used, next epoch would pass "
                    f"--max-hours {tcfg.max_hours}. Stopping. Run the same command to resume.")
                break

            train_loader.batch_sampler.set_epoch(epoch)
            epoch_start = window_start = time.time()
            window_audio, window_loss, window_n = 0.0, 0.0, 0
            for i, (feats, feat_lens, targets, target_lens, _) in enumerate(train_loader):
                logits, out_lens = model(feats.to(device), feat_lens.to(device))
                loss = ctc_loss(logits, out_lens, targets, target_lens)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step()
                sched.step()
                step += 1

                window_audio += feat_lens.sum().item() / 100
                window_loss += loss.item()
                window_n += 1
                if step % tcfg.log_every == 0:
                    dt = time.time() - window_start
                    speed = window_audio / dt
                    eta = (len(train_loader) - i - 1) * (time.time() - epoch_start) / (i + 1)
                    log(f"epoch {epoch + 1} step {step} [{i + 1}/{len(train_loader)}] "
                        f"loss {window_loss / window_n:.3f} | grad {grad_norm:.2f} | "
                        f"lr {sched.get_last_lr()[0]:.2e} | {speed:.0f}x realtime | "
                        f"epoch ETA {eta / 60:.0f} min")
                    window_start, window_audio, window_loss, window_n = time.time(), 0.0, 0.0, 0
                if tcfg.max_steps and step >= tcfg.max_steps:
                    break

            wer, pairs = evaluate(model, dev_loader, tokenizer, device)
            epoch_secs.append(time.time() - epoch_start)   # includes eval, for the time budget
            improved = wer < best_wer
            best_wer = min(best_wer, wer)
            save(last, model, opt, sched, epoch + 1, step, best_wer, tcfg, mcfg)
            if improved:
                shutil.copy(last, run_dir / "best.pt")
            log(f"=== epoch {epoch + 1} done in {epoch_secs[-1] / 60:.1f} min | dev WER {wer:.2%}"
                f"{'  (new best)' if improved else ''} | total {(time.time() - run_start) / 3600:.2f} h")
            for ref, hyp in pairs[:: max(1, len(pairs) // 3)][:3]:
                log(f"    ref: {ref[:90]}")
                log(f"    hyp: {hyp[:90]}")
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"epoch": epoch + 1, "step": step, "dev_wer": wer,
                                    "minutes": epoch_secs[-1] / 60, "lr": sched.get_last_lr()[0]}) + "\n")
            if tcfg.max_steps and step >= tcfg.max_steps:
                log(f"reached --max-steps {tcfg.max_steps}")
                break
    except KeyboardInterrupt:
        # Ctrl+C: keep the progress of finished epochs; the interrupted epoch restarts on resume.
        log(f"interrupted during epoch {epoch + 1}; last.pt holds the last finished epoch")
    log(f"best dev WER {best_wer:.2%} | checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
