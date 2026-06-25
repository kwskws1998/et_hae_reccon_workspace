#!/usr/bin/env python
"""Apply runtime-only safety knobs to the cloned official RECCON script."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path("repos/RECCON/train_qa.py")
    if not target.exists():
        raise FileNotFoundError(target)

    text = target.read_text(encoding="utf-8")

    if "grad_accum_steps = int(os.environ.get('RECCON_GRAD_ACCUM_STEPS'" not in text:
        old = """    if fold == '1':
        num_steps = int(27915/batch_size)
    else:
        num_steps = int(25697/batch_size)
"""
        new = """    grad_accum_steps = int(os.environ.get('RECCON_GRAD_ACCUM_STEPS', '1'))
    effective_batch_size = batch_size * grad_accum_steps
    if fold == '1':
        num_steps = int(27915/effective_batch_size)
    else:
        num_steps = int(25697/effective_batch_size)
"""
        if old not in text:
            raise RuntimeError("Could not locate official RECCON num_steps block to patch.")
        text = text.replace(old, new, 1)

    if "'process_count': int(os.environ.get('RECCON_PROCESS_COUNT'" not in text:
        old = "        'fp16': False,\n"
        new = (
            "        'fp16': False,\n"
            "        'process_count': int(os.environ.get('RECCON_PROCESS_COUNT', '4')),\n"
            "        'gradient_accumulation_steps': grad_accum_steps,\n"
        )
        if old not in text:
            raise RuntimeError("Could not locate official RECCON train_args block to patch.")
        text = text.replace(old, new, 1)

    target.write_text(text, encoding="utf-8")
    print(f"[reccon-official] patched runtime knobs in {target}")


if __name__ == "__main__":
    main()
