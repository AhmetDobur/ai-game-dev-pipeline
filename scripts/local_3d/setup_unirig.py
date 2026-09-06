"""Make a fresh UniRig checkout run on this project's hardware.

UniRig ships configured for a datacentre card. Four of its defaults are wrong
for a Turing GPU (Titan RTX, SM 7.5) and one of its dependencies is wrong for
current numpy, so a clean clone fails four separate ways before it produces a
single weight. Every one of those was diagnosed by hand once; this applies the
same edits deterministically so the next install does not repeat it.

    python scripts/local_3d/setup_unirig.py /path/to/UniRig

Idempotent: safe to re-run after `git pull` in the UniRig checkout, which is
when it is actually needed, since these are edits to UniRig's own files.
"""
import pathlib
import subprocess
import sys


def patch(path, pairs):
    """Apply literal replacements to a file. Returns what changed."""
    if not path.exists():
        return []
    before = path.read_text(encoding="utf-8")
    after = before
    hit = []
    for old, new in pairs:
        if old in after:
            after = after.replace(old, new)
            hit.append(f"{old} -> {new}")
    if after != before:
        path.write_text(after, encoding="utf-8")
    return hit


def main(root):
    root = pathlib.Path(root)
    if not (root / "run.py").exists():
        raise SystemExit(f"{root} does not look like a UniRig checkout")
    changed = []

    # 1. FlashAttention 2 requires Ampere. On Turing it raises
    #    "FlashAttention only supports Ampere GPUs or newer" from mha_fwd.
    #    Both the transformer configs and PTv3's own default have to move.
    for cfg in (root / "configs" / "model").glob("*.yaml"):
        changed += [f"{cfg.name}: {h}" for h in patch(cfg, [
            ("_attn_implementation: flash_attention_2", "_attn_implementation: sdpa"),
            ("flash: True", "flash: False"),
            ("flash: true", "flash: false"),
        ])]
    ptv3 = root / "src" / "model" / "pointcept" / "models" / "PTv3Object.py"
    # PTv3 already carries a standard-attention path for exactly this case --
    # the `not self.enable_flash` branches -- it simply is not the default, and
    # the flash branch asserts flash_attn is importable before checking the card.
    changed += [f"PTv3Object.py: {h}" for h in patch(ptv3, [
        ("enable_flash=True,", "enable_flash=False,")])]

    # 2. Turing has no bfloat16 path in cuBLAS: the skin pass dies with
    #    CUBLAS_STATUS_EXECUTION_FAILED on a CUDA_R_16BF gemm. fp16 fails the
    #    same way here, so inference runs in fp32 -- it is one mesh, offline.
    for cfg in (root / "configs" / "task").glob("*.yaml"):
        changed += [f"{cfg.name}: {h}" for h in patch(cfg, [
            ("precision: bf16-mixed", "precision: 32"),
            ("precision: 16-mixed", "precision: 32")])]

    # 3. spconv 2.3.8 is built against numpy 1.x and segfaults against numpy 2
    #    partway through the skin pass -- no traceback, just a crash after the
    #    checkpoint loads.
    py = sys.argv[2] if len(sys.argv) > 2 else sys.executable
    try:
        import numpy
        if int(numpy.__version__.split(".")[0]) >= 2:
            subprocess.run([py, "-m", "pip", "install", "numpy<2"], check=False)
            changed.append("numpy pinned below 2 (spconv 2.3.8 segfaults on numpy 2)")
    except ImportError:
        pass

    for line in changed:
        print(f"[setup_unirig] {line}")
    if not changed:
        print("[setup_unirig] already patched")
    # Not fixed here because it is not UniRig's file: torch 2.6 defaults
    # torch.load to weights_only=True and refuses the Box object UniRig's
    # published checkpoints pickle beside their tensors. blender_motion's
    # try_unirig_skin sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for the stages it
    # launches, which is where it belongs -- the env var is scoped to the call
    # rather than loosening the default for the whole environment.
    print("[setup_unirig] note: the caller must set "
          "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (blender_motion does)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
