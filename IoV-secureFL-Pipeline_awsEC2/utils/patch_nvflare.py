"""
Patches nvflare.app_opt.xgboost.__init__ to guard the XGBBaggingRecipe import
behind a try/except.

Root cause (NVFlare 2.7.x): recipes/__init__.py eagerly imports all three
recipe classes including XGBHorizontalRecipe (histogram). That module imports
TBAnalyticsReceiver -> torch.utils.tensorboard -> tensorboard, even though
this project only uses tree-based bagging and never touches TensorBoard.

This script is idempotent — safe to run multiple times.
"""
import pathlib
import sys

TARGET = pathlib.Path(".venv/lib/python3.12/site-packages/nvflare/app_opt/xgboost/__init__.py")

ORIGINAL_IMPORT = 'from nvflare.app_opt.xgboost.recipes import XGBBaggingRecipe'

PATCHED_BLOCK = '''\
# Patched: recipes/__init__.py eagerly imports XGBHorizontalRecipe which
# requires torch+tensorboard via nvflare.app_opt.tracking.tb.tb_receiver.
# This project uses only tree-based bagging and never needs TensorBoard.
try:
    from nvflare.app_opt.xgboost.recipes import XGBBaggingRecipe
    __all__ = ["XGBBaggingRecipe"]
except ImportError:
    __all__ = []'''


def main():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found — run 'uv sync' first.", file=sys.stderr)
        sys.exit(1)

    content = TARGET.read_text()

    if "# Patched:" in content:
        print("nvflare patch already applied — nothing to do.")
        return

    if ORIGINAL_IMPORT not in content:
        print(
            f"ERROR: expected import line not found in {TARGET}.\n"
            "NVFlare version may have changed — review the patch.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Replace the bare import + __all__ line with the guarded block
    patched = content.replace(
        'from nvflare.app_opt.xgboost.recipes import XGBBaggingRecipe\n\n__all__ = ["XGBBaggingRecipe"]',
        PATCHED_BLOCK,
    )

    if patched == content:
        # Fallback: replace just the import line (layout may differ)
        patched = content.replace(ORIGINAL_IMPORT, PATCHED_BLOCK)

    TARGET.write_text(patched)
    print(f"Patched {TARGET}")


if __name__ == "__main__":
    main()
