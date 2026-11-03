"""Make a subset of the text files with a fixed number of random files per domain.

Only the hand-picked domains in DOMAINS and the document types in DOC_TYPES are
used. Texts where less than MIN_LATIN_SHARE of the letters are Latin are dropped,
and domains with fewer than N_PER_DOMAIN remaining files are skipped.
OUT_DIR is emptied before the subset is written.
"""

import random
import shutil
import unicodedata
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path("data/maalfrid")
IN_DIR = DATA_DIR / "textfiles"
OUT_DIR = DATA_DIR / "subset"
N_PER_DOMAIN = 100
SEED = 43
DOC_TYPES = ["html", "doc"]
MIN_LATIN_SHARE = 0.9

# Two pairs of closely related domains and one that stands apart, to see whether
# close domains still separate.
DOMAINS = [
    "fiskeridir.no",
    "mattilsynet.no",
    "uio.no",
    "uit.no",
    "varsom.no",
]


def is_latin(path: Path) -> bool:
    letters = [ch for ch in path.read_text(encoding="utf-8") if ch.isalpha()]
    n_latin = sum(unicodedata.name(ch, "").startswith("LATIN") for ch in letters)
    return n_latin >= MIN_LATIN_SHARE * len(letters)


def main() -> None:
    # Start from an empty OUT_DIR so files from earlier runs don't linger.
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    rng = random.Random(SEED)
    n_kept = n_skipped = 0
    for domain in tqdm(DOMAINS):
        domain_dir = IN_DIR / domain
        files = sorted(
            path
            for doc_type in DOC_TYPES
            for path in domain_dir.glob(f"{doc_type}_*.txt")
        )
        files = [path for path in files if is_latin(path)]
        if len(files) < N_PER_DOMAIN:
            n_skipped += 1
            continue
        out_dir = OUT_DIR / domain_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in rng.sample(files, N_PER_DOMAIN):
            shutil.copy2(path, out_dir / path.name)
        n_kept += 1
    print(
        f"Copied {N_PER_DOMAIN} files from each of {n_kept} domains to {OUT_DIR}/ ({n_skipped} skipped)"
    )


if __name__ == "__main__":
    main()
