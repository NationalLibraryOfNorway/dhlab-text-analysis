"""Make a subset of the text files with a fixed number of random files per domain.

Only the hand-picked domains in DOMAINS and the document types in DOC_TYPES are
used. Texts where less than MIN_LATIN_SHARE of the letters are Latin (e.g.
Chinese documents mislabelled as Nynorsk) are dropped, and domains with fewer
than N_PER_DOMAIN remaining files are skipped.
"""

import random
import shutil
import unicodedata
from pathlib import Path

DATA_DIR = Path("data/maalfrid")
IN_DIR = DATA_DIR / "textfiles"
OUT_DIR = DATA_DIR / "subset"
N_PER_DOMAIN = 10
SEED = 43
DOC_TYPES = ["html", "doc"]
MIN_LATIN_SHARE = 0.9

DOMAINS = [
    "22julisenteret.no",
    "aho.no",
    "aldringoghelse.no",
    "arbeidsretten.no",
    "arbeidstilsynet.no",
    "artsdatabanken.no",
    "banenor.no",
    "barneombudet.no",
    "bibliotekutvikling.no",
    "bioteknologiradet.no",
    "denkulturelleskolesekken.no",
    "digitalarkivet.no",
    "diskrimineringsnemnda.no",
    "ffi.no",
    "fhi.no",
    "fiskeridir.no",
    "forskning.no",
    "forsvaret.no",
    "helsedirektoratet.no",
    "helsenorge.no",
    "husbanken.no",
    "khrono.no",
    "mattilsynet.no",
    "nasjonalmuseet.no",
    "nmbu.no",
    "nynorskbok.no",
    "nysgjerrigper.no",
    "plantevernleksikonet.no",
    "riksantikvaren.no",
    "skeivtarkiv.no",
    "toll.no",
    "uio.no",
    "utdanning.no",
    "varsom.no",
    "vinmonopolet.no",
]


def is_latin(path: Path) -> bool:
    letters = [ch for ch in path.read_text(encoding="utf-8") if ch.isalpha()]
    n_latin = sum(unicodedata.name(ch, "").startswith("LATIN") for ch in letters)
    return n_latin >= MIN_LATIN_SHARE * len(letters)


def main() -> None:
    rng = random.Random(SEED)
    n_kept = n_skipped = 0
    for domain in DOMAINS:
        domain_dir = IN_DIR / domain
        files = sorted(path for doc_type in DOC_TYPES for path in domain_dir.glob(f"{doc_type}_*.txt"))
        files = [path for path in files if is_latin(path)]
        if len(files) < N_PER_DOMAIN:
            n_skipped += 1
            continue
        out_dir = OUT_DIR / domain_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in rng.sample(files, N_PER_DOMAIN):
            shutil.copy2(path, out_dir / path.name)
        n_kept += 1
    print(f"Copied {N_PER_DOMAIN} files from each of {n_kept} domains to {OUT_DIR}/ ({n_skipped} skipped)")


if __name__ == "__main__":
    main()
