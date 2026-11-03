"""Unpack the Målfrid .jsonl.gz files in data/maalfrid/ to plain-text files.

Documents are grouped by domain in data/maalfrid/textfiles/<domain>/, with one
<type>_<uuid>.txt per document containing its fulltext paragraphs.
"""

import gzip
import json
from pathlib import Path

DATA_DIR = Path("data/maalfrid")
OUT_DIR = DATA_DIR / "textfiles"


def main() -> None:
    jsonl_files = sorted(DATA_DIR.glob("*.jsonl.gz"))
    n_docs = 0
    for path in jsonl_files:
        domain, _lang, doc_type = path.name.removesuffix(".jsonl.gz").rsplit("_", 2)
        doc_dir = OUT_DIR / domain
        doc_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                # doc_id looks like "<urn:uuid:...>"
                doc_uuid = doc["doc_id"].strip("<>").removeprefix("urn:uuid:")
                text = "\n".join(doc["fulltext"])
                (doc_dir / f"{doc_type}_{doc_uuid}.txt").write_text(
                    text + "\n", encoding="utf-8"
                )
                n_docs += 1
    print(
        f"Wrote {n_docs} text files from {len(jsonl_files)} jsonl.gz files to {OUT_DIR}/"
    )


if __name__ == "__main__":
    main()
