"""Download the Nynorsk part of Målfrid 2026 from Språkbanken and extract it to data/.

Source: https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-107/
Licence: NLOD 2.0
"""

import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.nb.no/sbfil/tekst/maalfrid_2026/maalfrid_2026_nno.tar"
DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "maalfrid"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / URL.rsplit("/", 1)[-1]

    if not archive_path.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, archive_path)
    else:
        print(f"{archive_path} already exists, skipping download")

    print(f"Extracting {archive_path} to {OUT_DIR}/ ...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(OUT_DIR, filter="data")

    print("Done")


if __name__ == "__main__":
    main()
