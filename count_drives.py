from pathlib import Path
from zipfile import ZipFile

import pandas as pd


ZIP_PATH = Path.home() / "Downloads" / "data_Q1_2026.zip"

unique_drives: set[str] = set()
total_rows = 0

with ZipFile(ZIP_PATH) as archive:
    january_files = sorted(
        name
        for name in archive.namelist()
        if name.startswith("data_Q1_2026/2026-01-")
        and name.endswith(".csv")
    )

    print(f"January files found: {len(january_files)}")

    for index, filename in enumerate(january_files, start=1):
        with archive.open(filename) as file:
            frame = pd.read_csv(
                file,
                usecols=["serial_number"],
                dtype={"serial_number": "string"},
            )

        total_rows += len(frame)
        unique_drives.update(frame["serial_number"].dropna().tolist())

        print(
            f"[{index:02d}/{len(january_files)}] "
            f"{filename}: "
            f"{len(frame):,} rows | "
            f"{len(unique_drives):,} unique drives so far"
        )

print("\nFINAL COUNTS")
print(f"Drive-day rows processed: {total_rows:,}")
print(f"Unique drives processed: {len(unique_drives):,}")
