r"""inspect_ow_audio.py - size and format of the OpenWhistle audio, before downloading it.
Reads parquet footers (column sizes) and decodes a few clips from one row group, only if that
row group's audio chunk is under MAX_PEEK_MB.

Run:  python inspect_ow_audio.py
"""
import io
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from scipy.io import wavfile
from tqdm.auto import tqdm

from openwhistle_data import _parquet_files

MAX_PEEK_MB = 300


def main():
    fs, files = _parquet_files()
    print(f"{len(files)} parquet files")
    col_bytes, rows, total, first = defaultdict(int), 0, 0, None
    for f in tqdm(files, desc="reading footers"):
        pf = pq.ParquetFile(fs.open(f))
        md = pf.metadata
        rows += md.num_rows
        total += fs.info(f)["size"]
        for rg in range(md.num_row_groups):
            g = md.row_group(rg)
            for c in range(g.num_columns):
                col_bytes[g.column(c).path_in_schema.split(".")[0]] += g.column(c).total_compressed_size
        if first is None:
            first = pf
    print(f"\nrows: {rows}   total parquet size: {total / 1e9:.2f} GB")
    print("compressed size by column:")
    for k, v in sorted(col_bytes.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v / 1e6:10.1f} MB")

    pf = first
    names = [fld.name for fld in pf.schema_arrow]
    print("\nfields:", names)
    audio = [n for n in names if "audio" in n.lower()]
    if not audio:
        print("no column with 'audio' in its name - paste this output and I'll adapt")
        return
    name = audio[0]
    print(f"audio column: {name}   type: {pf.schema_arrow.field(name).type}")
    g = pf.metadata.row_group(0)
    chunk = sum(g.column(c).total_compressed_size for c in range(g.num_columns)
                if g.column(c).path_in_schema.split(".")[0] == name)
    print(f"row group 0: {g.num_rows} rows, audio chunk {chunk / 1e6:.1f} MB")
    if chunk > MAX_PEEK_MB * 1e6:
        print(f"chunk larger than {MAX_PEEK_MB} MB - not peeking; paste this output")
        return
    col = pf.read_row_group(0, columns=[name]).column(0)
    for i in range(min(3, len(col))):
        v = col[i].as_py()
        b = v.get("bytes") if isinstance(v, dict) else v
        path = v.get("path") if isinstance(v, dict) else None
        if b is None:
            print(f"clip {i}: no embedded bytes (path={path})")
            continue
        magic = bytes(b[:4])
        line = f"clip {i}: {len(b) / 1e3:.0f} kB  magic={magic}  path={path}"
        if magic == b"RIFF":
            sr, x = wavfile.read(io.BytesIO(b))
            line += f"  WAV  sr={sr}  dur={len(x) / sr:.2f} s  dtype={x.dtype}  channels={1 if x.ndim == 1 else x.shape[1]}"
        tqdm.write(line)


if __name__ == "__main__":
    main()