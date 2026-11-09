"""Read three ordered ZIP members in memory; never extract original time series."""
import argparse
import datetime as dt
import io
import json
import zipfile

import h5py
import numpy as np


def plain(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def inspect(path):
    report = {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "archive": path, "scope": "first, middle, last .nc member after lexical sorting; not a population audit",
              "source_url": "https://zenodo.org/records/16729675", "source_version": "v6",
              "samples": []}
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith(".nc"))
        report["nc_member_count"] = len(names)
        report["uncompressed_nc_bytes"] = sum(archive.getinfo(n).file_size for n in names)
        for index in (0, len(names) // 2, len(names) - 1):
            name = names[index]
            sample = {"member": name, "sorted_index": index, "variables": {}}
            with h5py.File(io.BytesIO(archive.read(name)), "r") as nc:
                sample["global_attributes"] = {k: plain(v) for k, v in nc.attrs.items()}
                for key, data in nc.items():
                    if not isinstance(data, h5py.Dataset):
                        continue
                    attrs = {k: plain(v) for k, v in data.attrs.items()
                             if k not in ("DIMENSION_LIST", "REFERENCE_LIST")}
                    entry = {"shape": list(data.shape), "dtype": str(data.dtype), "attributes": attrs}
                    if data.dtype.kind in "fiu":
                        values = np.asarray(data[...]).reshape(-1)
                        valid = np.isfinite(values)
                        for fill_key in ("_FillValue", "missing_value"):
                            if fill_key in data.attrs:
                                valid &= ~np.isin(values, np.asarray(data.attrs[fill_key]).reshape(-1))
                        entry.update(count=int(values.size), finite_nonfill_count=int(valid.sum()),
                                     finite_nonfill_fraction=float(valid.mean()),
                                     missing_count=int((~valid).sum()),
                                     negative_nonfill_count=int(((values < 0) & valid).sum()))
                        if valid.any():
                            entry.update(min=float(values[valid].min()), max=float(values[valid].max()),
                                         first_valid_index=int(np.flatnonzero(valid)[0]),
                                         last_valid_index=int(np.flatnonzero(valid)[-1]))
                        if key.lower() == "time":
                            entry["first_raw"] = plain(values[0])
                            entry["last_raw"] = plain(values[-1])
                            unique, counts = np.unique(np.diff(values), return_counts=True)
                            entry["raw_step_counts"] = {str(plain(v)): int(c) for v, c in zip(unique, counts)}
                    sample["variables"][key] = entry
            report["samples"].append(sample)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("output")
    args = parser.parse_args()
    report = inspect(args.archive)
    assert len(report["samples"]) == 3
    assert all(s["variables"] for s in report["samples"])
    with open(args.output, "w") as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
