"""Bounded official rainfall samples; no discharge data are read."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import io
import json
import lzma
import math
from pathlib import Path
import re
import socket
import signal
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib


LIMIT = 64 * 1024 * 1024


def request_deadline(*_):
    raise TimeoutError("25s wall-clock request deadline")


def uint7(data, pos):
    first, value, mask = data[pos], 0, 0x80
    pos += 1
    for index in range(8):
        if not first & mask:
            return value | ((first & (mask - 1)) << (8 * index)), pos
        value |= data[pos] << (8 * index)
        pos += 1
        mask >>= 1
    return value, pos


def encoded_directory(data):
    # This published header uses one LZMA1 coder; reject other 7z layouts.
    if data[:2] != b"\x17\x06":
        raise ValueError("Not the supported 7z encoded header")
    offset, pos = uint7(data, 2)
    streams, pos = uint7(data, pos)
    assert streams == 1 and data[pos] == 9
    size, pos = uint7(data, pos + 1)
    assert data[pos:pos+9] == b"\x00\x07\x0b\x01\x00\x01\x23\x03\x01"
    pos += 9
    assert data[pos:pos+2] == b"\x01\x05"
    props = data[pos+2:pos+7]
    assert data[pos+7] == 12
    unpacked, _ = uint7(data, pos + 8)
    prop = props[0]
    lc, rem = prop % 9, prop // 9
    return {"offset": offset + 32, "size": size, "unpacked": unpacked,
            "filter": {"id": lzma.FILTER_LZMA1, "dict_size": int.from_bytes(props[1:], "little"),
                       "lc": lc, "lp": rem % 5, "pb": rem // 5}}


def parse_rows(text):
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if len(fields) < 3 or not re.fullmatch(r"\d{4}/\d{2}/\d{2}", fields[0].strip()):
            continue
        date, clock = fields[0].strip(), fields[1].strip()
        hour = int(clock.split(":")[0])
        if not 0 <= hour <= 24 or clock.split(":")[1] != "00":
            raise ValueError("Unsupported hourly clock: " + clock)
        stamp = datetime.strptime(date, "%Y/%m/%d") + timedelta(hours=hour)
        rows.append({"raw_date": date, "raw_clock": clock,
                     "time_jst_assumed": stamp.replace(tzinfo=timezone(timedelta(hours=9))).isoformat(),
                     "raw_values": [value.strip() for value in fields[2:]]})
    return rows


def decode(raw):
    for encoding in ("utf-8", "euc-jp", "cp932"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("No supported Japanese encoding")


class Fetcher:
    def __init__(self, out):
        self.out, self.used, self.requests = out, 0, []
        self.deadline = time.monotonic() + 120

    def get(self, url, name, range_header=None, cap=2 * 1024 * 1024):
        """Count all consumed payload, including failed attempts; reject ignored Range."""
        for attempt in range(1, 3):
            entry = {"url": url, "attempt": attempt, "range": range_header}
            self.requests.append(entry)
            raw = bytearray()
            response = None
            try:
                if self.used + 16384 >= LIMIT:
                    raise ValueError("64MiB total transfer budget exhausted")
                seconds_left = self.deadline - time.monotonic()
                if seconds_left <= 0:
                    raise ValueError("120s collector wall-clock deadline reached")
                headers = {"User-Agent": "waterpaper-bounded-source-check/1.0", "Accept-Encoding": "identity"}
                if range_header:
                    headers["Range"] = range_header
                signal.signal(signal.SIGALRM, request_deadline)
                signal.alarm(min(25, max(1, math.ceil(seconds_left))))
                self.used += 16384
                response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=25)
                entry.update(status=response.status, final_url=response.url, headers=dict(response.headers))
                # Conservative accounting reserve for request/response headers.
                if range_header and response.status != 206:
                    raise ValueError("Range ignored: refused response body; no full archive download")
                length = response.headers.get("Content-Length")
                if length and int(length) > min(cap, LIMIT - self.used):
                    raise ValueError("Response exceeds per-request or cumulative cap")
                while True:
                    remaining = min(cap - len(raw), LIMIT - self.used)
                    if remaining <= 0:
                        raise ValueError("Transfer cap reached; response not accepted as complete")
                    chunk = response.read(min(16384, remaining))
                    if not chunk:
                        break
                    raw.extend(chunk)
                    self.used += len(chunk)
                path = self.out / f"{name}.attempt{attempt}.raw"
                path.write_bytes(raw)
                entry.update(file=str(path), payload_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), status_label="complete")
                return bytes(raw)
            except http.client.IncompleteRead as error:
                partial = error.partial or b""
                raw.extend(partial)
                self.used += len(partial)
                entry.update(error=str(error), status_label="failed_incomplete")
            except (OSError, ValueError, urllib.error.URLError) as error:
                entry.update(error=str(error), status_label="failed")
            finally:
                signal.alarm(0)
                if response:
                    response.close()
                entry["payload_bytes"] = len(raw)
                if raw and "file" not in entry:
                    path = self.out / f"{name}.attempt{attempt}.partial"
                    path.write_bytes(raw)
                    entry["file"] = str(path)
                time.sleep(1)
            if range_header or entry.get("status_label") == "failed_incomplete":
                break
        return None


def selfcheck():
    rows = parse_rows("2019/10/08,23:00,0.5,-\n2019/10/08,24:00,1.0,*\n2019/10/09,01:00,0.0,欠測")
    assert rows[1]["time_jst_assumed"] == "2019-10-09T00:00:00+09:00"
    assert rows[0]["raw_values"] == ["0.5", "-"]
    assert datetime.fromisoformat(rows[2]["time_jst_assumed"]) - datetime.fromisoformat(rows[1]["time_jst_assumed"]) == timedelta(hours=1)
    fixture = bytes.fromhex("17 06 f4 89 17 6d f2 01 09 c0 90 7a 00 07 0b 01 00 01 23 03 01 01 05 5d 00 00 06 00 0c c4 e6 74 0a 01 ff 21 ca 55 00 00")
    parsed = encoded_directory(fixture)
    assert parsed["size"] == 31376 and parsed["unpacked"] == 292070


def directory_followup(previous, out):
    prior = json.loads((previous / "receipt.json").read_text())
    fetch = Fetcher(out)
    fetch.used = prior["accounted_bytes"] + 32768  # First run omitted reserve for two timed-out requests.
    header = next(previous.glob("us_7z_next_header.*.raw")).read_bytes()
    info = encoded_directory(header)
    assert info["size"] < 4 * 1024 * 1024 and info["unpacked"] < 4 * 1024 * 1024
    forcing = next(f for f in prior["us"]["official_files"] if f["key"] == "timeseries.7z")
    packed = fetch.get(forcing["url"], "us_encoded_directory", f"bytes={info['offset']}-{info['offset']+info['size']-1}", cap=info["size"]+1)
    result = {"prior_receipt": str(previous / "receipt.json"), "prior_accounted_bytes": prior["accounted_bytes"],
              "host": socket.gethostname(), "status": "partial_no_P_values", "directory_encoding": info}
    if packed:
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=[info["filter"]])
        directory = decoder.decompress(packed, max_length=info["unpacked"]+1)
        assert len(directory) == info["unpacked"]
        (out / "us_decoded_directory.raw").write_bytes(directory)
        names = re.findall(r"[0-9]{8}[^\x00]*?\.nc", directory.decode("utf-16le", errors="ignore"))
        usgs_id = prior["us"]["existing_usgs_id"].split("_")[0]
        result.update(n_nc_names=len(names), nc_names_preview=names[:5],
                      existing_usgs_id=usgs_id,
                      existing_id_found=any(usgs_id in name for name in names))
        if directory[:3] == b"\x01\x04\x06":
            _, pos = uint7(directory, 3)
            streams, _ = uint7(directory, pos)
            result["n_packed_streams"] = streams
    # HTTP first attempt timed out; HTTPS is a distinct bounded fallback endpoint.
    base = "https://www1.river.go.jp"
    ids = ";".join(prior["jp"]["rain_ids"]) + ";"
    url = base + "/cgi-bin/SrchSiteSuiData2.exe?SUIKEI=83028000&BGNDATE=20191008&ENDDATE=20191015&ID=" + ids
    page = fetch.get(url, "jp_https_request_page")
    result["jp"] = {**prior["jp"], "status": "partial_https_failed"}
    if page:
        text, enc = decode(page)
        match = re.search(r'href=["\x27](/dat/dload/\d+\.dat)["\x27]', text, re.I)
        if match:
            dat = fetch.get(base + match.group(1), "jp_https_rain_original_dat")
            if dat:
                text, enc = decode(dat)
                rows = parse_rows(text)
                result["jp"].update(status="hourly_table_retrieved" if rows else "partial_no_hourly_rows",
                                     data_encoding=enc, header_lines=[line for line in text.splitlines() if line.startswith("#")],
                                     n_rows=len(rows), sample_rows=rows[:3],
                                     midnight_rows=[r for r in rows if r["raw_clock"].startswith("24:")][:2])
                (out / "jp_rows_preserved.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            result["jp"]["status"] = "partial_https_no_data_link"
    result.update(accounted_bytes_including_prior=fetch.used, requests=fetch.requests)
    (out / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


def first_member_layout(data):
    assert data[:3] == b"\x01\x04\x06"
    offset, pos = uint7(data, 3)
    count, pos = uint7(data, pos)
    assert data[pos] == 9
    pos += 1
    sizes = []
    for _ in range(count):
        size, pos = uint7(data, pos)
        sizes.append(size)
    assert data[pos:pos+3] == b"\x00\x07\x0b"
    folders, pos = uint7(data, pos + 3)
    assert folders == count and data[pos] == 0
    pos += 1
    properties = []
    for _ in range(folders):
        assert data[pos:pos+4] == b"\x01\x21\x21\x01"
        properties.append(data[pos+4])
        pos += 5
    assert data[pos] == 12
    pos += 1
    for _ in range(folders):
        _, pos = uint7(data, pos)
    assert data[pos:pos+3] == b"\x00\x08\x0d"
    pos += 3
    for _ in range(folders):
        _, pos = uint7(data, pos)
    assert data[pos] == 9
    first_size, _ = uint7(data, pos + 1)
    prop = properties[0]
    assert prop <= 40
    dictionary = 0xffffffff if prop == 40 else (2 | (prop & 1)) << (prop // 2 + 11)
    return offset + 32, sizes[0], first_size, dictionary


def station_followup(previous, out, resume=None):
    import h5py  # Existing server environment only; no installation.
    prior = json.loads((previous / "receipt.json").read_text())
    fetch = Fetcher(out)
    fetch.used = prior["accounted_bytes_including_prior"]
    directory = (previous / "us_decoded_directory.raw").read_bytes()
    start, packed_size, member_size, dictionary = first_member_layout(directory)
    assert prior["nc_names_preview"][0] == "01011000.nc" and member_size < 64 * 1024 * 1024
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "dict_size": dictionary}])
    member = bytearray()
    consumed = 0
    if resume:
        old = json.loads((resume / "receipt.json").read_text())
        fetch.used = old["accounted_bytes_including_prior"]
        prefix_file = resume.parents[3] / old["requests"][0]["file"]
        prefix = prefix_file.read_bytes()
        assert old["requests"][0]["range"].startswith("bytes=32-")
        member.extend(decoder.decompress(prefix, max_length=member_size))
        consumed = len(prefix)
    url = "https://zenodo.org/api/records/15066778/files/timeseries.7z/content"
    result = {"host": socket.gethostname(), "prior_receipt": str(previous / "receipt.json"),
              "first_member": "01011000.nc", "packed_stream_bytes": packed_size, "member_bytes": member_size,
              "status": "partial_incomplete_first_member", "q_values_read": False, "h5py_version": h5py.__version__}
    try:
        while len(member) < member_size:
            amount = min(256 * 1024, packed_size-consumed, LIMIT-fetch.used-16384-1)
            if amount <= 0:
                break
            raw = fetch.get(url, f"us_first_stream_offset{consumed}",
                            f"bytes={start+consumed}-{start+consumed+amount-1}", cap=amount+1)
            if not raw:
                break
            consumed += len(raw)
            member.extend(decoder.decompress(raw, max_length=member_size-len(member)))
        result.update(packed_prefix_bytes=consumed, decoded_first_member_bytes=len(member))
        if len(member) == member_size:
            path = out / "01011000.nc"
            path.write_bytes(member)
            result["first_member_sha256"] = hashlib.sha256(member).hexdigest()
            def plain(value):
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="replace")
                if hasattr(value, "tolist"):
                    return plain(value.tolist())
                if isinstance(value, list):
                    return [plain(item) for item in value]
                return value
            with h5py.File(path, "r") as file:
                result["variable_names"] = list(file.keys())
                result["precipitation"] = {}
                for name in file:
                    if any(word in name.lower() for word in ("precip", "rain")) or name.lower() == "time":
                        dataset = file[name]
                        result["precipitation"][name] = {"shape": list(dataset.shape),
                            "attrs": {key: plain(value) for key, value in dataset.attrs.items() if key not in ("DIMENSION_LIST", "REFERENCE_LIST")},
                            "first_three_raw": plain(dataset[:3])}
                result["status"] = "first_station_P_and_time_read" if any("precip" in name.lower() or "rain" in name.lower() for name in result["precipitation"]) else "partial_no_P_variable"
    except Exception as error:
        result["error"] = repr(error)
    finally:
        result.update(accounted_bytes_including_prior=fetch.used, requests=fetch.requests,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        (out / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--directory-of")
    ap.add_argument("--station-of")
    ap.add_argument("--resume-of")
    args = ap.parse_args()
    selfcheck()
    if args.selfcheck:
        print("parser selfcheck passed")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.station_of:
        station_followup(Path(args.station_of), out, Path(args.resume_of) if args.resume_of else None)
        return
    if args.directory_of:
        directory_followup(Path(args.directory_of), out)
        return
    fetch = Fetcher(out)
    result = {"host": socket.gethostname(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "limit_bytes": LIMIT, "data_role": "source_sample_only", "us": {}, "jp": {}}
    try:
        # ZIP central directory only: do not decompress/read existing Q members.
        with zipfile.ZipFile("./data/Hourly2.zip") as archive:
            names = sorted(name for name in archive.namelist() if name.endswith(".nc"))
        result["us"]["existing_q_member"] = names[0] if names else None
        result["us"]["existing_usgs_id"] = Path(names[0]).stem.split("_")[0] if names else None
        metadata = fetch.get("https://zenodo.org/api/records/15066778", "us_zenodo_metadata")
        if metadata:
            record = json.loads(metadata)
            files = [{"key": f["key"], "size": f["size"], "url": f["links"]["self"]} for f in record.get("files", [])]
            result["us"]["official_files"] = files
            forcing = next((f for f in files if f["key"] == "timeseries.7z"), None)
            if forcing:
                raw = fetch.get(forcing["url"], "us_7z_start_header", "bytes=0-31", cap=64)
                if raw and len(raw) == 32 and raw[:6] == b"7z\xbc\xaf\x27\x1c":
                    assert zlib.crc32(raw[12:32]) == struct.unpack("<I", raw[8:12])[0]
                    offset, size, crc = struct.unpack("<QQI", raw[12:32])
                    result["us"]["next_header"] = {"offset": 32 + offset, "size": size, "crc32": crc}
                    # The archive index alone does not provide independently compressed station blobs.
                    if size <= 4 * 1024 * 1024:
                        header = fetch.get(forcing["url"], "us_7z_next_header", f"bytes={32+offset}-{31+offset+size}", cap=size+1)
                        if header:
                            result["us"]["next_header_valid"] = len(header) == size and zlib.crc32(header) == crc
                result["us"]["status"] = "partial_archive_only_no_station_P"
            else:
                result["us"]["status"] = "partial_metadata_no_forcing_file"
        else:
            result["us"]["status"] = "partial_metadata_access_failed"
        stations = ["103031283301020:0101", "103031283301030:0101", "103031283301040:0101"]
        result["jp"].update(q_reference_id="303031283301030:0202", rain_ids=stations,
                            rain_names=["栗橋", "川俣", "八斗島"], river_system="利根川 83028000",
                            upstream_boundary="unknown", pairing_status="partial_same_river_system_not_basin_average")
        params = urllib.parse.urlencode({"SUIKEI": "83028000", "BGNDATE": "20191008", "ENDDATE": "20191015", "ID": ";".join(stations) + ";"})
        base = "http://www1.river.go.jp"
        page = fetch.get(base + "/cgi-bin/SrchSiteSuiData2.exe?" + params, "jp_request_page")
        if page:
            text, enc = decode(page)
            result["jp"]["request_encoding"] = enc
            match = re.search(r'href=["\x27](/dat/dload/\d+\.dat)["\x27]', text, re.I)
            if match:
                dat = fetch.get(base + match.group(1), "jp_rain_original_dat")
                if dat:
                    text, enc = decode(dat)
                    rows = parse_rows(text)
                    result["jp"].update(data_encoding=enc, header_lines=[line for line in text.splitlines() if line.startswith("#")],
                                        n_rows=len(rows), sample_rows=rows[:3], midnight_rows=[r for r in rows if r["raw_clock"].startswith("24:")][:2],
                                        status="hourly_table_retrieved" if rows else "partial_no_hourly_rows",
                                        timezone="JST_assumed_from_existing_collector_pending_raw_header", unit="see_raw_header")
                    (out / "jp_rows_preserved.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            else:
                result["jp"]["status"] = "partial_no_data_link"
        else:
            result["jp"]["status"] = "partial_request_failed"
    except Exception as error:
        result["unexpected_error"] = repr(error)
    finally:
        result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), accounted_bytes=fetch.used,
                      requests=fetch.requests, status="partial", q_values_read=False)
        (out / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
