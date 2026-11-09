"""One Tone station, official single-station pages, and one month JMA fallback."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.parse

JST = timezone(timedelta(hours=9))
LIMIT = 100 * 1024 * 1024


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.attrs = [], None, None, []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("td", "th") and self.row is not None:
            self.cell, self.attrs = [], [(tag, attrs)]
        elif self.cell is not None:
            self.attrs.append((tag, attrs))

    def handle_data(self, text):
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append({"text": "".join(self.cell).strip(), "markup": self.attrs})
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def decode(raw):
    for encoding in ("utf-8", "euc-jp", "cp932"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    return raw.decode("euc-jp", errors="replace"), "euc-jp-replacement"


def numeric(raw):
    return float(raw) if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw) else None


def jma_rows(text, day):
    if "降水量" not in text:
        return []
    table = Tables()
    table.feed(text)
    rows = []
    for cells in table.rows:
        if len(cells) < 4 or not re.fullmatch(r"\d{1,2}", cells[0]["text"]):
            continue
        hour = int(cells[0]["text"])
        if 1 <= hour <= 24:
            stamp = datetime.fromisoformat(day).replace(tzinfo=JST) + timedelta(hours=hour)
            cell = cells[3]
            rows.append({"time_jst": stamp.isoformat(), "raw_hour": hour, "P_mm": numeric(cell["text"]),
                         "raw_P": cell["text"], "quality_markup": cell["markup"], "role": "point_observation"})
    return rows


def monthly_rows(text, year, month):
    table = Tables()
    table.feed(text)
    rows = []
    for cells in table.rows:
        if len(cells) < 25:
            continue
        date_text=cells[0]["text"]
        if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}",date_text):
            yy,mm,day=map(int,date_text.split('/'))
            if (yy,mm)!=(year,month):
                continue
        elif re.fullmatch(r"\d{1,2}",date_text):
            day=int(date_text)
        else:
            continue
        try:
            date = datetime(year, month, day, tzinfo=JST)
        except ValueError:
            continue
        for hour, cell in enumerate(cells[1:25], 1):
            rows.append({"time_jst": (date+timedelta(hours=hour)).isoformat(), "raw_day": day,
                         "raw_hour": hour, "value": numeric(cell["text"]), "raw_value": cell["text"],
                         "quality_markup": cell["markup"]})
    return rows


class Collector:
    def __init__(self, out):
        self.out, self.used, self.requests = out, 0, []
        self.deadline = time.monotonic()+600

    def get(self, url, name, session=False, data=None):
        left = int(self.deadline-time.monotonic())
        if left <= 0 or self.used+16384 >= LIMIT:
            return None
        body, headers, errors = (self.out/f"{name}.{suffix}" for suffix in ("raw", "headers", "stderr"))
        cap = min(10*1024*1024, LIMIT-self.used-16384)
        args = ["curl", "-4", "-L", "--connect-timeout", "15", "--max-time", str(min(75, left)),
                "--max-filesize", str(cap), "-A", "Mozilla/5.0", "-sS", "-D", str(headers),
                "-o", str(body), "-w", "%{http_code} %{size_download} %{url_effective}"]
        if session:
            args += ["-b", str(self.out/"own_session.cookies"), "-c", str(self.out/"own_session.cookies")]
        if data:
            for key,value in data.items():
                args += ["--data-urlencode",key+"="+value]
        args.append(url)
        started = time.monotonic()
        with errors.open("wb") as stderr:
            process = subprocess.run(args, stdout=subprocess.PIPE, stderr=stderr, timeout=min(80,left+2))
        size = body.stat().st_size if body.exists() else 0
        self.used += size+16384
        entry = {"url": url, "file": str(body), "header_file": str(headers), "stderr_file": str(errors),
                 "exit_code": process.returncode, "curl_summary": process.stdout.decode(errors="replace"),
                 "payload_bytes": size, "elapsed_s": round(time.monotonic()-started,3)}
        self.requests.append(entry)
        if not size:
            return None
        entry["sha256"] = hashlib.sha256(body.read_bytes()).hexdigest()
        if process.returncode != 0 or not entry["curl_summary"].startswith("200 "):
            return None
        text, encoding = decode(body.read_bytes())
        entry["encoding"] = encoding
        return text


def existing_q(out):
    chunks = Path("./data/raw/japan_mlit_tone_Q_2015_2024/chunks")
    records, sources = {}, []
    # The first two October days are stored in the preceding eight-day chunk.
    paths = list(chunks.glob("20190925*.csv")) + list(chunks.glob("201910*.csv"))
    for path in sorted(paths):
        with path.open() as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "栗橋" not in reader.fieldnames:
                continue
            sources.append(str(path))
            for row in reader:
                rawtime = row[reader.fieldnames[0]]
                if "T24:" in rawtime:
                    date = datetime.fromisoformat(rawtime.split("T")[0]).replace(tzinfo=JST)+timedelta(days=1)
                else:
                    date = datetime.fromisoformat(rawtime)
                if datetime(2019,10,1,tzinfo=JST) < date <= datetime(2019,11,1,tzinfo=JST):
                    value = row["栗橋"].strip()
                    records[date.isoformat()] = {"time_jst": date.isoformat(), "Q_m3_s": numeric(value), "raw_Q": value}
    values = list(records.values())
    (out/"existing_kurihashi_q_oct2019.json").write_text(json.dumps(values,ensure_ascii=False,indent=2))
    return values, sources


def selfcheck():
    text = '<p>降水量(mm)</p><table><tr><td>24</td><td>1</td><td>2</td><td><font color="blue">3.0</font></td></tr></table>'
    rows = jma_rows(text,"2019-10-12")
    assert rows[0]["time_jst"] == "2019-10-13T00:00:00+09:00" and rows[0]["P_mm"]==3.0
    assert rows[0]["quality_markup"] and numeric("--") is None and numeric("1.0 ]") is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--selfcheck",action="store_true")
    args = ap.parse_args()
    selfcheck()
    if args.selfcheck:
        print("parser selfcheck passed")
        return
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    collector = Collector(out)
    receipt = {"host":socket.gethostname(),"started_at_utc":datetime.now(timezone.utc).isoformat(),
               "status":"partial", "Q_id":"303031283301030", "upstream_boundary":"unknown",
               "training":False,"credentials_used":False,"proxy_variable_names":[key for key in os.environ if key.lower().endswith("_proxy")]}
    try:
        authors = [
            ("https://open.metu.edu.tr/handle/11511/109654","ishikari_author_repository"),
            ("https://www.sciencedirect.com/science/article/pii/S0301479724009174","ishikari_final_article"),
            ("https://arxiv.org/pdf/2111.04732","ishikari_preprint"),
            ("https://api.github.com/search/repositories?q=CNNsLSTM","ishikari_repo_search")]
        for url, name in authors:
            collector.get(url,name)
        receipt["author_package"] = "not_confirmed_from_checked_final_paper_repository_preprint_search"
        q, q_sources = existing_q(out)
        receipt["existing_Q"]={"n_rows":len(q),"n_numeric":sum(row["Q_m3_s"] is not None for row in q),
                                "sources":q_sources,"sample":q[:3],"unit_source":"existing_MLIT_collector_and_source_headers"}
        base = "https://www1.river.go.jp"
        collector.get(base+"/contents.html","mlit_bootstrap",True)
        collector.get(base+"/cgi-bin/SiteInfo.exe?ID=303031283301030","mlit_Q_metadata",True)
        receipt["mlit_P"] = {}
        for station in ("103031283301020","103031283301030","103031283301040"):
            metadata = collector.get(base+"/cgi-bin/SiteInfo.exe?ID="+station,"mlit_P_metadata_"+station,True)
            url = base+"/cgi-bin/DspRainData.exe?KIND=1&ID="+station+"&BGNDATE=20191001&ENDDATE=20191031&KAWABOU=NO"
            text = collector.get(url,"mlit_P_month_"+station,True)
            if not text:
                text = collector.get(url.replace("https:","http:"),"mlit_P_http_month_"+station,True)
            rows = monthly_rows(text,2019,10) if text else []
            (out/("mlit_P_"+station+"_rows.json")).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
            receipt["mlit_P"][station]={"n_candidate_hourly_rows":len(rows),"metadata_received":bool(metadata),
                                        "unit_confirmed_mm":bool(text and "mm" in text),"status":"candidate_needs_header_alignment_review" if rows else "no_hourly_table"}
        # Different official source; do not relaunch the timed-out multi-station endpoint.
        all_p = []
        days = [12] + [day for day in range(1,32) if day!=12]
        for day in days:
            url = f"https://www.data.jma.go.jp/stats/etrn/view/hourly_s1.php?prec_no=42&block_no=47624&year=2019&month=10&day={day}&view=p1"
            text = collector.get(url,f"jma_maebashi_201910{day:02d}")
            rows = jma_rows(text,f"2019-10-{day:02d}") if text else []
            if len(rows)!=24:
                receipt["jma_stop_day"] = day
                break
            all_p.extend(rows)
            time.sleep(1)
        all_p.sort(key=lambda row:row["time_jst"])
        (out/"jma_maebashi_P_rows.json").write_text(json.dumps(all_p,ensure_ascii=False,indent=2))
        qmap = {row["time_jst"]:row for row in q}
        pairs = [{**row,**qmap[row["time_jst"]]} for row in all_p if row["time_jst"] in qmap]
        (out/"same_clock_point_P_Q.json").write_text(json.dumps(pairs,ensure_ascii=False,indent=2))
        receipt["JMA"]={"station":"前橋","block_no":"47624","prec_no":"42","unit":"mm","time_zone":"JST",
                         "P_role":"point_observation_not_basin_average","n_hourly_rows":len(all_p),"n_numeric_P":sum(row["P_mm"] is not None for row in all_p),
                         "sample":all_p[:3],"n_same_clock_Q_pairs":len(pairs),
                         "n_numeric_point_pairs":sum(row["P_mm"] is not None and row["Q_m3_s"] is not None for row in pairs),
                         "pairing_status":"same_clock_only_upstream_spatial_support_unverified","sample_pairs":[r for r in pairs if r["P_mm"] is not None and r["Q_m3_s"] is not None][:3]}
    except Exception as error:
        receipt["error"] = repr(error)
    finally:
        receipt.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),accounted_bytes=collector.used,
                       requests=collector.requests,limitations=["No upstream basin boundary/area weights validated", "Symbols retained; not filled with zero", "No final external validation/training"])
        (out/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        print(json.dumps({"status":receipt["status"],"bytes":collector.used,"JMA":receipt.get("JMA")},ensure_ascii=False))


if __name__ == "__main__":
    main()
