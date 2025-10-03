# eval/build_known_from_csv.py
import argparse, csv, io, json, pathlib, re
from urllib.parse import urlparse

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

def read_text_smart(path: pathlib.Path) -> str:
    data = path.read_bytes()
    for enc in ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    # last resort: latin-1 never fails
    return data.decode("latin-1", errors="replace")

def norm_key(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", s)

def find_col(cols_norm, *candidates):
    for c in candidates:
        if c in cols_norm: return c
    return None

def normalize_url(u: str) -> str:
    if not u: return ""
    p = urlparse(u.strip())
    netloc = p.netloc.lower().replace("www.", "")
    path = (p.path or "").rstrip("/").lower()
    return f"{netloc}{path}"

def load_url_to_pids(passages_path: pathlib.Path):
    url2pids = {}
    with passages_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            j = json.loads(line)
            nu = normalize_url(j.get("source_url", ""))
            if not nu: continue
            url2pids.setdefault(nu, []).append(j["pid"])
    return url2pids

def main():
    ap = argparse.ArgumentParser(description="Build questions_known_full.jsonl from gold_qa.csv")
    ap.add_argument("--csv", default=str(pathlib.Path(__file__).parent/"datasets/gold_qa.csv"))
    ap.add_argument("--passages", default=str(pathlib.Path(__file__).parents[1]/"kb/chunks/passages.jsonl"))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent/"datasets/questions_known_full.jsonl"))
    ap.add_argument("--drop-empty", action="store_true", help="Drop rows with no matching gold_pids")
    ap.add_argument("--report", default="", help="Optional path to write a CSV report of matches")
    args = ap.parse_args()

    passages_path = pathlib.Path(args.passages)
    if not passages_path.exists():
        raise SystemExit(f"Passages not found: {passages_path}. Run scripts.make_chunks first.")

    url2pids = load_url_to_pids(passages_path)

    csv_text = read_text_smart(pathlib.Path(args.csv))
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise SystemExit("CSV has no header row.")

    headers_norm = [norm_key(h) for h in reader.fieldnames]
    # Build row-normalizer mapping
    key_map = {h: nh for h, nh in zip(reader.fieldnames, headers_norm)}

    # Resolve columns (accept several variants)
    # qid
    qid_col = find_col(set(headers_norm), "qid", "id", "question_id")
    # question text
    q_col   = find_col(set(headers_norm), "question", "query", "q")
    # tag(s)
    tag_col = find_col(set(headers_norm), "tag", "tags", "category")
    # urls (allowed/gold)
    urls_col = find_col(set(headers_norm),
                        "allowed_urls", "allowedurl", "allowed_url", "urls", "gold_urls", "allowedpages")

    missing = [name for name, col in [("question", q_col), ("urls", urls_col), ("qid", qid_col)] if col is None]
    if missing:
        raise SystemExit(f"CSV is missing required columns: {missing}\nHeaders seen (normalized): {headers_norm}")

    rows_out = []
    report_rows = []
    total, empty = 0, 0

    for raw in reader:
        # normalize keys for this row
        row = {key_map[k]: (v or "").strip() for k, v in raw.items()}

        qid = row.get(qid_col, "") or f"q{total+1:03d}"
        question = row[q_col]
        tag = row.get(tag_col, "")
        urls_raw = row.get(urls_col, "")

        # split urls on comma/semicolon/newline
        parts = [u.strip().strip('"').strip("'") for u in re.split(r"[;\n,]+", urls_raw) if u.strip()]
        gold_pids = []
        for u in parts:
            nu = normalize_url(u)
            gold_pids.extend(url2pids.get(nu, []))

        gold_pids = sorted(set(gold_pids))
        if not gold_pids:
            empty += 1
            if args.drop_empty:
                total += 1
                continue

        rows_out.append({"qid": qid, "tag": tag, "question": question, "gold_pids": gold_pids})
        report_rows.append((qid, question, len(gold_pids), parts))

        total += 1

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for j in rows_out:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows_out)} rows to {outp} (source rows: {total}, empty gold: {empty})")

    if args.report:
        rp = pathlib.Path(args.report)
        with rp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qid", "question", "matched_gold_pid_count", "input_urls"])
            for qid, q, n, parts in report_rows:
                w.writerow([qid, q, n, " | ".join(parts)])
        print(f"Wrote report to {rp}")

if __name__ == "__main__":
    main()
