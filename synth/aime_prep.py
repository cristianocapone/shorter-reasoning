"""AIME 2024 + AIME 2025 (60 problems, integer answers 0-999) for the brevity study's harder benchmark (NatComm must-fix 8:
a benchmark on which a realistic 32k budget still binds). Run on a LOGIN node (needs internet), from the repo root:
    python synth/aime_prep.py
writes synth/aime/aime24_25.jsonl (id, problem, answer (int), level, year, source) and synth/aime/aime24_25_ids.txt, in the
format of synth/math/test500.jsonl, for
    MATH_EVAL=1 python synth/math_harvest.py --data synth/aime/aime24_25.jsonl --ids synth/aime/aime24_25_ids.txt ...
Sources (Hugging Face datasets, pinned to the revisions checked on 2026-11-02):
  AIME 2024  Maxwell-Jia/AIME_2024 (ID '2024-I-1', Problem, Answer). Not HuggingFaceH4/aime_2024 for the statements: its
             2024 II Problem 8 reads 'rests on the outside of S, it is externally tangent' for both radii, whereas the
             official problem (answer 127) has 'inside ... internally' for r_i. Three Maxwell-Jia statements contain a
             control character where a backslash escape was decoded ('\\f' of \\frac -> form feed in I-1 and I-15, '\\t' of
             \\tfrac -> tab in II-8); they are restored to the backslash sequence (the count is printed).
  AIME 2025  MathArena/aime_2025 (problem_idx 1-15 = AIME I, 16-30 = AIME II; answer). Statements as published there,
             including the TikZ/Asymptote source of the figures.
  Cross-check: each answer must equal the one in HuggingFaceH4/aime_2024 (2024) or yentinglin/aime_2025 (2025), matched
             by contest and problem number; all 60 agreed on 2026-11-02. A disagreement or a missing problem aborts;
             --no-crosscheck skips the check (e.g. if a check source disappears).
ids aime2024_I_01 ... aime2025_II_15; level 'AIME 2024 I' etc. The checker (math_harvest.correct) accepts integer answers.
"""
import argparse, json, os, re
from datasets import load_dataset

SRC = {2024: ("Maxwell-Jia/AIME_2024", "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792", "Problem", "Answer"),
       2025: ("MathArena/aime_2025", "c94da77eb22bbd6439e62a323bec18493a421302", "problem", "answer")}
CHECK = {2024: ("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"),
         2025: ("yentinglin/aime_2025", "6f71d77b0b89b9dabe07ab466c51df33f514df7f")}
ESC = {"\x07": "\\a", "\x08": "\\b", "\t": "\\t", "\x0b": "\\v", "\x0c": "\\f", "\r": "\\r"}   # decoded escapes -> LaTeX


def repair(s):
    """Restore backslash escapes that were decoded into control characters (newlines are kept); returns (text, count)."""
    s = s.replace("\r\n", "\n")
    return "".join(ESC.get(ch, ch) for ch in s), sum(ch in ESC for ch in s)


def contest_key(year, r):
    """('I' or 'II', problem number) of a source row."""
    if year == 2024:
        m = re.fullmatch(r"2024-(I|II)-(\d+)", r["ID"].strip())
        return m.group(1), int(m.group(2))
    i = int(r["problem_idx"])
    return ("I" if i <= 15 else "II"), (i - 1) % 15 + 1


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def url_key(r):
    m = re.search(r"(\d{4})_AIME_(I+)_Problems/Problem_(\d+)", r["url"])
    return m.group(2), int(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="synth/aime"); ap.add_argument("--name", default="aime24_25")
    ap.add_argument("--no-crosscheck", action="store_true")
    a = ap.parse_args()
    rows, n_fix = [], 0
    for year, (name, rev, pf, af) in SRC.items():
        got = {}
        for r in load_dataset(name, split="train", revision=rev):
            part, num = contest_key(year, r)
            problem, nf = repair(r[pf])
            n_fix += nf
            ans = int(str(r[af]).strip())
            need(0 <= ans <= 999, f"{name} {part}-{num}: answer {ans} is not an AIME answer")
            need((part, num) not in got, f"duplicate {year} {part}-{num} in {name}")
            got[part, num] = dict(id=f"aime{year}_{part}_{num:02d}", problem=problem.strip(), answer=ans,
                                  level=f"AIME {year} {part}", year=year, source=f"{name}@{rev[:7]}")
        want = {(p, i) for p in ("I", "II") for i in range(1, 16)}
        need(set(got) == want, f"{name}: missing {sorted(want - set(got))}, unexpected {sorted(set(got) - want)}")
        if not a.no_crosscheck:
            cname, crev = CHECK[year]
            ref = {url_key(r): int(str(r["answer"]).strip()) for r in load_dataset(cname, split="train", revision=crev)}
            bad = [(k, got[k]["answer"], ref.get(k)) for k in sorted(want) if ref.get(k) != got[k]["answer"]]
            need(not bad, f"{year}: answers differ from {cname} (contest, ours, theirs): {bad}")
            print(f"{year}: 30 answers agree with {cname}")
        rows += [got[p, i] for p in ("I", "II") for i in range(1, 16)]
    for r in rows:
        need(r["problem"] and not any(ord(c) < 32 and c != "\n" for c in r["problem"]), f"{r['id']}: empty or control characters")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{a.name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    open(f"{a.out_dir}/{a.name}_ids.txt", "w").write("\n".join(r["id"] for r in rows) + "\n")
    print(f"wrote {a.out_dir}/{a.name}.jsonl: {len(rows)} problems ({sum(r['year'] == 2024 for r in rows)} from 2024, "
          f"{sum(r['year'] == 2025 for r in rows)} from 2025); {n_fix} decoded escape(s) restored")


if __name__ == "__main__":
    main()
