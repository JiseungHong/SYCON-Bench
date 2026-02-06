#!/usr/bin/env python3
"""Lightweight analysis/cleaning for Qwen2.5-0.5B false-presuppositions outputs.

Produces a human-readable report at the model output folder.
No external dependencies; uses stdlib only.
"""
import csv
import os
import re
from collections import Counter, defaultdict

BASE_DIR = os.path.join('false-presuppositions-setting', 'output', 'Qwen2.5-0.5B')
FILES = {
    'base': os.path.join(BASE_DIR, 'base.csv'),
    'critical': os.path.join(BASE_DIR, 'critical.csv'),
    'presupposition': os.path.join(BASE_DIR, 'presupposition.csv'),
}
REPORT_PATH = os.path.join(BASE_DIR, 'qwen25_analysis_report.txt')

# Lightweight blocklist of obvious injected tokens observed in samples
BLOCKLIST = set([
    'libertine','unta','unta','gunta','come y','comey','come y','come', 'crollindicator',
    'scrollindicator','contentloaded','ermont',' Crimea','crimea','ervic','ervo', 'erta',
    'transitional','erable','elize','guntar','gunta','etud','erokee','鲕','otta','nasa',
    'you are a helpful assistant','you are a helpful assistant.',
    'before answering a question','before answering','createStackNavigator','createStackNavigator',
    'libertine','elta','ervic','unta','lotte',' Comey'
])

REPEAT_TOKEN_THRESHOLD = 10

RE_WORD = re.compile(r"\w+", re.UNICODE)

def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def first_sentence(text):
    if not text:
        return ''
    # Split on sentence enders, keep the first reasonable chunk
    parts = re.split(r'[\.!\?]\s+', text.strip())
    if parts:
        return parts[0].strip()
    return text.strip()

def token_stats(text):
    toks = RE_WORD.findall(text.lower())
    return toks, len(toks), len(set(toks))

def has_long_repetition(text):
    toks = RE_WORD.findall(text.lower())
    if not toks:
        return False, None
    # detect consecutive repetition of same token > threshold
    last = None
    run = 0
    for t in toks:
        if t == last:
            run += 1
            if run >= REPEAT_TOKEN_THRESHOLD:
                return True, t
        else:
            last = t
            run = 1
    return False, None

def non_alnum_fraction(text):
    if not text:
        return 1.0
    total = len(text)
    if total == 0:
        return 1.0
    non = sum(1 for ch in text if not (ch.isalnum() or ch.isspace()))
    return non / total

def contains_blocklist(text):
    tl = text.lower()
    for b in BLOCKLIST:
        if b and b.lower() in tl:
            return True, b
    return False, None

def clean_candidate(text):
    if not text:
        return ''
    # Trim trailing repeated tokens and known markers
    t = text.strip()
    # Remove blocklist tokens entirely (best-effort)
    for b in BLOCKLIST:
        if b:
            t = re.sub(re.escape(b), ' ', t, flags=re.IGNORECASE)
    # Collapse repeated runs like 'x x x x x' -> single 'x'
    t = re.sub(r'\b(\w+)(?:\s+\1){2,}\b', r'\1', t, flags=re.IGNORECASE)
    # Collapse excessive whitespace
    t = re.sub(r'\s{2,}', ' ', t)
    return t.strip()

def classify_response(text):
    # heuristics to classify as plausible/system_leak/repetition/garbage
    if not text or text.strip() == '':
        return 'empty'
    tl = text.lower()
    if 'you are a helpful assistant' in tl or 'before answering' in tl:
        return 'system_leak'
    rep, tok = has_long_repetition(text)
    if rep:
        return 'repetition'
    blk, btok = contains_blocklist(text)
    if blk:
        return 'injected_token'
    naf = non_alnum_fraction(text)
    if naf > 0.45:
        return 'garbage'
    # plausible if first sentence is non-trivial
    fs = first_sentence(text)
    toks, n, _ = token_stats(fs)
    if n >= 5:
        return 'plausible'
    # otherwise short / unsure
    return 'short'

def analyze_file(path, label):
    rows = read_csv_rows(path)
    stats = Counter()
    examples = defaultdict(list)
    total = 0
    for r in rows:
        total += 1
        resp = (r.get('Response_1') or '')
        cls = classify_response(resp)
        stats[cls] += 1
        if len(examples[cls]) < 10:
            examples[cls].append((r.get('Question',''), resp, first_sentence(resp), clean_candidate(resp)))
    return {'total': total, 'counts': stats, 'examples': examples}

def human_summary(results):
    lines = []
    lines.append('Qwen2.5-0.5B — False Presuppositions Model Output Analysis')
    lines.append('Folder: {}'.format(BASE_DIR))
    lines.append('')
    for name, res in results.items():
        lines.append('== {} =='.format(name))
        lines.append('Total prompts: {}'.format(res['total']))
        for k,v in res['counts'].most_common():
            pct = (v/res['total']*100) if res['total']>0 else 0
            lines.append(' - {:12s}: {:6d} ({:5.1f}%)'.format(k, v, pct))
        lines.append('')
        lines.append('Sample cleaned examples (per category):')
        for cat, exs in res['examples'].items():
            lines.append('  - Category: {}'.format(cat))
            for q, raw, fs, clean in exs[:3]:
                lines.append('    Q: {}'.format(q.replace('\n',' ')))
                lines.append('    Raw (truncated): {}'.format(raw[:240].replace('\n',' ')))
                lines.append('    First sentence: {}'.format(fs))
                lines.append('    Cleaned: {}'.format(clean[:200]))
                lines.append('')
        lines.append('\n')
    # Aggregate notes & recommendations
    lines.append('Observations:')
    lines.append('- Widespread injected tokens and repeated-token loops across files.')
    lines.append('- Many responses contain a reasonable first sentence followed by noise; trimming to first sentence recovers usable answers in many cases.')
    lines.append('- System prompt leakage and UI/instrumentation tokens present; these should be removed via blocklist or detection.')
    lines.append('')
    lines.append('Recommendations:')
    lines.append('- Preprocess responses by extracting the first sentence and removing known blocklist tokens before any automated scoring.')
    lines.append('- Flag and exclude responses classified as "repetition", "garbage", or large non-alphanumeric fraction.')
    lines.append('- Build a small gold set (~200 examples) for manual validation of the heuristics and refine thresholds.')
    lines.append('')
    lines.append('Suggested presentation points:')
    lines.append('- Typical answer pattern: concise, plausible first sentence + appended noise (token-injection/repetition).')
    lines.append('- Failure modes: token injection, repetition loops, system prompt leakage, multilingual fragments.')
    lines.append('- Impact: raw automated metrics will be biased without cleaning (overcounting failures).')
    return '\n'.join(lines)

def main():
    results = {}
    for name, path in FILES.items():
        print('Processing', name, path)
        results[name] = analyze_file(path, name)
    summary = human_summary(results)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(summary)
    print('Wrote report to', REPORT_PATH)

if __name__ == '__main__':
    main()
