#!/usr/bin/env python3
"""论文事实核查闸门。任何一项失败即 exit 1,禁止提交。

三道检查:
  A. 引用真实性:main.tex 里每个 \cite key 必须在 references.bib 中,
     且该条目必须带 arXiv/DOI 标识(可被独立核验)
  B. 数字可溯源:每个 \meas{} 数字必须出现在 EVIDENCE.md 的溯源表里,
     并指向一个真实存在的结果文件
  C. 占位符未残留:提交版不得含 \num{} 或 \todo{}
用法:
  python3 check_claims.py           # 草稿模式(允许 \num/\todo)
  python3 check_claims.py --final   # 提交模式(全部必须清零)
"""
import re, sys, os, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "main.tex"
BIB = ROOT / "paper" / "references.bib"
EVID = ROOT / "paper" / "EVIDENCE.md"
FINAL = "--final" in sys.argv
fails, warns = [], []

tex = TEX.read_text(encoding="utf-8")
# 注释行不渲染,其中的 \num/\todo 字样不是占位符 —— 计入会造成永远无法通过的误报;
# 但只剥离整行注释,行内 % 后可能是合法内容(如 \%),不碰。
tex = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
bib = BIB.read_text(encoding="utf-8") if BIB.exists() else ""

# ---------- A. 引用真实性 ----------
cited = set()
for m in re.finditer(r"\\cite[tp]?\{([^}]*)\}", tex):
    cited.update(k.strip() for k in m.group(1).split(",") if k.strip())
entries = {}
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n@|@\w+\{([^,]+),(.*)$", bib, re.S):
    pass
for m in re.finditer(r"@\w+\{([^,\s]+)\s*,", bib):
    key = m.group(1)
    start = m.end()
    nxt = bib.find("\n@", start)
    entries[key] = bib[start: nxt if nxt > 0 else len(bib)]
missing = sorted(cited - set(entries))
if missing:
    fails.append(f"[A1] 引用了 bib 中不存在的键: {missing}")
# A2 要求"可独立核验",而非"必须有 arXiv" —— 很多论文只在会议/期刊发表,
# 没有预印本。强求 arXiv 会逼人编造编号,恰是本闸门要防的事。
# 接受任一凭据:arXiv 号 / DOI / URL / (会议或期刊名 + 年份)。
def verifiable(body):
    if re.search(r"arxiv|doi|\burl\b|howpublished", body, re.I):
        return True
    has_venue = re.search(r"\b(booktitle|journal)\s*=", body, re.I)
    has_year = re.search(r"\byear\s*=\s*[{\"]?\s*(19|20)\d\d", body, re.I)
    return bool(has_venue and has_year)

unverifiable = sorted(k for k in cited & set(entries) if not verifiable(entries[k]))
if unverifiable:
    fails.append(f"[A2] 以下条目缺少可核验凭据(需 arXiv/DOI/URL,或 venue+year): {unverifiable}")

# A4:凡声称有 arXiv 号的,格式必须合法(防手抖或编造出畸形编号)
badax = []
for k in cited & set(entries):
    for m in re.finditer(r"arxiv[:\s]*([0-9]{4}\.[0-9]{4,5})", entries[k], re.I):
        yy, mm = int(m.group(1)[:2]), int(m.group(1)[2:4])
        if not (7 <= yy <= 99 and 1 <= mm <= 12):
            badax.append(f"{k}:{m.group(1)}")
if badax:
    fails.append(f"[A4] arXiv 编号格式非法(年月不合理): {badax}")
uncited = sorted(set(entries) - cited)
if uncited:
    warns.append(f"[A3] bib 中未被引用(可删): {uncited}")

# ---------- B. 数字可溯源 ----------
meas = set()
for m in re.finditer(r"\\meas\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", tex):
    v = re.sub(r"\\[a-zA-Z]+|[{}]", "", m.group(1)).strip()
    if re.search(r"\d", v):
        meas.add(v)
if not EVID.exists():
    fails.append("[B0] 缺少 paper/EVIDENCE.md —— 每个 \\meas{} 数字都必须在此溯源")
else:
    ev = EVID.read_text(encoding="utf-8")
    ev_nums = set()
    for line in ev.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and re.search(r"\d", cells[0]):
                ev_nums.add(cells[0].replace("`", "").strip())
    # 登记行必须写来源;此前无来源的行被静默跳过,等于登记即通过
    for line in ev.splitlines():
        if not line.strip().startswith("|"): continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 3 and re.search(r"\d", cells[0]) and "---" not in cells[0]:
            if not [x for x in re.findall(r"`([^`]+)`", cells[2]) if "/" in x]:
                fails.append(f"[B3] EVIDENCE.md 中 {cells[0]} 未填来源文件")
    undocumented = sorted(v for v in meas if v not in ev_nums)
    if undocumented:
        fails.append(f"[B1] 以下 \\meas{{}} 数字未在 EVIDENCE.md 溯源: {undocumented}")
    # 溯源指向的文件必须真实存在
    for line in ev.splitlines():
        if line.strip().startswith("|") and "/" in line:
            for tok in re.findall(r"`([^`]+)`", line):
                if "/" in tok and not tok.startswith("http"):
                    p = ROOT / tok
                    if not p.exists():
                        fails.append(f"[B2] EVIDENCE.md 指向的文件不存在: {tok}")

# ---------- C. 占位符 ----------
n_num = len(re.findall(r"\\num\{", tex))
n_todo = len(re.findall(r"\\todo\{", tex))
if FINAL:
    if n_num: fails.append(f"[C1] 提交版仍有 {n_num} 个 \\num{{}} 占位")
    if n_todo: fails.append(f"[C2] 提交版仍有 {n_todo} 个 \\todo{{}} 占位")
else:
    warns.append(f"[C] 草稿状态: {n_num} 个待填结果, {n_todo} 个待补信息")

# ---------- 报告 ----------
print(f"引用 {len(cited)} 条 | bib {len(entries)} 条 | \\meas 数字 {len(meas)} 个")
for w in warns: print("  WARN " + w)
for f in fails: print("  FAIL " + f)
print("PASS" if not fails else f"\n{len(fails)} 项失败 —— 禁止提交")
sys.exit(1 if fails else 0)
