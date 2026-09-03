#!/usr/bin/env python3
"""D 关:数字必须能从源文件里真的读出来(而不只是"登记过")。

对可机器验证的来源(JSON / 结构化文本),直接把 EVIDENCE.md 声称的数值
拿去源文件里比对。对不上就 FAIL —— 这堵住了"登记表里写假数字指向真文件"的漏洞。

用法: python3 verify_numbers.py
"""
import json, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
fails, checked, skipped = [], 0, []

# ── 关键修复 ──────────────────────────────────────────────
# 此前 D 关拿脚本里硬编码的期望字符串去比对重算值,等于自己和自己对账:
# 论文里把数字改掉也不会被发现。现在改为从 main.tex 抽出该数字**实际写的值**,
# 只有当论文确实声称了某个值,才拿它与重算结果比对。
TEXT = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")

def claimed_in_paper(token):
    """论文是否声称了这个值;返回论文中实际出现的形式(可能已被篡改)。"""
    core = token.replace("\\%", "").replace("$", "").replace("\\", "").strip()
    pat = re.escape(core).replace(r"\+", r"\+?")
    m = re.search(r"\\meas\{[^{}]*" + pat + r"[^{}]*\}", TEXT)
    return m.group(0) if m else None

def norm(x):
    """把 '26,999' / '2.50B' / '20.6\\%' 归一成 float"""
    s = str(x).replace(",", "").replace("\\%", "").replace("%", "").replace("$", "").strip()
    mult = 1.0
    if s.endswith("B"): mult, s = 1e9, s[:-1]
    elif s.endswith("M"): mult, s = 1e6, s[:-1]
    elif s.endswith("k"): mult, s = 1e3, s[:-1]
    try: return float(s) * mult
    except ValueError: return None

def close(a, b, tol=0.01, exact=False):
    """整数计数要求精确(允许千分位/单位换算后的整数相等);
    比率与统计量容忍论文中的舍入,默认 1%。一刀切的容差对小样本会误伤。"""
    if a is None or b is None: return False
    if exact:
        return abs(a - b) < 0.5          # 计数:换算后应为同一整数
    if b == 0: return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= tol

def is_count(claim):
    """判定是否为"应精确匹配的计数"。

    带 B/M/k 单位后缀的写法(354M、2.50B)是论文中正常的舍入表述,不是计数 ——
    要求它们精确等于 353,593,717 会误伤合法写法,逼人绕过闸门。
    只有裸整数(26,999 / 931)才要求精确。
    """
    c = str(claim).strip()
    if "%" in c or "$" in c or c.startswith(("+", "-")): return False
    if c.endswith(("B", "M", "k")): return False        # 带单位 = 近似量
    return "." not in c.replace(",", "")

# ---- 从 datasheet_stats.json 直接核对(权威来源,可机器验证)----
ds_path = ROOT / "experiments/results/datasheet_stats.json"
if ds_path.exists():
    d = json.loads(ds_path.read_text())
    th = d.get("thinking", {})
    red = d.get("redaction", {})
    expect = {
        "26,999":    d.get("sessions_exported"),
        "1.65M":     th.get("assistant_turns"),
        "15.3":      d.get("core_metric_nonempty_thinking_per_100_assistant_turns"),
        "317,956":   th.get("blocks"),
        "20.6\\%":   (th.get("empty_rate") or 0) * 100,
        "16.4\\%":   ((th.get("by_position") or {}).get("last", {}).get("empty_rate") or 0) * 100,
        "22.5\\%":   ((th.get("by_position") or {}).get("earlier", {}).get("empty_rate") or 0) * 100,
        "1,924,228": red.get("total"),
        "10,507":    red.get("distinct_usernames_aliased"),
        "82,031":    (red.get("counts") or {}).get("PRIVATE_IP"),
        "11,375":    (red.get("counts") or {}).get("PHONE_CN"),
        "1.81M":     (red.get("counts") or {}).get("USERNAME_PATH"),
        "931":       (d.get("decontamination") or {}).get("human_like_sessions_touching_benchmark_repos"),
        "26,468":    (d.get("near_duplicate_clusters") or {}).get("singletons"),
        "2.50B":     sum(v.get("tokens", 0) for v in (d.get("models") or {}).values()),
    }
    # datasheet 附录的逐类计数与模型构成,同样逐项核对
    rc = (red.get("counts") or {})
    models = d.get("models") or {}
    expect.update({
        "1,807,473": rc.get("USERNAME_PATH"), "16,073": rc.get("SECRET_ASSIGN"),
        "1,968": rc.get("SECRET_CLI"), "1,420": rc.get("CN_ID"),
        "1,105": rc.get("DB_URI_CREDS"), "1,077": rc.get("JWT"),
        "811": rc.get("BEARER"), "385": rc.get("URL_BASIC_AUTH"),
        "150": rc.get("GITLAB_TOKEN"), "97": rc.get("PRIVATE_KEY"), "9": rc.get("EMAIL"),
        "254": sum(rc.get(k, 0) for k in ["GCP_KEY","HF_TOKEN","OPENAI_KEY","STRIPE_KEY",
                                          "SLACK_TOKEN","AWS_AKID","AWS_SECRET","SECRET_HEADER"]),
        "8,282": (models.get("claude-opus-4-8") or {}).get("sessions"),
        "7,926": (models.get("claude-opus-4-7") or {}).get("sessions"),
        "5,360": (models.get("claude-opus-4-6") or {}).get("sessions"),
        "3,943": (models.get("claude-sonnet-4-6") or {}).get("sessions"),
        "907":   (models.get("claude-fable-5") or {}).get("sessions"),
        "578":   (models.get("claude-haiku-4-5") or {}).get("sessions"),
        "867M":  (models.get("claude-opus-4-8") or {}).get("tokens"),
        "816M":  (models.get("claude-opus-4-7") or {}).get("tokens"),
        "354M":  (models.get("claude-opus-4-6") or {}).get("tokens"),
        "315M":  (models.get("claude-sonnet-4-6") or {}).get("tokens"),
        "122M":  (models.get("claude-fable-5") or {}).get("tokens"),
        "24.7M": (models.get("claude-haiku-4-5") or {}).get("tokens"),
    })
    for claim, actual in expect.items():
        checked += 1
        a = float(actual) if actual is not None else None
        tol = 0.01 if not str(claim).strip().endswith(("B", "M", "k")) else 0.005
        if not close(norm(claim), a, tol=tol, exact=is_count(claim)):
            kind = ("计数(需精确)" if is_count(claim)
                    else f"近似量(容差{tol*100:.1f}%)")
            fails.append(f"[D] 论文写 {claim},源文件实际 {actual} —— {kind}")
else:
    skipped.append("datasheet_stats.json 不存在")

# ---- 从 smoke500 原始结果重算(不信摘要,直接算)----
sm = ROOT / "experiments/results/smoke500_7b.jsonl"
if sm.exists():
    import statistics
    rows = [json.loads(l) for l in sm.open()]
    real = [r["suff_real"] for r in rows]
    mis = [r["suff_mismatch"] for r in rows]
    n = len(rows)
    recomputed = {
        "500": n,
        "+2.84": statistics.median(real),
        "$-4.10$": statistics.median(mis),
        "80.6\\%": sum(1 for a, b in zip(real, mis) if a > b) / n * 100,
    }
    for claim, actual in recomputed.items():
        checked += 1
        if not close(norm(claim), actual, tol=0.03):
            fails.append(f"[D] EVIDENCE 登记 {claim},重算得 {actual:.3f}")
        if claimed_in_paper(claim) is None:
            fails.append(f"[D] 重算值 {actual:.2f} 对应的声称 {claim} "
                         f"未在 main.tex 中找到 —— 论文数字可能被改动过")
else:
    skipped.append("smoke500_7b.jsonl 不存在")

# ---- 从 hindsight.jsonl 重算 T1/T2 核心论据 ----
hs = ROOT / "experiments/results/hindsight.jsonl"
if hs.exists():
    rows = [json.loads(l) for l in hs.open()]
    n = len(rows); TH = 3.0
    rec = {
        "49.6\\%": sum(1 for r in rows if r["suff_hindsight"] >= TH) / n * 100,
        "1.6\\%":  sum(1 for r in rows if r["suff_clean"] >= TH) / n * 100,
        "44.0\\%": sum(1 for r in rows if r["suff_hindsight"] >= TH
                        and r["nat_hindsight"] >= -2.8) / n * 100,
        "$-2.73$": __import__("statistics").median(r["nat_real"] for r in rows),
        "$-1.08$": __import__("statistics").median(r["nat_clean"] for r in rows),
    }
    for claim, actual in rec.items():
        checked += 1
        if not close(norm(claim), actual, tol=0.03):
            fails.append(f"[D-hindsight] EVIDENCE 登记 {claim},重算得 {actual:.3f}")
        if claimed_in_paper(claim) is None:
            fails.append(f"[D-hindsight] 重算值 {actual:.2f} 对应的声称 {claim} "
                         f"未在 main.tex 中找到 —— 论文数字可能被改动过")
else:
    skipped.append("hindsight.jsonl 不存在")

# ---- 从 scores_calib.jsonl 重算校准结论 ----
sc = ROOT / "experiments/results/scores_calib.jsonl"
if sc.exists():
    import statistics
    rows = [json.loads(l) for l in sc.open()]
    leaked = sum(1 for r in rows for x in r["results"] if x.get("leak"))
    total_c = sum(len(r["results"]) for r in rows)
    best = [max((x["s_suff"] for x in r["results"] if not x.get("leak")), default=None)
            for r in rows]
    best = [b for b in best if b is not None]
    singles = [x["s_suff"] for r in rows for x in r["results"]]
    rec2 = {
        "1.0\\%": leaked / total_c * 100,
        "+4.02":   statistics.median(best) - statistics.median(singles),
    }
    for claim, actual in rec2.items():
        checked += 1
        if not close(norm(claim), actual, tol=0.05):
            fails.append(f"[D-calib] EVIDENCE 登记 {claim},重算得 {actual:.3f}")
        if claimed_in_paper(claim) is None:
            fails.append(f"[D-calib] 重算值 {actual:.2f} 对应的声称 {claim} "
                         f"未在 main.tex 中找到 —— 论文数字可能被改动过")
else:
    skipped.append("scores_calib.jsonl 不存在")

# ---- 主实验:逐臂从 eval JSON 核对 M1/M2 ----
ARMS = {
    "base.json":       ("12.663", "$-1.2442$"),
    "B_minus_s1.json": ("9.516",  "$-0.8982$"),
    "B_minus_s2.json": ("9.463",  "$-0.9176$"),
    "B_s1.json":       ("7.738",  "$-0.8924$"),
    "B_s2.json":       ("7.745",  "$-0.8898$"),
    "R_s1.json":       ("7.765",  "$-0.8525$"),
    "R_s2.json":       ("7.780",  "$-0.8594$"),
    "Rdense_s1.json":  ("7.731",  "$-0.8575$"),
}
for fn, (m1c, m2c) in ARMS.items():
    fp = ROOT / "experiments/results" / fn
    if not fp.exists():
        skipped.append(f"{fn} 不存在"); continue
    d = json.loads(fp.read_text())
    for claim, actual, name in ((m1c, d.get("M1_think_ppl"), "M1"),
                                (m2c, d.get("M2_action_lp_per_tok"), "M2")):
        checked += 1
        if not close(norm(claim), actual, tol=0.002):
            fails.append(f"[D-arms] {fn} {name}: 论文写 {claim},JSON 为 {actual}")
        if claimed_in_paper(claim) is None:
            fails.append(f"[D-arms] {fn} {name} 的值 {claim} 未出现在 main.tex")

# ---- 附录 worked example:六个候选分数逐一核对 ----
sc2 = ROOT / "experiments/results/scores_calib.jsonl"
if sc2.exists():
    KEY = "43342f99-c6eb-85a2-8343-2cc7c2bf9ed6:59:0"
    for line in sc2.open():
        d = json.loads(line)
        if d.get("key") != KEY: continue
        for i, r in enumerate(d["results"], 1):
            claim = f"{r['s_suff']:+.2f}".replace("+-", "-")
            checked += 1
            if claimed_in_paper(claim.replace("+", "$+").replace("-", "$-") + "$") is None \
               and claimed_in_paper(claim) is None:
                fails.append(f"[D-example] 候选{i} 的分数 {claim} 未出现在 main.tex")
        break

# ---- 文本类来源:弱校验,但覆盖剩余数字 ----
# 无法重算的(摘要 md、txt),至少要求该数字字面出现在其来源文件中。
# 弱校验远好过零校验:它拦得住"登记表里写个源文件里根本没有的数"。
EV = ROOT / "paper" / "EVIDENCE.md"
if EV.exists():
    grep_checked = grep_missing = 0
    weak = []
    for line in EV.read_text().splitlines():
        if not line.strip().startswith("|"): continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3: continue
        claim = cells[0].replace("`", "").strip()
        if not re.search(r"\d", claim): continue
        # 外部引用的数字(来自他人论文)无法从我们的产物重算 ——
        # 它们的正确性由引用本身承担,这里只要求登记了文献出处。
        # 强行要求"能在 bib 里字面匹配"会误伤,且并不检验任何真实性。
        if "bib" in cells[2] or "外部" in cells[1] or "非本文测量" in cells[1]:
            continue
        srcs = [t for t in re.findall(r"`([^`]+)`", cells[2]) if "/" in t]
        if not srcs: continue
        src = ROOT / srcs[0]
        if not src.exists() or src.suffix == ".json": continue   # json 已由上面强校验
        body = src.read_text(errors="replace")
        bare = claim.replace("\\%", "").replace("$", "").replace("\\", "").strip()
        core = bare.lstrip("+-")
        if not core: continue
        # 自欺检测:短数字(<=3 字符)在任何文本里都能匹配,grep 通过毫无意义。
        # 与其报一个虚假的"已核验",不如如实标为未核验 —— 失效的检查比没有检查更危险,
        # 因为它制造"已经查过"的错觉。
        if len(core.replace(".", "").replace(",", "")) <= 3:
            weak.append(f"{claim} <- {srcs[0]}")
            continue
        grep_checked += 1
        if core not in body and core.replace(",", "") not in body.replace(",", ""):
            grep_missing += 1
            fails.append(f"[D-text] {claim} 未出现在其来源 {srcs[0]} 中")
    checked += grep_checked
    print(f"(其中 {grep_checked} 个为文本来源的字面校验)")
    if weak:
        print(f"  未核验 {len(weak)} 个:数字过短,字面匹配无鉴别力,需人工核对:")
        for w in weak: print(f"      {w}")

print(f"机器核验 {checked} 个数字")
for s in skipped: print("  SKIP " + s)
for f in fails: print("  FAIL " + f)
print("PASS" if not fails else f"\n{len(fails)} 项对不上 —— 禁止提交")
sys.exit(1 if fails else 0)
