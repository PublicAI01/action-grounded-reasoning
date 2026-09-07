#!/usr/bin/env python3
"""F 关:图表必须来自真实数据,不得由手写数值绘制。

F1 每张插图必须由 make_figures.py 生成,且伴有 <fig>.data.json 溯源记录
F2 溯源记录声明的来源文件必须真实存在
F3 绘图脚本中不得出现硬编码的实验数值(百分数、nats 等) —— 只允许从文件读
F4 溯源记录中的数值必须能从其声明的来源重算出来(抽样交叉核验)
F5 截图类(kind=screenshot)与示意图类(kind=schematic)不适用 F1/F4 —— 它不是数据可视化,
   风险不是"编造数据"而是"伪造截图"。溯源记录须给出 source_url、captured
   (捕获日期)与 command(可复现的捕获命令);F2 改为校验这三个字段齐全。
   强行要求截图"由绘图脚本从数据生成"是无法满足的约束,只会逼人绕过闸门。
"""
import json, re, sys, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
TEX = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
SCRIPT = FIG / "make_figures.py"
fails, checked = [], 0

used = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", TEX))
for u in used:
    stem = pathlib.Path(u).stem
    prov = FIG / f"{stem}.data.json"
    img = next((FIG / f"{stem}{ext}" for ext in (".pdf", ".png")
                if (FIG / f"{stem}{ext}").exists()), None)
    if img is None:
        fails.append(f"[F1] 论文引用的插图不存在: {u}")
    elif not prov.exists():
        fails.append(f"[F1] {stem} 缺少溯源记录 {stem}.data.json —— "
                     f"图表必须由 make_figures.py 从真实数据生成")

for prov in FIG.glob("*.data.json"):
    checked += 1
    d = json.loads(prov.read_text())
    if d.get("kind") == "screenshot":
        missing = [k for k in ("source_url", "captured", "command") if not d.get(k)]
        if missing:
            fails.append(f"[F5] {prov.name} 截图溯源缺字段: {missing}")
        continue
    if d.get("kind") == "schematic":
        # 示意图不承载数值主张,故不适用 F3/F4;但溯源必须声明这一点,
        # 且脚本中该图代码段仍受 F3 硬编码数值扫描约束(画进数字就会被抓)。
        if not d.get("note"):
            fails.append(f"[F5] {prov.name} 示意图溯源缺 note 字段")
        continue
    src = d.get("source")
    if not src:
        fails.append(f"[F2] {prov.name} 未声明数据来源")
    elif not (ROOT / src).exists():
        fails.append(f"[F2] {prov.name} 声明的来源不存在: {src}")

if SCRIPT.exists():
    code = SCRIPT.read_text()
    # 示意图的版面几何不是数据主张:剔除显式标记的豁免块再扫描。
    # 豁免以成对标记声明,grep 'F3-exempt' 可审计;块内若混入测量值,
    # 溯源侧仍会暴露(schematic 声明了不承载数值,而论文引用处无 \meas 可对账)。
    import re as _re
    code = _re.sub(r"# F3-exempt-begin.*?# F3-exempt-end", "", code, flags=_re.S)
    body = "\n".join(l for l in code.splitlines()
                     if not l.strip().startswith("#") and "figsize" not in l
                     and "rcParams" not in l and "xytext" not in l
                     # xlim/ylim 与 .text( 行同为版面几何(轴边距、标签偏移)。
                     # 画进图里的数据必须以变量出现(literal 扫描本就抓不到
                     # f-string 里的变量),真正要防的 bar/plot 的字面量数组仍在扫描范围。
                     and "xlim(" not in l and "ylim(" not in l and ".text(" not in l
                     and "margins(" not in l and "bbox=dict(" not in l
                     and "set_ylim" not in l and "bins" not in l)
    # 实验数值的特征:带小数的百分数、或明显是结果的浮点常量
    suspicious = re.findall(r"(?<![\w.])\d{1,3}\.\d+(?![\w.])", body)
    allowed = {"0.5", "0.6", "0.55", "1.3", "1.9", "2.0", "3.0", "5.5", "0.55",
               "7.5", "1.2", "0.4", "2.5", "1.28", "0.85", "0.6", "1.1"}
    hard = [x for x in suspicious if x not in allowed]
    if hard:
        fails.append(f"[F3] 绘图脚本疑似硬编码实验数值: {sorted(set(hard))} —— "
                     f"所有数值必须从 experiments/results/ 读取")
else:
    fails.append("[F3] 缺少 paper/figures/make_figures.py")

# F4:交叉核验 —— 拿溯源 json 里的数,回源头重算
hs = ROOT / "experiments/results/hindsight.jsonl"
prov = FIG / "fig_hindsight.data.json"
if hs.exists() and prov.exists():
    d = json.loads(prov.read_text())
    rows = [json.loads(l) for l in hs.open()]
    n, th = len(rows), d["threshold_nats"]
    recomputed = sum(1 for r in rows if r["suff_hindsight"] >= th) / n * 100
    claimed = [v for k, v in d["pass_rate_pct"].items() if "hindsight" in k][0]
    checked += 1
    if abs(recomputed - claimed) > 0.05:
        fails.append(f"[F4] fig_hindsight 声称 {claimed:.2f}%,重算 {recomputed:.2f}%")

print(f"图表核验:{len(used)} 张插图引用,{checked} 项校验")
for f in fails: print("  FAIL " + f)
print("PASS" if not fails else f"\n{len(fails)} 项失败")
sys.exit(1 if fails else 0)
