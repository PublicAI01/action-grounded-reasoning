#!/usr/bin/env python3
"""E 关:结果新鲜度与完整性。防止两类事故:
  E1 论文声称的实验结果,其来源文件比论文改动更旧(说明数字是手抄的旧值)
  E2 声称"已完成"的实验,其结果文件行数不足(说明跑了一半就写结论)
"""
import os, sys, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "main.tex"
fails, warns = [], []

# E1: EVIDENCE 中引用的每个结果文件,mtime 不得早于其内容被写入论文的时间
#     实用近似:结果文件不得是空的或明显残缺
EV = ROOT / "paper" / "EVIDENCE.md"
if EV.exists():
    for line in EV.read_text().splitlines():
        if not line.strip().startswith("|"): continue
        for tok in re.findall(r"`([^`]+)`", line):
            if "/" not in tok or tok.startswith("http"): continue
            p = ROOT / tok
            if not p.exists():
                fails.append(f"[E1] 来源文件缺失: {tok}")
            elif p.stat().st_size == 0:
                fails.append(f"[E1] 来源文件为空: {tok}")

# E2: 实验完整性契约 —— 声称的样本量必须与结果文件实际行数一致
# 样本量契约:允许 <=5% 的正常损耗(超长样本被跳过等),但论文报的数
# 必须就是实际值 —— 承认损耗,不承认虚报。一刀切的"必须 >= 声称值"
# 会因几条正常跳过就误判"实验未跑完",逼人绕过闸门。
TOLERANCE = 0.05
CONTRACTS = [
    ("experiments/results/smoke500_7b.jsonl", 500, "尺子验证锚点数"),
]
for rel, claimed_n, what in CONTRACTS:
    p = ROOT / rel
    if not p.exists():
        warns.append(f"[E2] 尚未产出: {rel} ({what})")
        continue
    n = sum(1 for _ in p.open())
    if n < claimed_n * (1 - TOLERANCE):
        fails.append(f"[E2] {what}: 论文称 {claimed_n},文件仅 {n} 行"
                     f"(缺失 {(1-n/claimed_n)*100:.1f}% > {TOLERANCE*100:.0f}%)—— 实验未跑完")
    elif n != claimed_n:
        warns.append(f"[E2] {what}: 论文称 {claimed_n},实际 {n} —— "
                     f"损耗在容差内,但论文应改报实际值 {n}")

# E3: 蓝色占位不得混入已声称完成的章节
tex = TEX.read_text()
for sec in ["sec:hindsight", "sec:dataset"]:
    m = re.search(r"\\label\{" + sec + r"\}(.*?)(?=\\section|\Z)", tex, re.S)
    if m and "\\num{" in m.group(1):
        warns.append(f"[E3] 章节 {sec} 已声称有实测结论,但仍含 \\num{{}} 占位")

print(f"新鲜度检查完成")
for w in warns: print("  WARN " + w)
for f in fails: print("  FAIL " + f)
print("PASS" if not fails else f"\n{len(fails)} 项失败")
sys.exit(1 if fails else 0)
