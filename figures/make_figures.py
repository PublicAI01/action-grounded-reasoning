#!/usr/bin/env python3
"""生成论文全部图表。铁律:所有数值必须从 experiments/results/ 下的真实文件读取,
脚本内不得出现任何硬编码的实验数值。闸门 F 关会检查这一点。

每个图输出:  figX.pdf(插图用) + figX.data.json(该图所有数值的溯源记录)
后者供 verify_figures.py 交叉核验,确保图上画的就是数据里的。
"""
import json, sys, pathlib, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
OUT = ROOT / "paper" / "figures"
plt.rcParams.update({"font.size": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
INK, ACC, WARN = "#2b2b2b", "#3b6ea5", "#b5442e"

def dump(name, data):
    (OUT / f"{name}.data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))

# ── 图1:充分性判据偏爱后见之明(核心发现)──────────────────
rows = [json.loads(l) for l in (RES / "hindsight.jsonl").open()]
n, TH = len(rows), 3.0
pass_rate = {
    "honest\n(blind)":     sum(1 for r in rows if r["suff_clean"] >= TH) / n * 100,
    "hindsight\n(sees action)": sum(1 for r in rows if r["suff_hindsight"] >= TH) / n * 100,
    "real thinking":       sum(1 for r in rows if r["suff_real"] >= TH) / n * 100,
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.28))
ks = list(pass_rate)
bars = ax1.bar(ks, [pass_rate[k] for k in ks],
               color=[ACC, WARN, "#7a7a7a"], width=0.6)
for b, k in zip(bars, ks):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2,
             f"{pass_rate[k]:.1f}%", ha="center", fontsize=7.5)
ax1.set_ylabel("pass rate (%)")
ax1.set_ylim(0, max(pass_rate.values()) * 1.28)
ax1.set_title("(a) the filter prefers cheating", fontsize=8, loc="left")

med = {k: statistics.median(r[c] for r in rows)
       for k, c in [("honest", "suff_clean"), ("hindsight", "suff_hindsight"),
                    ("real", "suff_real")]}
ax2.barh(list(med), list(med.values()),
         color=[ACC, WARN, "#7a7a7a"], height=0.55)
ax2.axvline(0, color=INK, lw=0.6)
for i, (k, v) in enumerate(med.items()):
    inside = v < 0
    ax2.text(v + 0.35, i, f"{v:+.2f}", va="center", ha="left", fontsize=7.5,
             color="white" if inside else INK)
ax2.set_xlabel("median sufficiency (nats)")
lo, hi = min(med.values()), max(med.values())
ax2.set_xlim(lo * 1.10, hi + (hi - lo) * 0.42)
ax2.invert_yaxis()
ax2.set_title("(b) and scores them higher", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_hindsight.pdf"); plt.close(fig)
dump("fig_hindsight", {"source": "experiments/results/hindsight.jsonl", "n": n,
                       "threshold_nats": TH, "pass_rate_pct": pass_rate, "median_nats": med})

# ── 图2:尺子有效性(真实 vs 错位推理的充分性分布)────────────
sm = [json.loads(l) for l in (RES / "smoke500_7b.jsonl").open()]
real = [r["suff_real"] for r in sm]
mis = [r["suff_mismatch"] for r in sm]
fig, ax = plt.subplots(figsize=(2.6, 1.3))
bins = [x * 2 - 20 for x in range(21)]
ax.hist(mis, bins=bins, color=WARN, alpha=0.55, label="mismatched")
ax.hist(real, bins=bins, color=ACC, alpha=0.55, label="own (correct)")
ax.axvline(0, color=INK, lw=0.6, ls=":")
ax.set_xlabel("$\\Delta$ log P(true action) (nats)")
ax.set_ylabel("positions")
ax.legend(frameon=False, fontsize=7)
win = sum(1 for a, b in zip(real, mis) if a > b) / len(sm) * 100
ax.set_title(f"pairwise win rate {win:.1f}%", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_ruler.pdf"); plt.close(fig)
dump("fig_ruler", {"source": "experiments/results/smoke500_7b.jsonl", "n": len(sm),
                   "median_real": statistics.median(real),
                   "median_mismatch": statistics.median(mis), "win_rate_pct": win})

# ── 图3:语料的推理密度(动机图)────────────────────────
ds = json.loads((RES / "datasheet_stats.json").read_text())
th = ds["thinking"]; pos = th["by_position"]
order = [("last", "final"), ("second_last", "2nd-last"),
         ("3rd_to_5th", "3rd–5th"), ("earlier", "earlier")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 1.5))
dens = th["nonempty_per_turn"] * 100
a1.bar(["non-empty\nreasoning", "no reasoning"], [dens, 100 - dens],
       color=[ACC, "#d9d9d9"], width=0.55)
a1.text(0, dens + 2.5, f"{dens:.1f}", ha="center", fontsize=8, color=ACC)
a1.text(1, 100 - dens + 2.5, f"{100-dens:.1f}", ha="center", fontsize=8)
a1.set_ylabel("per 100 assistant turns"); a1.set_ylim(0, 108)
a1.set_title("(a) actions vastly outnumber reasons", fontsize=8, loc="left")
labs = [l for _, l in order]
vals = [pos[k]["empty_rate"] * 100 for k, _ in order]
a2.plot(labs, vals, "o-", color=WARN, lw=1.3, ms=4)
for i, (x, v) in enumerate(zip(labs, vals)):
    # 标签统一放在点的上方并留出间距;首尾点向内侧偏,避免压到轴与刻度
    dx = 9 if i == 0 else (-9 if i == len(vals) - 1 else 0)
    a2.annotate(f"{v:.1f}%", (x, v), textcoords="offset points",
                xytext=(dx, 7), ha="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
a2.set_ylabel("empty blocks (%)"); a2.set_xlabel("turn position")
a2.margins(x=0.14)
a2.set_ylim(min(vals) - 4, max(vals) + 8)
a2.set_title("(b) earlier turns are stripped more", fontsize=8, loc="left")
fig.tight_layout(w_pad=3.0); fig.savefig(OUT / "fig_density.pdf"); plt.close(fig)
dump("fig_density", {"source": "experiments/results/datasheet_stats.json",
                     "nonempty_per_100_turns": dens,
                     "empty_rate_by_position_pct": dict(zip(labs, vals))})

# ── 图4:主实验结果 ──────────────────────────────────
import glob as _g
arms = [("B$^-$", ["B_minus_s1", "B_minus_s2"]), ("B", ["B_s1", "B_s2"]),
        ("R", ["R_s1", "R_s2"]), ("R$_{dense}$", ["Rdense_s1"])]
m1, m2, labs = [], [], []
for lab, files in arms:
    vs1 = [json.loads((RES / f"{f}.json").read_text())["M1_think_ppl"] for f in files
           if (RES / f"{f}.json").exists()]
    vs2 = [json.loads((RES / f"{f}.json").read_text())["M2_action_lp_per_tok"] for f in files
           if (RES / f"{f}.json").exists()]
    if vs1: labs.append(lab); m1.append(vs1); m2.append(vs2)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(5.5, 1.20))
x = range(len(labs))
b1.bar(x, [sum(v)/len(v) for v in m1], color=[ACC if l != "R" else WARN for l in labs], width=0.55)
for i, v in enumerate(m1):
    b1.plot([i]*len(v), v, "o", color=INK, ms=2.5)
b1.set_xticks(list(x)); b1.set_xticklabels(labs)
b1.set_ylabel("M1 think-ppl $\\downarrow$"); b1.set_ylim(min(min(v) for v in m1)-0.6, max(max(v) for v in m1)+0.5)
b1.set_title("(a) reasoning supervision matters", fontsize=7, loc="left")
b2.bar(x, [sum(v)/len(v) for v in m2], color=[ACC if l != "R" else WARN for l in labs], width=0.55)
for i, v in enumerate(m2):
    b2.plot([i]*len(v), v, "o", color=INK, ms=2.5)
b2.set_xticks(list(x)); b2.set_xticklabels(labs)
b2.set_ylabel("M2 action-lp $\\uparrow$")
b2.set_ylim(min(min(v) for v in m2)-0.01, max(max(v) for v in m2)+0.008)
b2.set_title("(b) synthesis adds decision signal", fontsize=7, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_results.pdf"); plt.close(fig)
dump("fig_results", {"source": "experiments/results/B_s1.json",
                     "arms": {l: {"M1": v1, "M2": v2} for l, v1, v2 in zip(labs, m1, m2)}})


print("生成 4 张图 + 溯源 json")
for f in sorted(OUT.glob("fig_*.pdf")): print(f"  {f.name}")
