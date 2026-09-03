#!/usr/bin/env python3
"""组装训练四臂:B⁻ / B / R / R-dense。
输入:300会话轨迹 + 生成的候选 + (可选)尺子打分
输出:LLaMA-Factory 兼容 jsonl(sharegpt 风格),每臂一个文件 + manifest。

臂定义(§19 定稿,同一批会话,仅推理监督密度不同):
  B_minus : 剥光全部 thinking            (0% 覆盖)
  B       : 保留原生 thinking            (~37% 覆盖,本子集)
  R       : B + 在缺失位置补写(经筛选)   (目标 ~100%)
  R_dense : B + 在所有缺失位置补写(不筛) (对照:全补是否更差)
"""
import json, sys, os, collections

def strip_thinking(msg):
    c = msg.get("content")
    if not isinstance(c, list): return msg
    return {**msg, "content": [b for b in c if not (isinstance(b, dict) and b.get("type") == "thinking")]}

def insert_thinking(msg, text):
    c = msg.get("content")
    if not isinstance(c, list): return msg
    keep = [b for b in c if not (isinstance(b, dict) and b.get("type") == "thinking")]
    return {**msg, "content": [{"type": "thinking", "thinking": text}] + keep}

def to_sharegpt(rec):
    """转成训练框架吃的格式:system + 多轮对话,assistant 段计 loss。"""
    sys_txt = rec.get("system")
    if isinstance(sys_txt, list):
        sys_txt = "\n".join(b.get("text","") for b in sys_txt if isinstance(b, dict))
    conv = []
    for m in rec.get("messages") or []:
        role = m.get("role"); c = m.get("content")
        if isinstance(c, str):
            conv.append({"from": "human" if role == "user" else "gpt", "value": c}); continue
        if not isinstance(c, list): continue
        parts = []
        for b in c:
            if not isinstance(b, dict): continue
            t = b.get("type")
            if t == "text": parts.append(b.get("text") or "")
            elif t == "thinking":
                th = (b.get("thinking") or "").strip()
                if th: parts.append(f"<thinking>{th}</thinking>")
            elif t == "tool_use":
                try: parts.append(f"<tool_use name=\"{b.get('name')}\">{json.dumps(b.get('input'), ensure_ascii=False)}</tool_use>")
                except Exception: pass
            elif t == "tool_result":
                cc = b.get("content")
                txt = cc if isinstance(cc, str) else " ".join(x.get("text","") for x in cc if isinstance(x, dict)) if isinstance(cc, list) else ""
                parts.append(f"<tool_result>{txt}</tool_result>")
        val = "\n".join(p for p in parts if p)
        if not val: continue
        conv.append({"from": "gpt" if role == "assistant" else "human", "value": val})
    # 保证 human/gpt 交替(框架要求),连续同角色合并
    merged = []
    for turn in conv:
        if merged and merged[-1]["from"] == turn["from"]:
            merged[-1]["value"] += "\n" + turn["value"]
        else: merged.append(turn)
    if merged and merged[0]["from"] != "human":
        merged.insert(0, {"from": "human", "value": "(开始)"})
    return {"system": (sys_txt or "")[:8000], "conversations": merged}

def main(arm_file, positions_file, cand_file, scores_file, outdir):
    os.makedirs(outdir, exist_ok=True)
    # 位置索引
    pos_of = {}
    sids = set()
    with open(positions_file) as f:
        for line in f:
            p = json.loads(line)
            pos_of[f'{p["sid"]}:{p["msg_idx"]}:{p["block_idx"]}'] = p
            sids.add(p["sid"])
    # 候选
    cands = {}
    if os.path.exists(cand_file):
        with open(cand_file) as f:
            for line in f:
                try:
                    d = json.loads(line); cands[d["key"]] = d["candidates"]
                except Exception: pass
    # 选取策略(2026-08-27 校准后定稿):
    #   实体泄漏 = 硬门槛(零误报已验证,泄漏一律拒)
    #   充分性   = 排序器,不设绝对阈值
    # 理由:充分性与防作弊直接对立 —— 要"充分"须点名具体对象,要"诚实"恰恰禁止点名。
    # 实测 DeepSeek 诚实候选中位 -7.05,绝对阈值会筛空,且系统性偏向后见之明
    # (作弊 49.6% vs 诚实 1.6%);相对排序无此偏置。
    best = {}
    stats_sel = collections.Counter()
    if scores_file and os.path.exists(scores_file):
        with open(scores_file) as f:
            for line in f:
                try: d = json.loads(line)
                except Exception: continue
                cands = d.get("results", [])
                clean = [r for r in cands if not r.get("leak")]
                stats_sel["positions"] += 1
                stats_sel["leaked_cands"] += sum(1 for r in cands if r.get("leak"))
                if not clean:
                    stats_sel["all_leaked_dropped"] += 1
                    continue
                best[d["key"]] = max(clean, key=lambda r: r.get("s_suff", -1e9))["r"]
                stats_sel["selected"] += 1
    print("选取统计:", dict(stats_sel))
    print(f"positions={len(pos_of):,} sessions={len(sids):,} candidates={len(cands):,} scored={len(best):,}")
    arms = {k: open(os.path.join(outdir, f"{k}.jsonl"), "w")
            for k in ("B_minus", "B", "R", "R_dense")}
    stats = collections.Counter()
    with open(arm_file) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec.get("session_id")
            if sid not in sids: continue
            msgs = rec.get("messages") or []
            def build(mode):
                out = []
                for mi, m in enumerate(msgs):
                    if m.get("role") != "assistant": out.append(m); continue
                    if mode == "B_minus": out.append(strip_thinking(m)); continue
                    if mode == "B": out.append(m); continue
                    has_real = any(isinstance(b, dict) and b.get("type") == "thinking"
                                   and (b.get("thinking") or "").strip()
                                   for b in (m.get("content") or []) if isinstance(b, dict))
                    if has_real: out.append(m); continue
                    key = None
                    for bi, b in enumerate(m.get("content") or []):
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            key = f'{sid}:{mi}:{bi}'; break
                    txt = None
                    if key:
                        if mode == "R": txt = best.get(key)
                        else: txt = best.get(key) or (cands.get(key) or [None])[0]
                    if txt:
                        out.append(insert_thinking(m, txt)); stats[f"{mode}_filled"] += 1
                    else: out.append(m)
                return {**rec, "messages": out}
            for k in arms:
                arms[k].write(json.dumps(to_sharegpt(build(k)), ensure_ascii=False) + "\n")
            stats["sessions"] += 1
    for h in arms.values(): h.close()
    print(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump({"stats": dict(stats), "arms": list(arms)}, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else None, sys.argv[5] if len(sys.argv) > 5 else "train_arms")
