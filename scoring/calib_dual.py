#!/usr/bin/env python3
"""双判据联合校准 + 位置价值分析(一次扫描,三产出)。

对每个锚点位置 t 计算四种"推理"下的分数:
  own    = 该位置真实思考           期望:充分性高、自然度高
  future = 同一会话中更晚位置的真实思考(人造"后见之明":提到当时还不该知道的东西)
           期望:充分性也高(它提到了答案!)但自然度低 ← 这正是第二判据要抓的
  random = 别的会话的思考           期望:两者都低
  (base) = 不插推理                 base logprob 本身 = 该位置的"补写价值"

产出:
  1) 非透视性判据的判别力(future vs own 的自然度差)
  2) 单判据失效的证据(future 的充分性 ≈ own → 光看充分性会收下后见之明)
  3) 位置价值分布(base 越低 = 基座越没把握 = 越值得花钱补写)
"""
import json, sys, os, random, statistics, collections

MODEL = os.environ.get("RULER_MODEL", "models/qwen7b-awq")
MAXLEN = int(os.environ.get("MAXLEN", "8192"))
CTX_CHARS = int(os.environ.get("CTX_CHARS", "12000"))

def render_context(rec, msg_idx, max_chars):
    parts = []
    for m in (rec.get("messages") or [])[:msg_idx]:
        role = m.get("role"); c = m.get("content")
        if isinstance(c, str): parts.append(f"[{role}] {c}")
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict): continue
                t = b.get("type")
                if t == "text": parts.append(f"[{role}] {b.get('text','')}")
                elif t == "tool_use":
                    try: parts.append(f"[{role} 调用 {b.get('name')}] {json.dumps(b.get('input'),ensure_ascii=False)[:400]}")
                    except Exception: pass
                elif t == "tool_result":
                    cc = b.get("content")
                    txt = cc if isinstance(cc, str) else " ".join(x.get("text","") for x in cc if isinstance(x,dict)) if isinstance(cc,list) else ""
                    parts.append(f"[工具返回] {txt[:800]}")
    return "\n".join(parts)[-max_chars:]

def main(pos_file, arm_file, out_file, n_anchor=400):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=MAXLEN, enable_prefix_caching=True,
              gpu_memory_utilization=0.85, dtype="float16", enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)

    def batch_logprob(pairs):
        prompts, cuts = [], []
        for prefix, target in pairs:
            ids_p = tok.encode(prefix); ids_t = tok.encode(target)
            if len(ids_p) + len(ids_t) > MAXLEN - 8:
                keep = MAXLEN - 8 - len(ids_t)
                prefix = tok.decode(ids_p[-keep:]); ids_p = tok.encode(prefix)
            prompts.append(prefix + target); cuts.append(len(ids_p))
        outs = llm.generate(prompts, sp, use_tqdm=False)
        res = []
        for o, cut in zip(outs, cuts):
            vals = [list(d.values())[0].logprob for d in (o.prompt_logprobs or [])[cut:] if d]
            res.append((sum(vals), max(len(vals), 1)))
        return res

    random.seed(11)
    by_sid = collections.defaultdict(list)
    all_sums = []
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line)
            if p.get("has_real_thinking") and p.get("real_thinking_summary"):
                by_sid[p["sid"]].append(p); all_sums.append(p["real_thinking_summary"])
    # 只保留同一会话内有 >=2 个锚点的,才能构造 future 负样本
    usable = [(sid, ps) for sid, ps in by_sid.items() if len(ps) >= 2]
    random.shuffle(usable)
    picks = []
    for sid, ps in usable:
        ps.sort(key=lambda x: x["msg_idx"])
        i = random.randrange(0, len(ps) - 1)
        picks.append((ps[i], ps[-1]))          # (当前位置, 该会话最后一个锚点=最强后见之明)
        if len(picks) >= n_anchor: break
    need = {p["sid"] for p, _ in picks}
    recs = {}
    with open(arm_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("session_id") in need: recs[r["session_id"]] = r
    print(f"picks={len(picks)} sessions={len(recs)}", flush=True)
    out = open(out_file, "w"); rows = []
    B = 8
    for i in range(0, len(picks), B):
        chunk = [(p, fut) for p, fut in picks[i:i+B] if p["sid"] in recs]
        pairs, metas = [], []
        for p, fut in chunk:
            ctx = render_context(recs[p["sid"]], p["msg_idx"], CTX_CHARS)
            act = f'调用工具 {p["action_name"]},参数: {p["action_input"][:400]}'
            own = p["real_thinking_summary"][:800]
            future = fut["real_thinking_summary"][:800]
            rnd = random.choice(all_sums)[:800]
            head = ctx + "\n[assistant 思考] "
            pairs += [
                (ctx + "\n[assistant 接下来的动作] ", act),                                  # 0 base
                (head + own    + "\n[assistant 接下来的动作] ", act),                        # 1 suff own
                (head + future + "\n[assistant 接下来的动作] ", act),                        # 2 suff future
                (head + rnd    + "\n[assistant 接下来的动作] ", act),                        # 3 suff random
                (head, own), (head, future), (head, rnd),                                    # 4-6 naturalness
            ]
            metas.append((p, fut))
        res = batch_logprob(pairs)
        for j, (p, fut) in enumerate(metas):
            k = 7 * j
            base = res[k][0]
            row = {
                "key": f'{p["sid"]}:{p["msg_idx"]}',
                "base": round(base, 2), "base_ntok": res[k][1],
                "suff_own":    round(res[k+1][0] - base, 2),
                "suff_future": round(res[k+2][0] - base, 2),
                "suff_random": round(res[k+3][0] - base, 2),
                "nat_own":     round(res[k+4][0] / res[k+4][1], 3),
                "nat_future":  round(res[k+5][0] / res[k+5][1], 3),
                "nat_random":  round(res[k+6][0] / res[k+6][1], 3),
            }
            rows.append(row); out.write(json.dumps(row) + "\n")
        if len(rows) % 40 < B and rows:
            print(f"[{len(rows)}/{len(picks)}] "
                  f"suff own={statistics.median(r['suff_own'] for r in rows):.2f} "
                  f"future={statistics.median(r['suff_future'] for r in rows):.2f} | "
                  f"nat own={statistics.median(r['nat_own'] for r in rows):.2f} "
                  f"future={statistics.median(r['nat_future'] for r in rows):.2f}", flush=True)
    out.close()
    n = len(rows)
    med = lambda k: statistics.median(r[k] for r in rows)
    print(f"\n=== RESULT n={n} ===")
    print(f"充分性  own={med('suff_own'):7.2f}  future={med('suff_future'):7.2f}  random={med('suff_random'):7.2f}")
    print(f"自然度  own={med('nat_own'):7.3f}  future={med('nat_future'):7.3f}  random={med('nat_random'):7.3f}")
    fut_suff_ok = sum(1 for r in rows if r["suff_future"] >= 3.0) / n
    own_suff_ok = sum(1 for r in rows if r["suff_own"] >= 3.0) / n
    print(f"\n单判据(仅充分性>=3.0)通过率:  own={own_suff_ok*100:.1f}%  future(后见之明)={fut_suff_ok*100:.1f}%")
    wins_nat = sum(1 for r in rows if r["nat_own"] > r["nat_future"]) / n
    print(f"自然度判别力(own > future 的比例) = {wins_nat*100:.1f}%   (>65% = 第二判据有效)")
    bases = sorted(r["base"] / max(r["base_ntok"],1) for r in rows)
    print(f"\n位置价值(base 每token logprob)分位: "
          f"p10={bases[n//10]:.2f} 中位={statistics.median(bases):.2f} p90={bases[9*n//10]:.2f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 400)
