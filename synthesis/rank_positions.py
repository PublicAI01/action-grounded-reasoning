#!/usr/bin/env python3
"""位置价值排序:对每个位置算基座对真实动作的 base logprob。
低 = 基座没把握 = 补写推理最有价值(实测:仅 60% 位置连真实 thinking 都提升不了预测)。
输出按价值排序的位置清单,供生成阶段挑选,直接决定 API 与打分成本。
用法: rank_positions.py positions.jsonl arm.jsonl out.jsonl [N会话]
"""
import json, sys, os, statistics, collections

MODEL = os.environ.get("RULER_MODEL", "models/qwen7b-awq")
MAXLEN = int(os.environ.get("MAXLEN", "8192"))
CTX_CHARS = int(os.environ.get("CTX_CHARS", "9000"))

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

def main(pos_file, arm_file, out_file, n_sessions=300):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=MAXLEN, enable_prefix_caching=True,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")), dtype="float16", enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)

    def batch_lp(pairs):
        prompts, cuts = [], []
        for prefix, target in pairs:
            ids_p = tok.encode(prefix); ids_t = tok.encode(target)
            if len(ids_p) + len(ids_t) > MAXLEN - 8:
                prefix = tok.decode(ids_p[-(MAXLEN - 8 - len(ids_t)):]); ids_p = tok.encode(prefix)
            prompts.append(prefix + target); cuts.append(len(ids_p))
        outs = llm.generate(prompts, sp, use_tqdm=False)
        res = []
        for o, c in zip(outs, cuts):
            v = [list(d.values())[0].logprob for d in (o.prompt_logprobs or [])[c:] if d]
            res.append((sum(v), max(len(v), 1)))
        return res

    # 优先取带 thinking 的会话(B/B⁻/R 三臂需同一批会话,见 §19)
    by_sid = collections.defaultdict(list)
    has_think = collections.Counter()
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line)
            by_sid[p["sid"]].append(p)
            if p.get("has_real_thinking"): has_think[p["sid"]] += 1
    sids = [s for s in by_sid if has_think[s] > 0]
    sids.sort(key=lambda s: -has_think[s])
    sids = sids[:n_sessions]
    recs = {}
    with open(arm_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("session_id") in set(sids): recs[r["session_id"]] = r
    positions = [p for s in sids if s in recs for p in by_sid[s]]
    print(f"sessions={len(recs)} positions={len(positions)}", flush=True)
    done = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                try: done.add(json.loads(line)["key"])
                except Exception: pass
    positions = [p for p in positions
                 if f'{p["sid"]}:{p["msg_idx"]}:{p["block_idx"]}' not in done]
    print(f"to score: {len(positions)} (resumed {len(done)})", flush=True)
    out = open(out_file, "a"); vals = []
    B = 24
    for i in range(0, len(positions), B):
        chunk = positions[i:i+B]
        pairs = []
        for p in chunk:
            ctx = render_context(recs[p["sid"]], p["msg_idx"], CTX_CHARS)
            act = f'调用工具 {p["action_name"]},参数: {p["action_input"][:400]}'
            pairs.append((ctx + "\n[assistant 接下来的动作] ", act))
        res = batch_lp(pairs)
        for p, (lp, nt) in zip(chunk, res):
            row = {"key": f'{p["sid"]}:{p["msg_idx"]}:{p["block_idx"]}',
                   "sid": p["sid"], "action": p["action_name"],
                   "base_lp": round(lp, 2), "ntok": nt,
                   "base_per_tok": round(lp / nt, 3),
                   "has_real_thinking": bool(p.get("has_real_thinking"))}
            vals.append(row["base_per_tok"]); out.write(json.dumps(row) + "\n")
        if len(vals) % 480 < B:
            out.flush()
            print(f"[rank] {len(vals)}/{len(positions)} median={statistics.median(vals):.3f}", flush=True)
    out.close()
    v = sorted(vals); n = len(v)
    print(f"\n=== 位置价值分布 (base_per_tok, 越低=越没把握=越值得补) n={n} ===")
    for q, lbl in ((0.1,"p10"),(0.25,"p25"),(0.5,"中位"),(0.75,"p75"),(0.9,"p90")):
        print(f"  {lbl}: {v[int(n*q)]:.3f}")
    for th in (-2.0, -1.5, -1.2, -1.0):
        k = sum(1 for x in v if x <= th)
        print(f"  阈值 {th}: 入选 {k:,} ({k/n*100:.1f}%)  预计生成成本 ${k*0.00072:.2f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 300)
