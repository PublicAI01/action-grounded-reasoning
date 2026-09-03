#!/usr/bin/env python3
"""给生成的候选打分(双判据 + 实体泄漏),本地校准与云端全量共用。
输出每候选:s_suff(充分性)、leak(实体泄漏)、rank。支持分片与断点续跑。
用法: score_candidates.py cand.jsonl positions.jsonl arm.jsonl out.jsonl [N] [shard] [nshard]
"""
import json, sys, os, re, statistics, collections

MODEL = os.environ.get("RULER_MODEL", "models/qwen7b-awq")
MAXLEN = int(os.environ.get("MAXLEN", "8192"))
CTX_CHARS = int(os.environ.get("CTX_CHARS", "9000"))
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.85"))

ENT = re.compile(r"""
    [A-Za-z0-9_\-./]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|c|cc|cpp|h|hpp|json|yaml|yml|toml|cfg|ini|md|sh|sql|vue|css|html)\b
  | \b[a-z_][a-z0-9_]*_[a-z0-9_]+\b | \b[a-z]+[A-Z][A-Za-z0-9]+\b
  | \b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b | (?:line|行)\s*\d{2,}
""", re.X)
STOP = {"tool_use", "tool_result", "assistant", "user_id", "session_id"}
def ents(t): return {m.group(0).strip().lower() for m in ENT.finditer(t or "")} - STOP

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

def main(cand_file, pos_file, arm_file, out_file, limit=0, shard=0, nshard=1):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=MAXLEN, enable_prefix_caching=True,
              gpu_memory_utilization=GPU_UTIL, dtype="float16", enforce_eager=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)
    def lp(pairs):
        prompts, cuts = [], []
        for prefix, target in pairs:
            ip = tok.encode(prefix); it = tok.encode(target)
            if len(ip) + len(it) > MAXLEN - 8:
                prefix = tok.decode(ip[-(MAXLEN - 8 - len(it)):]); ip = tok.encode(prefix)
            prompts.append(prefix + target); cuts.append(len(ip))
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return [sum(list(d.values())[0].logprob for d in (o.prompt_logprobs or [])[c:] if d)
                for o, c in zip(outs, cuts)]
    pos = {}
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line); pos[f'{p["sid"]}:{p["msg_idx"]}:{p["block_idx"]}'] = p
    done = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                try: done.add(json.loads(line)["key"])
                except Exception: pass
    items = []
    with open(cand_file) as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            k = d["key"]
            if k in done or k not in pos: continue
            if nshard > 1 and hash(k) % nshard != shard: continue
            items.append(d)
    if limit: items = items[:limit]
    need = {pos[d["key"]]["sid"] for d in items}
    recs = {}
    with open(arm_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("session_id") in need: recs[r["session_id"]] = r
    items = [d for d in items if pos[d["key"]]["sid"] in recs]
    print(f"to score: {len(items)} positions (resumed {len(done)})", flush=True)
    out = open(out_file, "a"); allscores = []
    B = 6
    for i in range(0, len(items), B):
        chunk = items[i:i+B]; pairs = []; metas = []
        for d in chunk:
            p = pos[d["key"]]
            ctx = render_context(recs[p["sid"]], p["msg_idx"], CTX_CHARS)
            act = f'调用工具 {p["action_name"]},参数: {p["action_input"][:400]}'
            ctx_e = ents(ctx); act_e = ents(p["action_name"] + " " + p["action_input"])
            secret = act_e - ctx_e
            pairs.append((ctx + "\n[assistant 接下来的动作] ", act))
            for r in d["candidates"]:
                pairs.append((ctx + f"\n[assistant 思考] {r}\n[assistant 接下来的动作] ", act))
            metas.append((d, secret))
        vals = lp(pairs)
        vi = 0
        for d, secret in metas:
            base = vals[vi]; vi += 1
            res = []
            for r in d["candidates"]:
                s = vals[vi] - base; vi += 1
                leaked = sorted(ents(r) & secret)
                res.append({"r": r, "s_suff": round(s, 2), "leak": leaked})
                allscores.append(s)
            out.write(json.dumps({"key": d["key"], "base": round(base, 2),
                                  "results": res}, ensure_ascii=False) + "\n")
        if len(allscores) % 180 < B * 6:
            out.flush()
            print(f"[score] {i+len(chunk)}/{len(items)} pos  "
                  f"suff median={statistics.median(allscores):.2f}", flush=True)
    out.close()
    if allscores:
        v = sorted(allscores); n = len(v)
        print(f"\n=== 候选充分性分布 n={n} ===")
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            print(f"  p{int(q*100)}: {v[int(n*q)]:.2f}")
        for th in (0, 1, 2, 3, 5):
            print(f"  阈值 {th}: 通过 {sum(1 for x in v if x >= th)/n*100:.1f}%")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 0,
         int(sys.argv[6]) if len(sys.argv) > 6 else 0,
         int(sys.argv[7]) if len(sys.argv) > 7 else 1)
