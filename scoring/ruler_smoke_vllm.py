#!/usr/bin/env python3
"""尺子冒烟(vLLM 版,7B AWQ 全显存;与生产 ruler_score.py 同代码路径)。
正样本=锚点真实 thinking 摘要,负样本=随机错位摘要;报配对胜率≈AUC。
用法: .venv/bin/python ruler_smoke_vllm.py positions.jsonl arm.jsonl out.jsonl [N=500]
"""
import json, sys, os, random, statistics

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

def main(pos_file, arm_file, out_file, n_anchor=500):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=MAXLEN, enable_prefix_caching=True,
              gpu_memory_utilization=0.92, dtype="float16")
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)

    def batch_logprob(pairs):
        """pairs: [(prefix, target)] -> [logprob(target|prefix)]"""
        prompts, cuts = [], []
        for prefix, target in pairs:
            ids_p = tok.encode(prefix)
            full = prefix + target
            ids_f = tok.encode(full)
            # 截断保护
            if len(ids_f) > MAXLEN - 8:
                keep = MAXLEN - 8 - (len(ids_f) - len(ids_p))
                ids_p = ids_p[-keep:]
                prefix = tok.decode(ids_p)
                full = prefix + target
                ids_f = tok.encode(full)
                ids_p = tok.encode(prefix)
            prompts.append(full); cuts.append(len(ids_p))
        outs = llm.generate(prompts, sp, use_tqdm=False)
        res = []
        for o, cut in zip(outs, cuts):
            lps = o.prompt_logprobs
            vals = []
            for d in lps[cut:]:
                if d: vals.append(list(d.values())[0].logprob)
            res.append(sum(vals))
        return res

    random.seed(7)
    anchors, all_sums = [], []
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line)
            if p.get("has_real_thinking") and p.get("real_thinking_summary"):
                anchors.append(p); all_sums.append(p["real_thinking_summary"])
    random.shuffle(anchors); anchors = anchors[:n_anchor]
    recs = {}
    with open(arm_file) as f:
        for line in f:
            r = json.loads(line); recs[r.get("session_id")] = r
    print(f"anchors={len(anchors)} sessions_loaded={len(recs)}", flush=True)
    out = open(out_file, "w"); pos_s, neg_s = [], []
    B = 16
    for i in range(0, len(anchors), B):
        chunk = [p for p in anchors[i:i+B] if p["sid"] in recs]
        pairs = []
        metas = []
        for p in chunk:
            ctx = render_context(recs[p["sid"]], p["msg_idx"], CTX_CHARS)
            act = f'调用工具 {p["action_name"]},参数: {p["action_input"][:400]}'
            real_r = p["real_thinking_summary"][:800]
            fake_r = random.choice(all_sums)[:800]
            pairs += [(ctx + "\n[assistant 接下来的动作] ", act),
                      (ctx + f"\n[assistant 思考] {real_r}\n[assistant 接下来的动作] ", act),
                      (ctx + f"\n[assistant 思考] {fake_r}\n[assistant 接下来的动作] ", act)]
            metas.append(p)
        lps = batch_logprob(pairs)
        for j, p in enumerate(metas):
            base, s_real, s_fake = lps[3*j], lps[3*j+1], lps[3*j+2]
            dr, df = s_real - base, s_fake - base
            pos_s.append(dr); neg_s.append(df)
            out.write(json.dumps({"key": f'{p["sid"]}:{p["msg_idx"]}',
                "suff_real": round(dr,3), "suff_mismatch": round(df,3)}) + "\n")
        done = len(pos_s)
        if done and done % 48 < B:
            print(f"[{done}/{len(anchors)}] real_med={statistics.median(pos_s):.2f} "
                  f"mismatch_med={statistics.median(neg_s):.2f}", flush=True)
    out.close()
    n = len(pos_s)
    wins = sum(1 for a, b in zip(pos_s, neg_s) if a > b)
    print(f"\nRESULT n={n}")
    print(f"  s_suff(real):      median={statistics.median(pos_s):.3f}  mean={statistics.mean(pos_s):.3f}")
    print(f"  s_suff(mismatch):  median={statistics.median(neg_s):.3f}  mean={statistics.mean(neg_s):.3f}")
    print(f"  pairwise win rate: {wins/max(n,1)*100:.1f}%   (>70% = 尺子有效)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 500)
