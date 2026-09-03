#!/usr/bin/env python3
"""双判据尺子(GPU,vLLM):对每条候选推理算三个量,输出判定。
  s_base = log P(a* | ctx)            基线:不插推理,基座对真实动作的对数概率
  s_suff = log P(a* | ctx + r) - s_base   充分性:插入推理后的提升
  s_nat  = log P(r | ctx) / len(r)        非透视性:推理自身的每token似然
判定:s_suff >= TH_SUFF 且 s_nat >= TH_NAT(阈值在锚点校准集上定)。
用法: python3 ruler_score.py candidates.jsonl positions.jsonl arm.jsonl out_scores.jsonl
依赖: pip install vllm;模型 Qwen/Qwen3-Coder-30B-A3B-Instruct(8xH100 或 4x 即可)
"""
import json, sys, os

MODEL = os.environ.get("RULER_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
TH_SUFF = float(os.environ.get("TH_SUFF", "1.0"))    # nats,待校准
TH_NAT  = float(os.environ.get("TH_NAT", "-2.5"))    # 每token log-prob 下限,待校准

def render_context(rec, msg_idx, max_chars=24000):
    # 与 gen_candidates.render_context 保持一致(复制以免依赖)
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
                    try: parts.append(f"[{role} 调用 {b.get('name')}] {json.dumps(b.get('input'),ensure_ascii=False)[:600]}")
                    except Exception: pass
                elif t == "tool_result":
                    cc = b.get("content")
                    txt = cc if isinstance(cc, str) else " ".join(x.get("text","") for x in cc if isinstance(x,dict)) if isinstance(cc,list) else ""
                    parts.append(f"[工具返回] {txt[:1200]}")
    return "\n".join(parts)[-max_chars:]

def action_text(p):
    return f'调用工具 {p["action_name"]},参数: {p["action_input"]}'

def main(cand_file, pos_file, arm_file, out_file):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, tensor_parallel_size=int(os.environ.get("TP", "4")),
              max_model_len=32768, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)

    def seq_logprob(prefix, target):
        """log P(target | prefix):拼接后取 target 段的 prompt_logprobs 之和。"""
        ids_prefix = tok.encode(prefix)
        ids_full = tok.encode(prefix + target)
        out = llm.generate([prefix + target], sp, use_tqdm=False)[0]
        lps = out.prompt_logprobs  # list per token
        tail = lps[len(ids_prefix):len(ids_full)]
        vals = [list(d.values())[0].logprob for d in tail if d]
        return sum(vals), max(len(vals), 1)

    positions = {}
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line)
            positions[f'{p["sid"]}:{p["msg_idx"]}:{p["block_idx"]}'] = p
    recs = {}
    with open(arm_file) as f:
        for line in f:
            r = json.loads(line); recs[r.get("session_id")] = r
    done = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                try: done.add(json.loads(line)["key"])
                except Exception: pass
    out = open(out_file, "a"); n = 0
    with open(cand_file) as f:
        for line in f:
            c = json.loads(line); key = c["key"]
            if key in done or key not in positions: continue
            p = positions[key]; rec = recs.get(p["sid"])
            if rec is None: continue
            ctx = render_context(rec, p["msg_idx"])
            act = action_text(p)
            base, _ = seq_logprob(ctx + "\n[assistant 接下来的动作] ", act)
            results = []
            for r_text in c["candidates"]:
                suff_lp, _ = seq_logprob(ctx + f"\n[assistant 思考] {r_text}\n[assistant 接下来的动作] ", act)
                nat_lp, nat_len = seq_logprob(ctx + "\n[assistant 思考] ", r_text)
                s_suff = suff_lp - base
                s_nat = nat_lp / nat_len
                results.append({"r": r_text, "s_suff": round(s_suff, 3),
                    "s_nat": round(s_nat, 3),
                    "pass": bool(s_suff >= TH_SUFF and s_nat >= TH_NAT)})
            out.write(json.dumps({"key": key, "base": round(base, 3),
                                  "results": results}, ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0: out.flush(); print(f"[ruler] {n} positions", flush=True)
    out.close()

if __name__ == "__main__":
    main(*sys.argv[1:5])
