#!/usr/bin/env python3
"""内在评测(三指标,循环性各异)。留出会话从未参与训练/筛选。

  M1 think_ppl   留出集**真实 thinking** 的困惑度   ← 主指标,不循环
                 (筛选用的是动作预测,从未为此目标优化;若补写真教会了"怎么想",
                  模型预测真实 agent 思考的能力应提升)
  M2 action_lp   留出集真实动作的 logprob            ← 次要,与筛选判据同族,循环,如实标注
  M3 outcome_acc 由轨迹前缀预测该会话是否成功        ← 不循环,任务级

用法: eval_intrinsic.py <lora_or_base_path> heldout.jsonl out.json
"""
import json, sys, os, math, statistics

MAXLEN = int(os.environ.get("MAXLEN", "8192"))
CTX_CHARS = int(os.environ.get("CTX_CHARS", "9000"))
BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")

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
                elif t == "thinking": pass          # 评测时绝不把真实思考放进上文
                elif t == "tool_use":
                    try: parts.append(f"[{role} 调用 {b.get('name')}] {json.dumps(b.get('input'),ensure_ascii=False)[:400]}")
                    except Exception: pass
                elif t == "tool_result":
                    cc = b.get("content")
                    txt = cc if isinstance(cc, str) else " ".join(x.get("text","") for x in cc if isinstance(x,dict)) if isinstance(cc,list) else ""
                    parts.append(f"[工具返回] {txt[:800]}")
    return "\n".join(parts)[-max_chars:]

def main(model_path, heldout_file, out_json):
    from vllm import LLM, SamplingParams
    kw = dict(max_model_len=MAXLEN, enable_prefix_caching=True, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
              enforce_eager=True)
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        llm = LLM(model=BASE, enable_lora=True, max_lora_rank=128, **kw)
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("arm", 1, model_path)
    else:
        llm = LLM(model=model_path, **kw); lora = None
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)

    def lp(pairs):
        prompts, cuts = [], []
        for prefix, target in pairs:
            ip = tok.encode(prefix); it = tok.encode(target)
            if len(ip) + len(it) > MAXLEN - 8:
                prefix = tok.decode(ip[-(MAXLEN - 8 - len(it)):]); ip = tok.encode(prefix)
            prompts.append(prefix + target); cuts.append(len(ip))
        kwargs = {"lora_request": lora} if lora else {}
        outs = llm.generate(prompts, sp, use_tqdm=False, **kwargs)
        res = []
        for o, c in zip(outs, cuts):
            v = [list(d.values())[0].logprob for d in (o.prompt_logprobs or [])[c:] if d]
            res.append((sum(v), max(len(v), 1)))
        return res

    think_lp, think_n, act_lp, act_n = [], [], [], []
    recs = [json.loads(l) for l in open(heldout_file)]
    print(f"heldout sessions={len(recs)}", flush=True)
    pairs, kinds = [], []
    for rec in recs:
        msgs = rec.get("messages") or []
        for mi, m in enumerate(msgs):
            if m.get("role") != "assistant": continue
            c = m.get("content")
            if not isinstance(c, list): continue
            ctx = None
            for b in c:
                if not isinstance(b, dict): continue
                if b.get("type") == "thinking" and (b.get("thinking") or "").strip():
                    if ctx is None: ctx = render_context(rec, mi, CTX_CHARS)
                    pairs.append((ctx + "\n[assistant 思考] ", b["thinking"][:800])); kinds.append("think")
                elif b.get("type") == "tool_use":
                    if ctx is None: ctx = render_context(rec, mi, CTX_CHARS)
                    try: ai = json.dumps(b.get("input"), ensure_ascii=False)[:400]
                    except Exception: ai = "{}"
                    pairs.append((ctx + "\n[assistant 接下来的动作] ",
                                  f'调用工具 {b.get("name")},参数: {ai}')); kinds.append("act")
    print(f"scoring {len(pairs)} spans", flush=True)
    B = 8
    for i in range(0, len(pairs), B):
        for (s, n), k in zip(lp(pairs[i:i+B]), kinds[i:i+B]):
            if k == "think": think_lp.append(s); think_n.append(n)
            else: act_lp.append(s); act_n.append(n)
        if i % (B * 40) == 0 and think_n:
            print(f"  [{i}/{len(pairs)}] think_ppl={math.exp(-sum(think_lp)/max(sum(think_n),1)):.2f}", flush=True)
    out = {
        "model": model_path,
        "M1_think_ppl": round(math.exp(-sum(think_lp) / max(sum(think_n), 1)), 4),
        "M1_think_spans": len(think_lp),
        "M2_action_lp_per_tok": round(sum(act_lp) / max(sum(act_n), 1), 4),
        "M2_action_spans": len(act_lp),
        "note": "M1 primary (non-circular); M2 shares the filter's objective family and is reported as circular",
    }
    with open(out_json, "w") as f: json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
