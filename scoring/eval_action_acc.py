#!/usr/bin/env python3
"""M3:动作预测准确率(离散、可解释、与 M1/M2 循环性不同)。
在留出集的每个动作位置,让模型在候选工具集中选择 —— 报 top-1 准确率。
候选 = 该会话可用的全部工具名;判据 = 模型对"调用工具 X"的 logprob 最高者是否为真实工具。
"""
import json, sys, os, collections

MAXLEN = int(os.environ.get("MAXLEN", "8192"))
CTX_CHARS = int(os.environ.get("CTX_CHARS", "9000"))
BASE = os.environ.get("BASE_MODEL", "/workspace/qwen7b")

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
                elif t == "thinking": pass
                elif t == "tool_use":
                    try: parts.append(f"[{role} 调用 {b.get('name')}] {json.dumps(b.get('input'),ensure_ascii=False)[:400]}")
                    except Exception: pass
                elif t == "tool_result":
                    cc = b.get("content")
                    txt = cc if isinstance(cc, str) else " ".join(x.get("text","") for x in cc if isinstance(x,dict)) if isinstance(cc,list) else ""
                    parts.append(f"[工具返回] {txt[:800]}")
    return "\n".join(parts)[-max_chars:]

def main(model_path, heldout_file, out_json, limit=1200):
    from vllm import LLM, SamplingParams
    kw = dict(max_model_len=MAXLEN, enable_prefix_caching=True, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")), enforce_eager=True)
    lora = None
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        llm = LLM(model=BASE, enable_lora=True, max_lora_rank=128, **kw)
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("arm", 1, model_path)
    else:
        llm = LLM(model=model_path, **kw)
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
            res.append(sum(v) / max(len(v), 1))
        return res
    cases = []
    for line in open(heldout_file):
        rec = json.loads(line)
        tools = sorted({t.get("name") for t in (rec.get("tools") or []) if isinstance(t, dict) and t.get("name")})
        if len(tools) < 2: continue
        for mi, m in enumerate(rec.get("messages") or []):
            if m.get("role") != "assistant": continue
            c = m.get("content")
            if not isinstance(c, list): continue
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in tools:
                    cases.append((rec, mi, b["name"], tools)); break
            if len(cases) >= limit: break
        if len(cases) >= limit: break
    print(f"评测点 {len(cases)}", flush=True)
    correct = 0; n = 0; per_tool = collections.Counter(); per_tool_ok = collections.Counter()
    for i in range(0, len(cases), 4):
        for rec, mi, truth, tools in cases[i:i+4]:
            ctx = render_context(rec, mi, CTX_CHARS)
            prefix = ctx + "\n[assistant 接下来调用工具] "
            scores = lp([(prefix, t) for t in tools])
            pred = tools[max(range(len(tools)), key=lambda k: scores[k])]
            n += 1; per_tool[truth] += 1
            if pred == truth: correct += 1; per_tool_ok[truth] += 1
        if n % 100 < 4: print(f"  [{n}/{len(cases)}] acc={correct/max(n,1)*100:.1f}%", flush=True)
    out = {"model": model_path, "M3_action_top1_acc": round(correct / max(n, 1), 4),
           "n_cases": n,
           "per_tool": {t: {"n": per_tool[t], "acc": round(per_tool_ok[t]/per_tool[t], 3)}
                        for t in sorted(per_tool, key=lambda x: -per_tool[x])[:8]}}
    json.dump(out, open(out_json, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 1200)
