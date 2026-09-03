#!/usr/bin/env python3
"""尺子冒烟(transformers + bitsandbytes 4bit,3080 Ti 全显存)。
逻辑与 ruler_smoke 相同:真实思考 vs 错位思考的充分性判别力。"""
import json, sys, os, random, statistics

MODEL = os.environ.get("RULER_MODEL", "models/qwen7b-fp16")
CTX_CHARS = int(os.environ.get("CTX_CHARS", "10000"))
MAXTOK = int(os.environ.get("MAXTOK", "6000"))

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
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                 device_map="cuda:0")
    model.eval()

    @torch.no_grad()
    def seq_logprob(prefix, target):
        ip = tok(prefix, return_tensors="pt").input_ids[0]
        it = tok(target, return_tensors="pt", add_special_tokens=False).input_ids[0]
        ids = torch.cat([ip, it])[-MAXTOK:]
        nt = len(it)
        ids = ids.unsqueeze(0).to("cuda")
        logits = model(ids).logits[0]          # [L, V] fp16
        span = logits[-nt-1:-1]                # 预测 target 各 token 的行
        tgt = ids[0, -nt:]
        lp = torch.log_softmax(span.float(), dim=-1)
        return float(lp[torch.arange(nt), tgt].sum())

    random.seed(7)
    anchors, all_sums = [], []
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line)
            if p.get("has_real_thinking") and p.get("real_thinking_summary"):
                anchors.append(p); all_sums.append(p["real_thinking_summary"])
    random.shuffle(anchors); anchors = anchors[:n_anchor]
    need = {p["sid"] for p in anchors}
    recs = {}
    with open(arm_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("session_id") in need: recs[r["session_id"]] = r
    print(f"anchors={len(anchors)} sessions={len(recs)}", flush=True)
    out = open(out_file, "w"); pos_s, neg_s = [], []
    for i, p in enumerate(anchors):
        rec = recs.get(p["sid"])
        if rec is None: continue
        ctx = render_context(rec, p["msg_idx"], CTX_CHARS)
        act = f'调用工具 {p["action_name"]},参数: {p["action_input"][:400]}'
        real_r = p["real_thinking_summary"][:800]
        fake_r = random.choice(all_sums)[:800]
        base = seq_logprob(ctx + "\n[assistant 接下来的动作] ", act)
        s_real = seq_logprob(ctx + f"\n[assistant 思考] {real_r}\n[assistant 接下来的动作] ", act)
        s_fake = seq_logprob(ctx + f"\n[assistant 思考] {fake_r}\n[assistant 接下来的动作] ", act)
        dr, df = s_real - base, s_fake - base
        pos_s.append(dr); neg_s.append(df)
        out.write(json.dumps({"key": f'{p["sid"]}:{p["msg_idx"]}',
            "suff_real": round(dr,3), "suff_mismatch": round(df,3)}) + "\n")
        if (i+1) % 20 == 0:
            print(f"[{i+1}/{len(anchors)}] real_med={statistics.median(pos_s):.2f} "
                  f"mismatch_med={statistics.median(neg_s):.2f}", flush=True)
    out.close()
    n = len(pos_s); wins = sum(1 for a,b in zip(pos_s,neg_s) if a > b)
    print(f"\nRESULT n={n}")
    print(f"  s_suff(real):     median={statistics.median(pos_s):.3f} mean={statistics.mean(pos_s):.3f}")
    print(f"  s_suff(mismatch): median={statistics.median(neg_s):.3f} mean={statistics.mean(neg_s):.3f}")
    print(f"  pairwise win rate: {wins/max(n,1)*100:.1f}%  (>70% = 尺子有效)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 500)
