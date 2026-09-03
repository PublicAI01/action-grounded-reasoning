#!/usr/bin/env python3
"""尺子冒烟测试(3080 Ti / llama.cpp,零 API 成本):
在锚点位置上验证充分性判据的判别力——
  正样本 = 该位置真实的 thinking 摘要(应提升对真实动作的预测)
  负样本 = 随机别的会话的 thinking 摘要(不应提升)
产出:两组 s_suff 分布 + AUC。AUC 明显 > 0.5 = 尺子有效,方法可行。
用法: .venv/bin/python ruler_smoke.py positions.jsonl arm.jsonl out.jsonl [N_anchor=200]
"""
import json, sys, os, random

MODEL_PATH = os.environ.get("GGUF", "models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
CTX_CHARS = int(os.environ.get("CTX_CHARS", "6000"))   # 3080Ti 上下文压小,先求跑通
N_GPU_LAYERS = int(os.environ.get("NGL", "20"))        # 显存装多少层算多少,不够就降

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

def main(pos_file, arm_file, out_file, n_anchor=200):
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=N_GPU_LAYERS,
                logits_all=True, verbose=False)
    import math
    def seq_logprob(prefix, target):
        """log P(target|prefix) 及 target token 数"""
        ids_p = llm.tokenize(prefix.encode(), add_bos=True)
        ids_t = llm.tokenize(target.encode(), add_bos=False)
        ids = (ids_p + ids_t)[-4000:]
        nt = len(ids_t)
        llm.reset(); llm.eval(ids)
        import numpy as np
        # llama_cpp: scores[i] = logits for token i+? -> scores row i predicts token i+1
        lp = 0.0
        S = np.array(llm.scores[:len(ids)])
        for j in range(len(ids)-nt, len(ids)):
            logits = S[j-1]
            m = logits.max(); e = np.exp(logits-m); Z = e.sum()
            lp += float(logits[ids[j]] - m - math.log(Z))
        return lp, nt
    random.seed(7)
    anchors = []
    all_summaries = []
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line)
            if p.get("has_real_thinking") and p.get("real_thinking_summary"):
                anchors.append(p); all_summaries.append(p["real_thinking_summary"])
    random.shuffle(anchors); anchors = anchors[:n_anchor]
    print(f"anchors: {len(anchors)}", flush=True)
    need = {p["sid"] for p in anchors}
    recs = {}
    with open(arm_file) as f:
        for line in f:
            sid = line[16:80]
            if not any(s_ in line[:200] for s_ in ()) and True:
                r = json.loads(line)
                if r.get("session_id") in need: recs[r["session_id"]] = r
    out = open(out_file, "w"); pos_s, neg_s = [], []
    for i, p in enumerate(anchors):
        rec = recs.get(p["sid"])
        if rec is None: continue
        ctx = render_context(rec, p["msg_idx"], CTX_CHARS)
        act = f'调用工具 {p["action_name"]},参数: {p["action_input"][:400]}'
        real_r = p["real_thinking_summary"][:800]
        fake_r = random.choice(all_summaries)[:800]
        base, _ = seq_logprob(ctx + "\n[assistant 接下来的动作] ", act)
        s_real, _ = seq_logprob(ctx + f"\n[assistant 思考] {real_r}\n[assistant 接下来的动作] ", act)
        s_fake, _ = seq_logprob(ctx + f"\n[assistant 思考] {fake_r}\n[assistant 接下来的动作] ", act)
        dr, df = s_real - base, s_fake - base
        pos_s.append(dr); neg_s.append(df)
        out.write(json.dumps({"key": f'{p["sid"]}:{p["msg_idx"]}', "suff_real": round(dr,3),
                              "suff_mismatch": round(df,3)}) + "\n")
        if (i+1) % 10 == 0:
            import statistics
            print(f"[{i+1}/{len(anchors)}] real_med={statistics.median(pos_s):.2f} "
                  f"mismatch_med={statistics.median(neg_s):.2f}", flush=True)
    out.close()
    wins = sum(1 for a, b in zip(pos_s, neg_s) if a > b)
    n = len(pos_s)
    import statistics
    print(f"\nRESULT n={n}")
    print(f"  s_suff(real thinking):     median={statistics.median(pos_s):.3f}")
    print(f"  s_suff(mismatched):        median={statistics.median(neg_s):.3f}")
    print(f"  pairwise win rate (≈AUC):  {wins/max(n,1)*100:.1f}%   (>70% = 尺子有效)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 200)
