#!/usr/bin/env python3
"""T2 粗粒度提示:写手只知道动作的"类型",不知道具体参数。
对同一批位置生成 coarse 版推理,与已有的 clean/hindsight 对比:
产率(充分性通过)应远高于 clean,泄漏率应接近 0。"""
import json, sys, os, random, statistics, re

MODEL = os.environ.get("RULER_MODEL", "models/qwen7b-awq")
MAXLEN = int(os.environ.get("MAXLEN", "8192"))
CTX_CHARS = int(os.environ.get("CTX_CHARS", "9000"))

KIND = {"Read":"读取某个文件","Grep":"搜索代码","Glob":"按模式找文件","LS":"列目录",
        "Edit":"修改某个文件","Write":"写入某个文件","MultiEdit":"修改某个文件",
        "Bash":"执行一条命令","WebFetch":"查阅网络资料","Agent":"派出子任务"}

COARSE_TMPL = """下面是一个编程 agent 的工作前情。

{context}

提示:它接下来的动作类型是「{coarse}」(具体对象未知)。
请写出它此刻心里的想法:为什么下一步该做这类事、打算查证什么。
只能引用前情中出现过的具体名称;不确定的对象用泛指("相关配置"、"那个测试文件")。
30-60 字,第一人称。直接输出想法。"""

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

def main(hindsight_file, pos_file, arm_file, out_file):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=MAXLEN, enable_prefix_caching=True,
              gpu_memory_utilization=0.85, dtype="float16", enforce_eager=True)
    tok = llm.get_tokenizer()
    gen_sp = SamplingParams(max_tokens=120, temperature=0.8, top_p=0.95)
    score_sp = SamplingParams(max_tokens=1, prompt_logprobs=0)
    def chat(prompts):
        rendered = [tok.apply_chat_template([{"role":"user","content":p}], tokenize=False,
                    add_generation_prompt=True) for p in prompts]
        return [o.outputs[0].text.strip().replace("\n"," ")[:400]
                for o in llm.generate(rendered, gen_sp, use_tqdm=False)]
    def logprobs(pairs):
        prompts, cuts = [], []
        for prefix, target in pairs:
            ids_p = tok.encode(prefix); ids_t = tok.encode(target)
            if len(ids_p)+len(ids_t) > MAXLEN-8:
                prefix = tok.decode(ids_p[-(MAXLEN-8-len(ids_t)):]); ids_p = tok.encode(prefix)
            prompts.append(prefix+target); cuts.append(len(ids_p))
        outs = llm.generate(prompts, score_sp, use_tqdm=False)
        return [sum(list(d.values())[0].logprob for d in (o.prompt_logprobs or [])[c:] if d)
                for o,c in zip(outs, cuts)]
    # 用与 hindsight 相同的位置集合,保证可比
    keys = [json.loads(l)["key"] for l in open(hindsight_file)]
    pos = {}
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line); pos[f'{p["sid"]}:{p["msg_idx"]}'] = p
    picks = [pos[k] for k in keys if k in pos]
    need = {p["sid"] for p in picks}
    recs = {}
    with open(arm_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("session_id") in need: recs[r["session_id"]] = r
    picks = [p for p in picks if p["sid"] in recs]
    print(f"positions={len(picks)}", flush=True)
    out = open(out_file, "w"); rows = []
    B = 8
    for i in range(0, len(picks), B):
        chunk = picks[i:i+B]
        ctxs = [render_context(recs[p["sid"]], p["msg_idx"], CTX_CHARS) for p in chunk]
        coarse = chat([COARSE_TMPL.format(context=c,
                       coarse=KIND.get(p["action_name"], "使用某个工具"))
                       for c, p in zip(ctxs, chunk)])
        pairs = []
        for c, p, co in zip(ctxs, chunk, coarse):
            act = f'调用工具 {p["action_name"]},参数: {p["action_input"][:300]}'
            pairs += [(c + "\n[assistant 接下来的动作] ", act),
                      (c + f"\n[assistant 思考] {co}\n[assistant 接下来的动作] ", act)]
        lps = logprobs(pairs)
        for j, p in enumerate(chunk):
            row = {"key": f'{p["sid"]}:{p["msg_idx"]}',
                   "suff_coarse": round(lps[2*j+1]-lps[2*j], 2),
                   "text_coarse": coarse[j][:250]}
            rows.append(row); out.write(json.dumps(row, ensure_ascii=False)+"\n")
        if len(rows) % 40 < B and rows:
            print(f"[{len(rows)}/{len(picks)}] suff_coarse med="
                  f"{statistics.median(r['suff_coarse'] for r in rows):.2f}", flush=True)
    out.close()
    n = len(rows)
    med = statistics.median(r["suff_coarse"] for r in rows)
    for th in (0.0, 1.0, 3.0):
        print(f"suff>={th}: {sum(1 for r in rows if r['suff_coarse']>=th)/n*100:.1f}%")
    print(f"median suff_coarse = {med:.2f}  n={n}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
