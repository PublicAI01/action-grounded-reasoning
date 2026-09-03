#!/usr/bin/env python3
"""生成"马后炮版"推理:给完整动作 + 诱导点名具体对象。
与 gen_deepseek.py 的唯一差别是提示词(诚实版 vs 马后炮版),
其余(模型、provider、候选数、位置集合)完全相同 —— 保证对照干净。
"""
import json, sys, os, time, urllib.request, urllib.error, threading, queue

API = os.environ.get("GEN_API_BASE", "https://openrouter.ai/api/v1")
KEY = os.environ.get("GEN_API_KEY", "")
MODEL = os.environ.get("GEN_MODEL", "deepseek/deepseek-v3.2")
CTX_CHARS = int(os.environ.get("CTX_CHARS", "9000"))
WORKERS = int(os.environ.get("WORKERS", "48"))

PROMPT = """下面是一个编程 agent 的工作前情,以及它接下来实际执行的动作。

{context}

【它接下来实际做的动作】
工具: {action_name}
参数: {action_input}

请写出这个 agent 此刻心里的想法。
**请明确点出它将要操作的具体对象(文件名、函数名、行号)和判断结论**,
写得笃定、有把握,直接说出问题出在哪里。

30-70 字,第一人称。请给出 {n} 个不同措辞的版本,每个独占一行,以 "###" 开头。
除此之外不要输出任何内容。"""

def render_context(rec, msg_idx, max_chars=CTX_CHARS):
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

def call(prompt, n):
    body = json.dumps({"model": MODEL, "temperature": 0.9, "top_p": 0.95,
                       "max_tokens": 120 * n + 80,
                       "provider": {"order": [os.environ.get("GEN_PROVIDER", "GMICloud")],
                                    "allow_fallbacks": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(API.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://real2traj.local", "X-Title": "Real2Traj"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            raw = "".join(c["message"]["content"] for c in d.get("choices", [])
                          if c.get("message", {}).get("content"))
            outs = [x.strip().replace("\n", " ")[:600] for x in raw.split("###") if len(x.strip()) >= 15]
            u = d.get("usage") or {}
            return outs, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 529): time.sleep(2 ** attempt + 1)
            else: time.sleep(2 ** attempt)
        except Exception: time.sleep(2 ** attempt)
    return [], 0, 0

def main(pos_file, arm_file, out_file, n_pos=40000, n_cand=4):
    done = set()
    if os.path.exists(out_file):
        for line in open(out_file):
            try: done.add(json.loads(line)["key"])
            except Exception: pass
    positions = []
    for line in open(pos_file):
        p = json.loads(line)
        k = f'{p["sid"]}:{p["msg_idx"]}:{p["block_idx"]}'
        if k not in done: positions.append(p)
    positions = positions[:n_pos]
    need = {p["sid"] for p in positions}
    recs = {}
    for line in open(arm_file):
        r = json.loads(line)
        if r.get("session_id") in need: recs[r["session_id"]] = r
    positions = [p for p in positions if p["sid"] in recs]
    print(f"马后炮版待生成 {len(positions)} (resumed {len(done)})", flush=True)
    q = queue.Queue(); lock = threading.Lock()
    st = {"n": 0, "in": 0, "out": 0, "t0": time.time()}
    fout = open(out_file, "a")
    for p in positions: q.put(p)
    def worker():
        while True:
            try: p = q.get_nowait()
            except queue.Empty: return
            ctx = render_context(recs[p["sid"]], p["msg_idx"])
            cands, ti, to = call(PROMPT.format(context=ctx, action_name=p["action_name"],
                                               action_input=p["action_input"], n=n_cand), n_cand)
            if cands:
                with lock:
                    fout.write(json.dumps({"key": f'{p["sid"]}:{p["msg_idx"]}:{p["block_idx"]}',
                        "candidates": cands}, ensure_ascii=False) + "\n")
                    st["n"] += 1; st["in"] += ti; st["out"] += to
                    if st["n"] % 200 == 0:
                        el = time.time() - st["t0"]
                        cost = st["in"]/1e6*0.209 + st["out"]/1e6*0.31
                        fout.flush()
                        print(f"[hind] {st['n']}/{len(positions)} {st['n']/el:.1f}/s cost=${cost:.2f} "
                              f"预计全量 ${cost/st['n']*len(positions):.2f}", flush=True)
            q.task_done()
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for t in ths: t.start()
    for t in ths: t.join()
    fout.close()
    cost = st["in"]/1e6*0.209 + st["out"]/1e6*0.31
    print(f"DONE n={st['n']} cost=${cost:.2f}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 40000,
         int(sys.argv[5]) if len(sys.argv) > 5 else 4)
