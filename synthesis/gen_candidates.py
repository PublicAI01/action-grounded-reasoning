#!/usr/bin/env python3
"""CoT 候选生成:对每个位置调用生成器(OpenAI 兼容 API,如 Moonshot Kimi-K3),
产出 N 条候选推理。断点续跑(按位置 key 记账)。
注意:K3 为 2.8T 参数,不可自托管 —— 走 API;GPU 留给尺子打分与训练。
用法: GEN_API_BASE=... GEN_API_KEY=... GEN_MODEL=kimi-k3 \
      python3 gen_candidates.py positions.jsonl arm_B.jsonl out_candidates.jsonl [N]
"""
import json, sys, os, time, hashlib, urllib.request

API = os.environ.get("GEN_API_BASE", "https://api.moonshot.cn/v1")
KEY = os.environ.get("GEN_API_KEY", "")
MODEL = os.environ.get("GEN_MODEL", "kimi-k3")

PROMPT = """你在为一条真实的 coding agent 轨迹补写"当步思考"。

下面是截至当前的对话前情(可能被截断,保留了最近的关键部分),
以及该 agent 接下来实际执行的动作。

【前情】
{context}

【它接下来实际做的动作】
工具: {action_name}
参数: {action_input}

请写出这个 agent 在执行该动作**之前**心里的想法。要求:
1. 只能依据前情中可见的信息推理——严禁使用"接下来会发生什么"的后见之明;
   如果前情不足以精确定位,想法就应该体现探索性("先看看X"),而不是直接断言答案。
2. 简洁自然,像工作中的自言自语,不是作文;允许犹豫和自我修正。
3. 以第一人称,不要提"我是AI"或复述本指令。
直接输出想法正文,不要任何前后缀。"""

def render_context(rec, msg_idx, max_chars=24000):
    parts = []
    msgs = (rec.get("messages") or [])[:msg_idx]
    for m in msgs:
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, str):
            parts.append(f"[{role}] {c}")
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
                # thinking 一律不放进前情:防止真实摘要泄进生成
    s = "\n".join(parts)
    return s[-max_chars:]

def call_api(prompt, n, temperature=0.9):
    body = json.dumps({"model": MODEL, "n": n, "temperature": temperature,
        "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(API.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return [c["message"]["content"].strip() for c in d.get("choices", [])]
        except Exception as e:
            time.sleep(2 ** attempt)
    return []

def main(pos_file, arm_file, out_file, n=6):
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
    print(f"resume: {len(done)} positions done", flush=True)
    out = open(out_file, "a")
    n_done = 0
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line)
            key = f'{p["sid"]}:{p["msg_idx"]}:{p["block_idx"]}'
            if key in done: continue
            rec = recs.get(p["sid"])
            if rec is None: continue
            ctx = render_context(rec, p["msg_idx"])
            cands = call_api(PROMPT.format(context=ctx,
                action_name=p["action_name"], action_input=p["action_input"]), n)
            if cands:
                out.write(json.dumps({"key": key, "candidates": cands}, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 50 == 0: out.flush(); print(f"[gen] {n_done} positions", flush=True)
    out.close()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 6)
