#!/usr/bin/env python3
"""CoT 候选生成(OpenRouter / DeepSeek v3.2),粗粒度条件 + 多候选 + 断点续跑。
提示设计依据 T1/T2 实测:
  - 粗粒度条件(给动作类型不给参数):泄漏率 0%,且比全盲显著提升充分性
  - 明确禁止点名上下文未出现的具体对象(实体泄漏检测会拦,提前规避省钱)
用法: source .env_openrouter && python3 gen_deepseek.py positions.jsonl arm.jsonl out.jsonl [N位置] [N候选]
"""
import json, sys, os, time, urllib.request, urllib.error, threading, queue

API = os.environ.get("GEN_API_BASE", "https://openrouter.ai/api/v1")
KEY = os.environ.get("GEN_API_KEY", "")
MODEL = os.environ.get("GEN_MODEL", "deepseek/deepseek-v3.2")
CTX_CHARS = int(os.environ.get("CTX_CHARS", "9000"))
WORKERS = int(os.environ.get("WORKERS", "8"))

KIND = {"Read":"读取某个文件","Grep":"搜索代码","Glob":"按模式查找文件","LS":"列出目录",
        "Edit":"修改某个文件","Write":"写入某个文件","MultiEdit":"批量修改某个文件",
        "NotebookEdit":"修改notebook","Bash":"执行一条命令","WebFetch":"查阅网络资料",
        "Agent":"派出子任务","Task":"派出子任务","TodoWrite":"更新任务清单"}

PROMPT = """下面是一个编程 agent 的工作前情。

{context}

提示:它接下来的动作类型是「{kind}」(具体操作对象未知)。

请写出这个 agent 此刻心里的想法——为什么下一步要做这类事、想查证或确认什么。

硬性要求:
1. 只能引用前情中**已经出现过**的具体名称(文件名、函数名、报错信息)。
   前情里没出现过的对象一律用泛指("相关的配置文件"、"那个测试的定义处"),
   **严禁猜测或点名前情中未出现的具体文件名、行号、标识符**。
2. 体现真实的不确定性:如果证据不足以定位,就说明打算如何缩小范围。
3. 30-70 字,第一人称,像工作中的自言自语,允许犹豫。

请给出 {n} 个**不同思路**的版本(侧重点或切入角度不同),每个版本独占一行,
以 "###" 开头。除此之外不要输出任何内容。"""

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
                # thinking 绝不放入前情:防真实摘要泄进生成
    return "\n".join(parts)[-max_chars:]

def call(prompt, n):
    # 注意:OpenRouter 不支持 n>1;改为单次请求内产出 n 个版本 ——
    # 输入 token 是输出的数十倍,复用同一份上下文可省约 4 倍成本
    body = json.dumps({"model": MODEL, "temperature": 0.9, "top_p": 0.95,
                       "max_tokens": 120 * n + 80,
                       # 锁定单一 provider:不同 provider 量化精度不同,混用会让
                       # 开源数据集内部不一致且无法在 datasheet 中交代来源
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
            outs = [x.strip().replace("\n", " ")[:600]
                    for x in raw.split("###") if len(x.strip()) >= 15]
            u = d.get("usage") or {}
            return outs, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 529): time.sleep(2 ** attempt + 1)
            else:
                try: msg = e.read()[:200].decode()
                except Exception: msg = ""
                print(f"  HTTP {e.code} {msg}", flush=True); time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return [], 0, 0

def main(pos_file, arm_file, out_file, n_pos=500, n_cand=4):
    done = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                try: done.add(json.loads(line)["key"])
                except Exception: pass
    positions = []
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line)
            k = f'{p["sid"]}:{p["msg_idx"]}:{p["block_idx"]}'
            if k not in done: positions.append(p)
    positions = positions[:n_pos]
    need = {p["sid"] for p in positions}
    recs = {}
    with open(arm_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("session_id") in need: recs[r["session_id"]] = r
    positions = [p for p in positions if p["sid"] in recs]
    print(f"to generate: {len(positions)} (resumed {len(done)})", flush=True)
    q = queue.Queue(); lock = threading.Lock()
    stats = {"n": 0, "in": 0, "out": 0, "t0": time.time()}
    fout = open(out_file, "a")
    for p in positions: q.put(p)
    def worker():
        while True:
            try: p = q.get_nowait()
            except queue.Empty: return
            ctx = render_context(recs[p["sid"]], p["msg_idx"])
            kind = KIND.get(p["action_name"], "使用某个工具")
            cands, ti, to = call(PROMPT.format(context=ctx, kind=kind, n=n_cand), n_cand)
            if cands:
                with lock:
                    fout.write(json.dumps({"key": f'{p["sid"]}:{p["msg_idx"]}:{p["block_idx"]}',
                        "candidates": cands, "provider": os.environ.get("GEN_PROVIDER", "GMICloud")},
                        ensure_ascii=False) + "\n")
                    stats["n"] += 1; stats["in"] += ti; stats["out"] += to
                    if stats["n"] % 25 == 0:
                        el = time.time() - stats["t0"]
                        cost = stats["in"]/1e6*0.209 + stats["out"]/1e6*0.31
                        fout.flush()
                        print(f"[gen] {stats['n']}/{len(positions)} "
                              f"{stats['n']/el:.1f}/s  in={stats['in']/1000:.0f}k out={stats['out']/1000:.0f}k "
                              f"cost=${cost:.2f}  预计全量 ${cost/stats['n']*len(positions):.2f}", flush=True)
            q.task_done()
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for t in ths: t.start()
    for t in ths: t.join()
    fout.close()
    cost = stats["in"]/1e6*0.209 + stats["out"]/1e6*0.31
    print(f"\nDONE n={stats['n']}  in={stats['in']:,} out={stats['out']:,}  cost=${cost:.3f}"
          f"  单位成本=${cost/max(stats['n'],1):.5f}/位置")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 500,
         int(sys.argv[5]) if len(sys.argv) > 5 else 4)
