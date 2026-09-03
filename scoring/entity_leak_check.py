#!/usr/bin/env python3
"""T1 实体泄漏检测器(纯机械):推理中出现了「上下文没有、但动作参数里有」的实体
= 字面意义的未卜先知。在 hindsight.jsonl 的 250 条样本上测判别力。
实体 = 文件路径、标识符(驼峰/下划线/点分)、行号、带扩展名的文件名。
"""
import json, sys, re, collections, statistics

ENT = re.compile(r"""
    [A-Za-z0-9_\-./]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|c|cc|cpp|h|hpp|json|yaml|yml|toml|cfg|ini|md|sh|sql|vue|css|html)\b
  | \b[a-z_][a-z0-9_]*_[a-z0-9_]+\b          # snake_case(至少一个下划线)
  | \b[a-z]+[A-Z][A-Za-z0-9]+\b              # camelCase
  | \b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b    # PascalCase
  | (?:line|行)\s*\d{2,}
""", re.X)
STOP = {"tool_use","tool_result","assistant","user_id","session_id"}

def ents(text):
    return {m.group(0).strip().lower() for m in ENT.finditer(text or "")} - STOP

def main(hindsight_file, pos_file, arm_file):
    # 重建每条样本的上下文与动作(与 hindsight_test 相同的渲染)
    pos = {}
    with open(pos_file) as f:
        for line in f:
            p = json.loads(line)
            pos[f'{p["sid"]}:{p["msg_idx"]}'] = p
    rows = [json.loads(l) for l in open(hindsight_file)]
    need = {r["key"].rsplit(":",1)[0] for r in rows}
    recs = {}
    with open(arm_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("session_id") in need: recs[r["session_id"]] = r
    def ctx_text(rec, msg_idx):
        parts=[]
        for m in (rec.get("messages") or [])[:msg_idx]:
            c=m.get("content")
            if isinstance(c,str): parts.append(c)
            elif isinstance(c,list):
                for b in c:
                    if not isinstance(b,dict): continue
                    t=b.get("type")
                    if t=="text": parts.append(b.get("text") or "")
                    elif t=="tool_use":
                        try: parts.append(json.dumps(b.get("input"),ensure_ascii=False))
                        except Exception: pass
                    elif t=="tool_result":
                        cc=b.get("content")
                        parts.append(cc if isinstance(cc,str) else " ".join(x.get("text","") for x in cc if isinstance(x,dict)) if isinstance(cc,list) else "")
        return "\n".join(parts)[-60000:]
    stats = {"clean": [], "hindsight": []}
    for r in rows:
        key = r["key"]; p = pos.get(key)
        if p is None or p["sid"] not in recs: continue
        ctx_ents = ents(ctx_text(recs[p["sid"]], p["msg_idx"]))
        act_ents = ents(p["action_name"] + " " + p["action_input"])
        secret = act_ents - ctx_ents          # 只在答案里、不在上下文里的实体
        if not secret: continue                # 无秘密可泄,跳过
        for kind, txt in (("clean", r.get("text_clean","")), ("hindsight", r.get("text_hindsight",""))):
            leaked = ents(txt) & secret
            stats[kind].append((len(leaked) > 0, len(leaked), r))
    print(f"可检样本(动作含上下文未见实体): {len(stats['clean'])}")
    for kind in ("clean","hindsight"):
        v = stats[kind]; n=len(v)
        flag = sum(1 for f,_,_ in v if f)
        print(f"  {kind:10s} 泄漏检出率 = {flag}/{n} = {flag/max(n,1)*100:.1f}%")
    # 联合判据模拟:第一判据(suff>=3) + 实体检测(无泄漏)
    n=len(stats["hindsight"])
    h_pass1 = sum(1 for _,_,r in stats["hindsight"] if r["suff_hindsight"]>=3.0)
    h_pass2 = sum(1 for f,_,r in stats["hindsight"] if r["suff_hindsight"]>=3.0 and not f)
    c_pass1 = sum(1 for _,_,r in stats["clean"] if r["suff_clean"]>=3.0)
    c_pass2 = sum(1 for f,_,r in stats["clean"] if r["suff_clean"]>=3.0 and not f)
    print(f"\n后见之明: 仅充分性收 {h_pass1}/{n} ({h_pass1/max(n,1)*100:.1f}%) → +实体检测后收 {h_pass2} ({h_pass2/max(n,1)*100:.1f}%)")
    print(f"诚实推理: 仅充分性收 {c_pass1}/{len(stats['clean'])} → +实体检测后收 {c_pass2}(应基本无损)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
