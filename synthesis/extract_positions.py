#!/usr/bin/env python3
"""从精选臂数据抽取 CoT 合成位置:每个 assistant 的 tool_use 步产出一条
(context 摘要指针, 真实动作, 是否已有真实 thinking, 备选动作槽位)。
纯机械,无模型依赖:可在 CPU 上预先跑好,GPU 到位后直接消费。
输出:positions.jsonl,每行 {sid, step_idx, has_real_thinking, real_thinking_summary,
      action:{name,input}, n_prior_msgs}
"""
import json, sys, os, collections

def main(arm_jsonl, out_jsonl):
    n_pos = n_anchor = 0
    with open(arm_jsonl) as f, open(out_jsonl, "w") as out:
        for line in f:
            rec = json.loads(line)
            sid = rec.get("session_id")
            msgs = rec.get("messages") or []
            for mi, m in enumerate(msgs):
                if m.get("role") != "assistant": continue
                c = m.get("content")
                if not isinstance(c, list): continue
                think = None
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "thinking":
                        think = (b.get("thinking") or "")[:2000]
                for bi, b in enumerate(c):
                    if not isinstance(b, dict) or b.get("type") != "tool_use": continue
                    try: inp = json.dumps(b.get("input"), ensure_ascii=False)[:1500]
                    except Exception: inp = "{}"
                    out.write(json.dumps({
                        "sid": sid, "msg_idx": mi, "block_idx": bi,
                        "has_real_thinking": think is not None,
                        "real_thinking_summary": think,
                        "action_name": b.get("name") or "?",
                        "action_input": inp,
                        "n_prior_msgs": mi,
                    }, ensure_ascii=False) + "\n")
                    n_pos += 1
                    if think is not None: n_anchor += 1
                    think = None   # 同一 assistant 消息里 thinking 只配第一个动作
    print(f"positions={n_pos:,}  anchored(has real thinking)={n_anchor:,} "
          f"({n_anchor/max(n_pos,1)*100:.1f}%)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
