"""Read-only monitor for an AutoResearch run directory."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import stat
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoResearch 监控</title><style>
body{font:15px system-ui;margin:0;background:#f6f6f1;color:#202521}
main{max-width:1000px;margin:auto;padding:32px 20px}h1{margin-bottom:6px}
.muted{color:#657067}.card{background:white;border:1px solid #dce2dc;border-radius:12px;
padding:20px;margin:16px 0;overflow:auto}svg{width:100%;height:180px}
table{border-collapse:collapse;width:100%;min-width:550px}td,th{text-align:left;
border-bottom:1px solid #e5eae5;padding:10px}th{color:#657067;font-size:12px}
</style><main><h1>AutoResearch</h1><p class="muted">只读监控 · 合成数据 smoke · 本地记录未认证</p>
<section class="card"><h2>当前状态</h2><p id="status">等待记录</p></section>
<section class="card"><h2>每轮 macro-F1</h2><svg id="chart" viewBox="0 0 800 180"
role="img" aria-label="每轮 macro-F1 趋势"></svg><p id="empty" class="muted">暂无评分</p></section>
<section class="card"><h2>实验记录</h2><table><thead><tr><th>Commit</th><th>分数</th>
<th>决定</th><th>假设</th><th>显存 MB</th></tr></thead><tbody id="rows"></tbody></table></section></main>
<script>
async function refresh(){try{const r=await fetch('/api',{cache:'no-store'});
if(!r.ok)throw Error('HTTP '+r.status);const d=await r.json();
document.getElementById('status').textContent=d.status+' · '+d.evidence;
const rows=document.getElementById('rows');rows.replaceChildren();
for(const item of d.results){const tr=document.createElement('tr');
for(const v of [item.commit,item.macro_f1,item.decision,item.description,item.peak_vram_mb]){
const td=document.createElement('td');td.textContent=v||'—';tr.append(td)}rows.append(tr)}
const svg=document.getElementById('chart');svg.replaceChildren();
const values=d.results.filter(x=>x.macro_f1.trim()!=='').map(x=>Number(x.macro_f1)).filter(Number.isFinite);
document.getElementById('empty').hidden=values.length>0;
if(values.length){const lo=Math.min(...values),hi=Math.max(...values),span=hi-lo||1;
const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');
line.setAttribute('points',values.map((v,i)=>`${20+i*760/Math.max(values.length-1,1)},${160-(v-lo)*140/span}`).join(' '));
line.setAttribute('fill','none');line.setAttribute('stroke','#28734f');line.setAttribute('stroke-width','3');svg.append(line)}
}catch(e){document.getElementById('status').textContent='监控读取失败：'+e.message}}
refresh();setInterval(refresh,3000);
</script></html>'''


def _read_file(path: Path, limit: int) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("monitor input is not a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("monitor input exceeds the limit")
        return data.decode("utf-8")
    finally:
        os.close(fd)


def read_run(root: Path) -> dict:
    status = "暂无运行状态"
    evidence = "NOT_AVAILABLE"
    try:
        item = json.loads(_read_file(root / "status.json", 16 * 1024))
        state = item.get("state", "NOT_AVAILABLE")
        evidence = item.get("evidence", "NOT_AVAILABLE")
        if state in ("PREPARING", "TRAINING") and time.time() > float(item.get("deadline", 0)):
            state = "STALE"
        status = str(state)
        if item.get("fold") is not None:
            status += f" · 折 {item['fold']}/{item.get('folds', '?')}"
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError):
        status = "状态不可读"
    rows = []
    path = root / "results.tsv"
    try:
        text = _read_file(path, 2 * 1024 * 1024)
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))[-500:]
    except FileNotFoundError:
        pass
    return {"status": status, "evidence": evidence, "results": [
        {key: str(row.get(key) or "")[:400] for key in
         ("commit", "macro_f1", "decision", "description", "peak_vram_mb")}
        for row in rows
    ]}


def make_server(root: Path, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            host = self.headers.get("Host", "").lower()
            port = self.server.server_port
            allowed = {"localhost", "127.0.0.1", f"localhost:{port}", f"127.0.0.1:{port}"}
            if host not in allowed:
                self.send_error(403)
                return
            if self.path == "/":
                payload, content_type = PAGE.encode(), "text/html; charset=utf-8"
            elif self.path == "/api":
                try:
                    payload = json.dumps(read_run(root), ensure_ascii=False).encode()
                except (OSError, ValueError):
                    self.send_error(500, "Cannot read run records")
                    return
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    make_server(args.run.resolve(strict=True), args.port).serve_forever()


if __name__ == "__main__":
    main()
