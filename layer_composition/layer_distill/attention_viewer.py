from __future__ import annotations

import argparse
import json
import math
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

import torch

DEFAULT_MODEL = "EleutherAI/pythia-31m"
DEFAULT_MAX_LENGTH = 96
MAX_ALLOWED_LENGTH = 384


@dataclass
class AttentionModelBundle:
    model_name: str
    tokenizer: Any
    model: Any
    device: torch.device
    dtype: str


_BUNDLE: AttentionModelBundle | None = None
_BUNDLE_LOCK = threading.Lock()


def _select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return torch.device(name)


def _torch_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError("dtype must be fp32, fp16, or bf16")


def get_model_bundle(
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    dtype: str = "fp32",
) -> AttentionModelBundle:
    global _BUNDLE
    with _BUNDLE_LOCK:
        if _BUNDLE is not None and _BUNDLE.model_name == model_name and _BUNDLE.dtype == dtype and str(_BUNDLE.device) == str(_select_device(device)):
            return _BUNDLE

        from transformers import AutoModelForCausalLM, AutoTokenizer

        selected_device = _select_device(device)
        torch_dtype = _torch_dtype(dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, attn_implementation="eager")
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        model.to(selected_device)
        model.eval()
        _BUNDLE = AttentionModelBundle(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            device=selected_device,
            dtype=dtype,
        )
        return _BUNDLE


def _config_value(bundle: AttentionModelBundle, *names: str, default: int = 0) -> int:
    config = getattr(bundle.model, "config", None)
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return default


def model_metadata(bundle: AttentionModelBundle) -> dict[str, Any]:
    return {
        "name": bundle.model_name,
        "device": str(bundle.device),
        "dtype": bundle.dtype,
        "layers": _config_value(bundle, "num_hidden_layers", "n_layer", default=0),
        "heads": _config_value(bundle, "num_attention_heads", "n_head", default=0),
        "model_type": str(getattr(getattr(bundle.model, "config", None), "model_type", "")),
        "max_length": DEFAULT_MAX_LENGTH,
        "max_allowed_length": MAX_ALLOWED_LENGTH,
    }


def _token_entries(tokenizer: Any, input_ids: torch.Tensor) -> list[dict[str, Any]]:
    ids = [int(value) for value in input_ids.detach().cpu().tolist()]
    raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    entries = []
    for index, (token_id, raw) in enumerate(zip(ids, raw_tokens, strict=True)):
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        display = text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        if display == "":
            display = str(raw)
        entries.append({"index": index, "id": token_id, "text": display, "raw": str(raw)})
    return entries


def summarize_attention_rows(matrix: torch.Tensor, tokens: list[dict[str, Any]], *, top_k: int = 5, eps: float = 1e-12) -> list[dict[str, Any]]:
    if matrix.ndim != 2:
        raise ValueError("attention matrix must be 2D")
    if matrix.shape[0] != len(tokens) or matrix.shape[1] != len(tokens):
        raise ValueError("attention matrix shape must match token count")
    probs = matrix.float().clamp_min(0.0)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    probs = probs / denom
    k = min(top_k, probs.shape[-1])
    rows: list[dict[str, Any]] = []
    for query_idx, row in enumerate(probs):
        values, indices = torch.topk(row, k=k)
        entropy = -(row * (row + eps).log()).sum()
        rows.append(
            {
                "query_index": query_idx,
                "query_token": tokens[query_idx]["text"],
                "entropy": float(entropy.item()),
                "max_weight": float(values[0].item()) if k else 0.0,
                "top_keys": [
                    {
                        "token_index": int(key_idx.item()),
                        "token": tokens[int(key_idx.item())]["text"],
                        "weight": float(weight.item()),
                    }
                    for weight, key_idx in zip(values, indices, strict=True)
                    if float(weight.item()) > 0.0
                ],
            }
        )
    return rows


def build_attention_payload(
    text: str,
    *,
    layer: int,
    head: int,
    max_length: int = DEFAULT_MAX_LENGTH,
    bundle: AttentionModelBundle | None = None,
) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("text must be non-empty")
    if max_length <= 0 or max_length > MAX_ALLOWED_LENGTH:
        raise ValueError(f"max_length must be between 1 and {MAX_ALLOWED_LENGTH}")
    bundle = bundle if bundle is not None else get_model_bundle()
    meta = model_metadata(bundle)
    layers = int(meta["layers"])
    heads = int(meta["heads"])
    if layer < 0 or layer >= layers:
        raise ValueError(f"layer must be between 0 and {layers - 1}")
    if head < 0 or head >= heads:
        raise ValueError(f"head must be between 0 and {heads - 1}")

    encoded = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(bundle.device)
    if input_ids.shape[-1] == 0:
        raise ValueError("text produced no tokens")

    with torch.inference_mode():
        outputs = bundle.model(input_ids=input_ids, use_cache=False, output_attentions=True)
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("model did not return attentions")

    selected = attentions[layer][0, head].detach().float().cpu()
    tokens = _token_entries(bundle.tokenizer, input_ids[0])
    matrix = selected.tolist()
    rows = summarize_attention_rows(selected, tokens)
    seq_len = len(tokens)
    return {
        "model": meta,
        "selection": {"layer": layer, "head": head},
        "text": {"characters": len(text), "tokens": seq_len, "truncated_to": max_length},
        "tokens": tokens,
        "attention": matrix,
        "rows": rows,
        "stats": {
            "mean_entropy": float(sum(row["entropy"] for row in rows) / max(seq_len, 1)),
            "mean_max_weight": float(sum(row["max_weight"] for row in rows) / max(seq_len, 1)),
        },
    }


def _json_response(status: int, payload: dict[str, Any]) -> tuple[int, dict[str, str], str]:
    return status, {"Content-Type": "application/json"}, json.dumps(payload)


def handle_api_request(
    path: str,
    body: bytes,
    *,
    bundle_provider: Callable[[], AttentionModelBundle] = get_model_bundle,
) -> tuple[int, dict[str, str], str]:
    try:
        if path == "/api/model":
            return _json_response(200, model_metadata(bundle_provider()))
        if path == "/api/attention":
            request = json.loads(body.decode("utf-8") or "{}")
            payload = build_attention_payload(
                str(request.get("text", "")),
                layer=int(request.get("layer", 0)),
                head=int(request.get("head", 0)),
                max_length=int(request.get("max_length", DEFAULT_MAX_LENGTH)),
                bundle=bundle_provider(),
            )
            return _json_response(200, payload)
        return _json_response(404, {"error": "not found"})
    except (ValueError, json.JSONDecodeError) as exc:
        return _json_response(400, {"error": str(exc)})
    except Exception as exc:
        return _json_response(500, {"error": str(exc)})


def index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pythia Attention Head Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f8f9f4;
      --panel: #ffffff;
      --ink: #17201a;
      --muted: #667064;
      --line: #d8ded3;
      --accent: #126b5f;
      --hot: #d9483b;
      --mid: #e6b450;
      --cold: #eef2ea;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 430px) minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 22px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
    }
    textarea, select, input, button {
      width: 100%;
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
    }
    textarea {
      min-height: 220px;
      resize: vertical;
      padding: 10px;
      line-height: 1.35;
    }
    select, input {
      height: 36px;
      padding: 0 9px;
    }
    button {
      height: 38px;
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.62;
      cursor: wait;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .meta div {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcf8;
    }
    .meta strong {
      display: block;
      color: var(--ink);
      font-size: 14px;
      margin-top: 2px;
      overflow-wrap: anywhere;
    }
    .workspace {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 0;
    }
    .toolbar {
      height: 52px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      gap: 18px;
      padding: 0 18px;
      background: #fbfcf8;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .viz {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
    }
    .heatmap-wrap {
      min-width: 0;
      overflow: auto;
      padding: 18px;
    }
    .heatmap-stage {
      display: grid;
      grid-template-columns: 70px auto;
      grid-template-rows: 58px auto;
      width: max-content;
      min-width: 100%;
    }
    .x-labels, .y-labels {
      color: var(--muted);
      font-size: 11px;
    }
    .x-labels {
      grid-column: 2;
      display: grid;
      align-items: end;
      height: 58px;
    }
    .x-labels span {
      transform: rotate(-45deg);
      transform-origin: bottom left;
      width: 86px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .y-labels {
      grid-row: 2;
      display: grid;
      align-items: center;
      padding-right: 8px;
    }
    .y-labels span {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      text-align: right;
    }
    canvas {
      grid-column: 2;
      grid-row: 2;
      image-rendering: pixelated;
      border: 1px solid var(--line);
      background: #fff;
    }
    .details {
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      overflow: auto;
    }
    .details h2 {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0;
    }
    .row-list {
      display: grid;
      gap: 8px;
    }
    .row {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcf8;
      font-size: 12px;
    }
    .row strong {
      display: block;
      font-size: 13px;
      margin-bottom: 5px;
    }
    .err { color: #a12622; font-weight: 700; }
    @media (max-width: 900px) {
      main, .viz { grid-template-columns: 1fr; }
      aside, .details { border: 0; border-bottom: 1px solid var(--line); }
      .details { max-height: 360px; }
    }
  </style>
</head>
<body>
<main>
  <aside>
    <div>
      <h1>Pythia 31M Attention</h1>
      <div id="status" class="err"></div>
    </div>
    <div>
      <label for="prompt-input">Prompt</label>
      <textarea id="prompt-input" spellcheck="false">The city council refused the demonstrators a permit because they feared violence.</textarea>
    </div>
    <div class="grid">
      <div>
        <label for="layer-select">Layer</label>
        <select id="layer-select"></select>
      </div>
      <div>
        <label for="head-select">Head</label>
        <select id="head-select"></select>
      </div>
      <div>
        <label for="max-length">Tokens</label>
        <input id="max-length" type="number" min="1" max="384" value="96">
      </div>
    </div>
    <button id="run-button">Render Head</button>
    <div class="meta">
      <div>Model<strong id="model-name">loading</strong></div>
      <div>Device<strong id="device-name">-</strong></div>
      <div>Tokens<strong id="token-count">-</strong></div>
      <div>Mean max weight<strong id="mean-max">-</strong></div>
    </div>
  </aside>
  <section class="workspace">
    <div class="toolbar" id="toolbar">Query tokens run down the left. Key tokens run across the top.</div>
    <div class="viz">
      <div class="heatmap-wrap">
        <div class="heatmap-stage">
          <div class="x-labels" id="x-labels"></div>
          <div class="y-labels" id="y-labels"></div>
          <canvas id="attention-canvas" width="640" height="640"></canvas>
        </div>
      </div>
      <div class="details">
        <h2>Top attended keys</h2>
        <div id="row-list" class="row-list"></div>
      </div>
    </div>
  </section>
</main>
<script>
const state = { payload: null, cell: 28 };
const $ = (id) => document.getElementById(id);

function color(weight) {
  const clamped = Math.max(0, Math.min(1, weight));
  const stops = [
    [238, 242, 234],
    [230, 180, 80],
    [217, 72, 59]
  ];
  const scaled = clamped * 2;
  const base = Math.min(1, Math.floor(scaled));
  const t = scaled - base;
  const a = stops[base], b = stops[Math.min(base + 1, 2)];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)}, ${Math.round(a[1] + (b[1] - a[1]) * t)}, ${Math.round(a[2] + (b[2] - a[2]) * t)})`;
}

function setStatus(message, isError = false) {
  $("status").textContent = message || "";
  $("status").className = isError ? "err" : "";
}

async function loadModel() {
  const response = await fetch("/api/model");
  const meta = await response.json();
  if (!response.ok) throw new Error(meta.error || "failed to load model metadata");
  $("model-name").textContent = meta.name;
  $("device-name").textContent = `${meta.device} ${meta.dtype}`;
  $("max-length").max = meta.max_allowed_length;
  for (let i = 0; i < meta.layers; i++) $("layer-select").append(new Option(String(i), String(i)));
  for (let i = 0; i < meta.heads; i++) $("head-select").append(new Option(String(i), String(i)));
}

function draw(payload) {
  state.payload = payload;
  const tokens = payload.tokens;
  const matrix = payload.attention;
  const cell = Math.max(18, Math.min(34, Math.floor(760 / Math.max(tokens.length, 1))));
  state.cell = cell;
  const size = cell * tokens.length;
  const canvas = $("attention-canvas");
  canvas.width = size;
  canvas.height = size;
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
  const ctx = canvas.getContext("2d", { alpha: false });
  for (let y = 0; y < tokens.length; y++) {
    for (let x = 0; x < tokens.length; x++) {
      ctx.fillStyle = color(matrix[y][x]);
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
  ctx.strokeStyle = "#d8ded3";
  ctx.lineWidth = 1;
  for (let i = 0; i <= tokens.length; i++) {
    ctx.beginPath();
    ctx.moveTo(i * cell + 0.5, 0);
    ctx.lineTo(i * cell + 0.5, size);
    ctx.moveTo(0, i * cell + 0.5);
    ctx.lineTo(size, i * cell + 0.5);
    ctx.stroke();
  }
  $("x-labels").style.gridTemplateColumns = `repeat(${tokens.length}, ${cell}px)`;
  $("y-labels").style.gridTemplateRows = `repeat(${tokens.length}, ${cell}px)`;
  $("x-labels").replaceChildren(...tokens.map((token) => Object.assign(document.createElement("span"), { textContent: token.text })));
  $("y-labels").replaceChildren(...tokens.map((token) => Object.assign(document.createElement("span"), { textContent: token.text })));
  $("token-count").textContent = String(payload.text.tokens);
  $("mean-max").textContent = payload.stats.mean_max_weight.toFixed(3);
  $("toolbar").textContent = `Layer ${payload.selection.layer}, head ${payload.selection.head}`;
  $("row-list").replaceChildren(...payload.rows.map((row) => {
    const node = document.createElement("div");
    node.className = "row";
    const keys = row.top_keys.map((key) => `${key.token} ${key.weight.toFixed(3)}`).join("  ");
    node.innerHTML = `<strong>${row.query_index}: ${row.query_token}</strong>${keys}`;
    return node;
  }));
}

async function render() {
  $("run-button").disabled = true;
  setStatus("Rendering");
  try {
    const response = await fetch("/api/attention", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: $("prompt-input").value,
        layer: Number($("layer-select").value || 0),
        head: Number($("head-select").value || 0),
        max_length: Number($("max-length").value || 96)
      })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "render failed");
    draw(payload);
    setStatus("");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    $("run-button").disabled = false;
  }
}

$("attention-canvas").addEventListener("mousemove", (event) => {
  if (!state.payload) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / state.cell);
  const y = Math.floor((event.clientY - rect.top) / state.cell);
  const tokens = state.payload.tokens;
  if (x < 0 || y < 0 || x >= tokens.length || y >= tokens.length) return;
  const weight = state.payload.attention[y][x];
  $("toolbar").textContent = `${tokens[y].text} -> ${tokens[x].text}: ${weight.toFixed(4)}`;
});

$("run-button").addEventListener("click", render);
loadModel().then(render).catch((err) => setStatus(err.message, true));
</script>
</body>
</html>"""


class AttentionViewerHandler(BaseHTTPRequestHandler):
    bundle_provider: Callable[[], AttentionModelBundle] = get_model_bundle

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, {"Content-Type": "text/html; charset=utf-8"}, index_html())
            return
        if path == "/api/model":
            self._send(*handle_api_request(path, b"", bundle_provider=self.bundle_provider))
            return
        self._send(404, {"Content-Type": "text/plain"}, "not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self._send(*handle_api_request(path, body, bundle_provider=self.bundle_provider))

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, status: int, headers: dict[str, str], body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(host: str, port: int, *, model_name: str = DEFAULT_MODEL, device: str = "auto", dtype: str = "fp32") -> ThreadingHTTPServer:
    class ConfiguredHandler(AttentionViewerHandler):
        bundle_provider = staticmethod(lambda: get_model_bundle(model_name=model_name, device=device, dtype=dtype))

    return ThreadingHTTPServer((host, port), ConfiguredHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local Pythia attention-head viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port, model_name=args.model, device=args.device, dtype=args.dtype)
    print(f"serving {args.model} at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
