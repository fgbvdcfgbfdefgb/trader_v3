#!/usr/bin/env python3
"""trader_v3 training orchestrator (fully offline).

- detects GPUs (nvidia-smi): one agent per GPU, all training in parallel;
  on CPU-only machines it runs several CPU agents in parallel instead.
- auto-assembles the Qwen3-8B GGUF from repo parts (if needed),
- auto-builds the data cache from the bundled minute bars (if needed),
- starts a local llama.cpp server (CPU) running Qwen3-8B and waits for it,
- spawns the agent workers (python -m trader.train ...),
- keeps runs/overview.png refreshed and shuts everything down cleanly
  (agents checkpoint themselves, so you can stop/resume any time).

Usage:
  python3 run_training.py                 # auto agents, run forever
  python3 run_training.py --agents 4 --epochs 5000
  python3 run_training.py --no-llm        # fast smoke test without Qwen
"""
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from trader import config as C  # noqa: E402
from trader import plots  # noqa: E402


def detect_gpus():
    try:
        if not shutil.which("nvidia-smi"):
            return 0
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=20).stdout
        return len([l for l in out.splitlines() if l.strip().startswith("GPU ")])
    except Exception:
        return 0


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_health(port, timeout):
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def run_step(cmd, desc):
    print(f"[setup] {desc}: {' '.join(cmd[:3])} ...", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode != 0:
        print(f"[setup] FAILED: {desc}", flush=True)
        return False
    return True


def read_history(run_dir):
    path = os.path.join(run_dir, "metrics.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(float(json.loads(line)["profit"]))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="auto",
                    help="'auto' or a number of parallel agents")
    ap.add_argument("--epochs", type=int, default=0,
                    help="epochs per agent (0 = run forever)")
    ap.add_argument("--no-llm", action="store_true",
                    help="run without Qwen (stub advisor features)")
    ap.add_argument("--llm-url", default="",
                    help="use an already-running llama.cpp server instead "
                         "of starting one")
    ap.add_argument("--port", type=int, default=0,
                    help="port for the llama.cpp server (0 = auto)")
    ap.add_argument("--llm-threads", type=int, default=0,
                    help="CPU threads for llama.cpp (0 = auto)")
    ap.add_argument("--coin-mode", choices=["all", "random"], default="all")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--png-keep", default="all")
    ap.add_argument("--require-llm", action="store_true",
                    help="abort if the LLM server cannot start")
    args = ap.parse_args()

    # ------------------------------------------------------------- deps check
    try:
        import numpy  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as e:
        print(f"[error] missing python package: {e}\n"
              f"        this repo expects numpy + matplotlib "
              f"(preinstalled on the Snowflake ML image).", flush=True)
        return 1

    n_gpu = detect_gpus()
    n_cpu = os.cpu_count() or 2
    if args.agents == "auto":
        n_agents = n_gpu if n_gpu > 0 else max(1, min(4, (n_cpu - 1) // 3))
    else:
        n_agents = max(1, int(args.agents))
    print(f"[setup] GPUs detected: {n_gpu} | CPUs: {n_cpu} | "
          f"agents: {n_agents} (one per GPU when GPUs exist)", flush=True)

    # ------------------------------------------------------------- data cache
    if not os.path.exists(os.path.join(C.CACHE_DIR, "days.json")):
        if not run_step([sys.executable, "scripts/prepare_data.py"],
                        "building data cache (one-time)"):
            return 1
    else:
        print("[setup] data cache present", flush=True)

    # ------------------------------------------------------------- LLM server
    server = None
    llm_url = args.llm_url
    use_llm = not args.no_llm
    if args.no_llm:
        use_llm = False
        print("[setup] LLM disabled (--no-llm): advisor will use neutral "
              "features", flush=True)
    elif not llm_url:
        if not os.path.exists(C.MODEL_GGUF):
            if not run_step([sys.executable, "scripts/assemble_model.py"],
                            "assembling Qwen3-8B GGUF from repo parts"):
                if args.require_llm:
                    return 1
                print("[setup] model assembly failed - continuing without LLM",
                      flush=True)
                use_llm = False
        if use_llm:
            port = args.port or free_port()
            threads = args.llm_threads or max(2, n_cpu // 2)
            slots = n_agents + 1
            ctx = 4096 * slots
            os.makedirs(C.RUNS_DIR, exist_ok=True)
            log = open(os.path.join(C.RUNS_DIR, "llama_server.log"), "a")
            cmd = [C.LLAMA_SERVER_BIN, "-m", C.MODEL_GGUF,
                   "--host", "127.0.0.1", "--port", str(port),
                   "-t", str(threads), "-np", str(slots), "-c", str(ctx),
                   "--jinja", "--no-warmup"]
            print(f"[setup] starting llama.cpp server (CPU, {threads} threads,"
                  f" {slots} slots): {' '.join(cmd[:8])} ...", flush=True)
            try:
                server = subprocess.Popen(cmd, cwd=C.LLAMA_DIR, stdout=log,
                                          stderr=subprocess.STDOUT)
            except Exception as e:
                print(f"[setup] failed to start llama-server: {e}", flush=True)
                server = None
            if server is not None:
                print("[setup] waiting for Qwen3-8B to load "
                      "(can take a few minutes on CPU) ...", flush=True)
                if wait_health(port, 900):
                    llm_url = f"http://127.0.0.1:{port}"
                    print(f"[setup] LLM server ready at {llm_url}", flush=True)
                else:
                    print("[setup] LLM server did not become healthy - "
                          "see runs/llama_server.log", flush=True)
                    server.terminate()
                    server = None
            if server is None and args.require_llm:
                return 1
            if server is None:
                use_llm = False
    if not use_llm:
        llm_url = ""

    # ------------------------------------------------------------- spawn agents
    procs = []
    for i in range(n_agents):
        env = os.environ.copy()
        if n_gpu > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(i % n_gpu)
        cmd = [sys.executable, "-m", "trader.train",
               "--agent-id", str(i),
               "--seed", str(args.seed + 17 * i),
               "--epochs", str(args.epochs),
               "--coin-mode", args.coin_mode,
               "--png-keep", args.png_keep]
        if llm_url:
            cmd += ["--llm-url", llm_url]
        p = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)
        procs.append((i, p))
        print(f"[spawn] agent_{i} started (pid {p.pid})", flush=True)

    # ------------------------------------------------------------- monitor
    stopping = {"flag": False}

    def _term(signum, frame):
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _term)
    signal.signal(signal.SIGTERM, _term)

    last_png = 0.0
    print("[run] all agents running - Ctrl+C / SIGTERM to stop "
          "(progress is checkpointed every epoch and resumable)", flush=True)
    while not stopping["flag"]:
        alive = [(i, p) for i, p in procs if p.poll() is None]
        if not alive:
            break
        now = time.time()
        if now - last_png > 60:
            last_png = now
            agents_data = []
            for i, _ in procs:
                rd = os.path.join(C.RUNS_DIR, f"agent_{i}")
                agents_data.append((f"agent_{i}", read_history(rd), []))
            try:
                plots.save_overview_png(
                    os.path.join(C.RUNS_DIR, "overview.png"), agents_data)
            except Exception:
                pass
        time.sleep(5)

    if stopping["flag"]:
        print("\n[stop] signal received - asking agents to checkpoint ...",
              flush=True)
        for _, p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        deadline = time.time() + 120
        for _, p in procs:
            try:
                p.wait(timeout=max(1, deadline - time.time()))
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    rc = [p.returncode for _, p in procs]
    print(f"[stop] agents exited with codes {rc}", flush=True)

    if server is not None:
        print("[stop] stopping llama.cpp server ...", flush=True)
        try:
            server.terminate()
            server.wait(timeout=30)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass

    agents_data = []
    for i, _ in procs:
        rd = os.path.join(C.RUNS_DIR, f"agent_{i}")
        agents_data.append((f"agent_{i}", read_history(rd), []))
    try:
        plots.save_overview_png(os.path.join(C.RUNS_DIR, "overview.png"),
                                agents_data)
    except Exception:
        pass
    print("[done] final reports: runs/overview.png, runs/agent_*/"
          "progress.png, runs/agent_*/epochs/*.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
