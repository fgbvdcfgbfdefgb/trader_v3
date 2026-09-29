#!/bin/bash
# Sanity-check the bundled llama.cpp CPU binaries on this machine.
cd "$(dirname "$0")"/..
BIN=llm/llama-b11249/llama-server
if [ ! -x "$BIN" ]; then chmod +x "$BIN" 2>/dev/null || true; fi
if [ ! -f "$BIN" ]; then echo "missing $BIN"; exit 1; fi
echo "-- version --"
"$BIN" --version 2>&1 | head -3 || { echo "FAILED to run llama-server (check glibc >= 2.34 / libssl3 / libgomp)"; exit 1; }
echo "-- missing shared libs (empty = all good) --"
ldd "$BIN" 2>/dev/null | grep "not found" || echo "none missing"
