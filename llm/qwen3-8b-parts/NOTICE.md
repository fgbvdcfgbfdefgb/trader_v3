# Qwen3-8B Q4_K_M GGUF (split into <100 MB parts)

- Source model: Qwen/Qwen3-8B (Apache-2.0 license, (c) Alibaba Cloud)
- GGUF quantization: bartowski/Qwen_Qwen3-8B-GGUF (file: Qwen_Qwen3-8B-Q4_K_M.gguf)
- Exact file: https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q4_K_M.gguf
- Size: 5,027,784,224 bytes
- Full-file sha256: see the first line of SHA256SUMS.txt

GitHub caps single files at 100 MB, so the GGUF is stored here as 51
`part-*` files. Re-assemble with:

    python3 scripts/assemble_model.py

(concatenates the parts and verifies every sha256 checksum).
