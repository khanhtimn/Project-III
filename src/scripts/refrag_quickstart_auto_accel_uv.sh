#!/usr/bin/env bash
set -euo pipefail

# ===== Settings (customize) =====
ENC_MODEL="${ENC_MODEL:-roberta-base}"
DEC_MODEL="${DEC_MODEL:-meta-llama/Llama-3.2-3B}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"
TOPK="${TOPK:-4}"
K="${K:-32}"
P="${P:-0.25}"
CTX_MAX="${CTX_MAX:-1024}"
MAX_NEW="${MAX_NEW:-128}"
STEPS="${STEPS:-200}"
LR_RECON="${LR_RECON:-2e-5}"
LR_NEXT="${LR_NEXT:-2e-5}"
LR_POLICY="${LR_POLICY:-1e-4}"
# ================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ ! -f "src/refrag.py" ]]; then
  echo "ERROR: src/refrag.py not found in $PROJECT_DIR."
  exit 1
fi

# ---- Python & venv ----
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python 3 not found. Install Python 3.10+."
  exit 1
fi

# ---- Detect accelerator ----
OS="$(uname -s)"
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then HAS_NVIDIA=1; fi
HAS_ROCM=0
if command -v rocminfo >/dev/null 2>&1; then HAS_ROCM=1; fi
if [[ -d "/opt/rocm" ]]; then HAS_ROCM=1; fi

echo "Detected OS: $OS; NVIDIA: $HAS_NVIDIA; ROCm: $HAS_ROCM"

# ---- Install dependencies using uv sync ----
echo "Installing dependencies using uv sync..."
uv sync

# Perf/env niceties
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
export KMP_DUPLICATE_LIB_OK=TRUE


# 1) Toy corpus + index
mkdir -p data runs/index
# cat > data/wiki_lines.txt << 'EOF'
# Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London.
# The capital of France is Paris.
# Alan Turing proposed the Turing test in 1950.
# Penicillin is an antibiotic derived from Penicillium fungi.
# Large language models can use retrieval to augment their context.
# EOF

uv run python src/refrag.py index \
  --corpus data/corpus_large.txt \
  --index_dir runs/index \
  --embed_model "${EMBED_MODEL}"

# 2) Quick generate
uv run python src/refrag.py generate \
  --index_dir runs/index \
  --embed_model "${EMBED_MODEL}" \
  --enc "${ENC_MODEL}" \
  --dec "${DEC_MODEL}" \
  --question "Who discovered penicillin?" \
  --topk ${TOPK} \
  --k ${K} \
  --p ${P} \
  --ctx_max ${CTX_MAX} \
  --max_new ${MAX_NEW} \
  --temperature 0.0

uv run python src/refrag.py generate \
  --index_dir runs/index \
  --embed_model "${EMBED_MODEL}" \
  --enc "${ENC_MODEL}" \
  --dec "${DEC_MODEL}" \
  --question "Which river flows through City_19?" \
  --topk ${TOPK} \
  --k ${K} \
  --p ${P} \
  --ctx_max ${CTX_MAX} \
  --max_new ${MAX_NEW} \
  --temperature 0.0

# 3) CPT datasets
# cat > data/cpt_train.jsonl << 'EOF'
# {"id":"ex1","tokens":"Penicillin revolutionized medicine by enabling treatment of bacterial infections.","split":{"s":1024,"o":128}}
# {"id":"ex2","tokens":"Alan Turing's work laid the foundations of computer science and artificial intelligence.","split":{"s":1024,"o":128}}
# {"id":"ex3","tokens":"Paris is the capital and most populous city of France, known for art, fashion, and gastronomy.","split":{"s":1024,"o":128}}
# EOF

# 3A) Reconstruction
uv run python src/refrag.py cpt_recon \
  --train_json data/cpt_train.jsonl \
  --enc "${ENC_MODEL}" \
  --dec "${DEC_MODEL}" \
  --k 64 \
  --steps ${STEPS} \
  --lr ${LR_RECON} \
  --log_every 20 \
  --out_dir runs/cpt_recon

# 3B) Next-paragraph
uv run python src/refrag.py cpt_next \
  --train_json data/cpt_train.jsonl \
  --enc "${ENC_MODEL}" \
  --dec "${DEC_MODEL}" \
  --k 64 \
  --steps ${STEPS} \
  --lr ${LR_NEXT} \
  --expand_frac 0.25 \
  --log_every 20 \
  --load_dir runs/cpt_recon \
  --out_dir runs/cpt_next

# 4) Policy training
# cat > data/rag_train.jsonl << 'EOF'
# {"id":"q1","question":"Who discovered penicillin?","answers":["Alexander Fleming"]}
# {"id":"q2","question":"What is the capital of France?","answers":["Paris"]}
# EOF

uv run python src/refrag.py train_policy \
  --rag_json data/rag_train.jsonl \
  --index_dir runs/index \
  --embed_model "${EMBED_MODEL}" \
  --enc "${ENC_MODEL}" \
  --dec "${DEC_MODEL}" \
  --k 32 \
  --steps ${STEPS} \
  --lr ${LR_POLICY} \
  --p ${P} \
  --topk ${TOPK} \
  --log_every 20 \
  --out_dir runs/policy

echo "---- Generate with trained policy ----"
uv run python src/refrag.py generate \
  --index_dir runs/index \
  --embed_model "${EMBED_MODEL}" \
  --enc "${ENC_MODEL}" \
  --dec "${DEC_MODEL}" \
  --load_dir runs/policy \
  --question "Explain how penicillin was discovered and by whom." \
  --topk ${TOPK} --k ${K} --p ${P} --max_new 192

uv run python src/refrag.py generate \
  --index_dir runs/index \
  --embed_model "${EMBED_MODEL}" \
  --enc "${ENC_MODEL}" \
  --dec "${DEC_MODEL}" \
  --load_dir runs/policy \
  --question "Which river flows through City_19?" \
  --topk ${TOPK} \
  --k ${K} \
  --p ${P} \
  --ctx_max ${CTX_MAX} \
  --max_new ${MAX_NEW} \
  --temperature 0.0

echo "---- Generate with CPT-tuned full model ----"
uv run python src/refrag.py generate \
  --index_dir runs/index \
  --embed_model "${EMBED_MODEL}" \
  --enc "${ENC_MODEL}" \
  --dec "${DEC_MODEL}" \
  --load_dir runs/cpt_next \
  --question "Explain how penicillin was discovered and by whom." \
  --topk ${TOPK} --k ${K} --p ${P} --max_new 192

echo "✅ Done."
