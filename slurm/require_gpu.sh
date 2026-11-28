# Source from a Slurm job script, after `module load`:  source slurm/require_gpu.sh
#
# Stops the job if it landed on a different GPU model than our results use.
# All of Euler's instruction GPU nodes advertise the same gres (gpu:rtx4000ada),
# but some GPUs are RTX 4000 *SFF* Ada (lower bandwidth and clocks), so --gres
# alone cannot pin the model. Results from different models are not comparable.
#
# Override with REQUIRE_GPU="<exact nvidia-smi name>", or REQUIRE_GPU=any to skip.

REQUIRE_GPU="${REQUIRE_GPU:-NVIDIA RTX 4000 Ada Generation}"

if [[ "$REQUIRE_GPU" != "any" ]]; then
    _gpus="$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true)"
    _first="${CUDA_VISIBLE_DEVICES:-}"
    _first="${_first%%,*}"
    _gpu_name=""
    # If nvidia-smi sees several GPUs (devices not constrained), pick the one Slurm gave us.
    if [[ "$(printf '%s\n' "$_gpus" | grep -c .)" -gt 1 && "$_first" =~ ^[0-9]+$ ]]; then
        _gpu_name="$(printf '%s\n' "$_gpus" | awk -F', ' -v i="$_first" '$1 == i { print $2 }')"
    fi
    if [[ -z "$_gpu_name" ]]; then
        _gpu_name="$(printf '%s\n' "$_gpus" | head -1 | sed 's/^[0-9]*, //')"
    fi
    if [[ "$_gpu_name" != "$REQUIRE_GPU" ]]; then
        echo "WRONG GPU: got '${_gpu_name:-none}' on $(hostname), need '$REQUIRE_GPU'." >&2
        echo "Nothing was run. Resubmit with --exclude=$(hostname -s), or set REQUIRE_GPU." >&2
        exit 3
    fi
    echo "GPU check ok: $_gpu_name on $(hostname -s)"
    unset _gpus _first _gpu_name
fi
