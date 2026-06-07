import os
import subprocess

import modal


REMOTE_ROOT = "/root/speculative_prefill"
SECRETS = [modal.Secret.from_local_environ(["HF_TOKEN"])] if "HF_TOKEN" in os.environ else []

app = modal.App("spec-prefill-embedding-norm")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".",
        remote_path=REMOTE_ROOT,
        copy=True,
        ignore=[
            ".git",
            "**/__pycache__",
            "**/*.pyc",
            "local",
        ],
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    secrets=SECRETS,
)
def run_latency(
    mode: str = "embedding_norm",
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    config: str = "configs/config_embedding_norm_p3.yaml",
    spec_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    input_len: int = 2048,
    output_len: int = 1,
    batch_size: int = 4,
    warmup_iters: int = 1,
    iters: int = 3,
) -> str:
    if mode not in {"baseline", "embedding_norm", "spec_prefill"}:
        raise ValueError("mode must be baseline, embedding_norm, or spec_prefill")

    cmd = [
        "python",
        "-m",
        "speculative_prefill.vllm_benchmarks.latency",
        "--model",
        model,
        "--enforce-eager",
        "--no-enable-chunked-prefill",
        "--tensor-parallel-size",
        "1",
        "--max_model_len",
        str(input_len + output_len + 16),
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--batch-size",
        str(batch_size),
        "--num-iters-warmup",
        str(warmup_iters),
        "--num-iters",
        str(iters),
    ]

    env = {"VLLM_USE_V1": "0"}
    if mode == "embedding_norm":
        cmd.extend(["--embedding-norm-prefill"])
        env["SPEC_CONFIG_PATH"] = config
    elif mode == "spec_prefill":
        cmd.extend(["--spec-prefill", "--spec-model", spec_model])
        env["SPEC_CONFIG_PATH"] = config

    import os

    merged_env = os.environ.copy()
    merged_env.update(env)

    result = subprocess.run(
        cmd,
        cwd=REMOTE_ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return result.stdout


@app.local_entrypoint()
def main(
    mode: str = "embedding_norm",
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    config: str = "configs/config_embedding_norm_p3.yaml",
    spec_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    input_len: int = 2048,
    output_len: int = 1,
    batch_size: int = 4,
    warmup_iters: int = 1,
    iters: int = 3,
):
    output = run_latency.remote(
        mode=mode,
        model=model,
        config=config,
        spec_model=spec_model,
        input_len=input_len,
        output_len=output_len,
        batch_size=batch_size,
        warmup_iters=warmup_iters,
        iters=iters,
    )
    print(output)
