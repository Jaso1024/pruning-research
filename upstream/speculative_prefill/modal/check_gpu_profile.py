import subprocess

import modal


app = modal.App("spec-prefill-gpu-profile-check")

image = modal.Image.from_registry(
    "nvidia/cuda:12.1.1-base-ubuntu22.04",
    add_python="3.11",
)


@app.function(image=image, gpu="L4", timeout=10 * 60)
def check_gpu() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


@app.local_entrypoint()
def main():
    print(check_gpu.remote())
