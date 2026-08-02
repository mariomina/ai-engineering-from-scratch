import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def timing_comparison():
    print("=== Timing: List vs NumPy ===\n")
    size = 1_000_000

    start = time.perf_counter()
    python_list = [x ** 2 for x in range(size)]
    list_time = time.perf_counter() - start

    start = time.perf_counter()
    numpy_array = np.arange(size) ** 2
    numpy_time = time.perf_counter() - start

    print(f"List comprehension: {list_time:.4f}s")
    print(f"NumPy:              {numpy_time:.4f}s")
    print(f"Speedup:            {list_time / numpy_time:.1f}x")


def inline_plotting():
    print("\n=== Inline Plotting ===\n")
    np.random.seed(42)
    x = np.linspace(0, 10, 200)
    y_noisy = np.sin(x) + np.random.normal(0, 0.2, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, np.sin(x), label="sin(x)")
    axes[0].plot(x, y_noisy, alpha=0.5, label="noisy")
    axes[0].legend()
    axes[1].hist(y_noisy - np.sin(x), bins=30, edgecolor="black")
    plt.tight_layout()
    plt.savefig("notebook_plot.png", dpi=100)
    print("Saved plot to notebook_plot.png")


def dataframe_display():
    print("\n=== DataFrame Display ===\n")
    df = pd.DataFrame({
        "model": ["Linear Regression", "Random Forest", "Neural Network", "XGBoost"],
        "accuracy": [0.72, 0.89, 0.94, 0.91],
        "train_time_sec": [0.1, 2.3, 45.6, 8.2],
        "parameters": [102, 50_000, 1_200_000, 25_000],
    })
    print(df.to_string(index=False))
    print(f"\nBest model: {df.loc[df['accuracy'].idxmax(), 'model']}")


def gpu_check():
    print("\n=== GPU Check ===\n")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch not installed")


if __name__ == "__main__":
    print(f"Colab VM: {sys.platform} / python {sys.version.split()[0]}\n")
    gpu_check()
    timing_comparison()
    inline_plotting()
    dataframe_display()
    print("\n=== Done ===")
