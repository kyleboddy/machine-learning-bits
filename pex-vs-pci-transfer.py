import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

def generate_data(num_elements):
    print("Generating random data...")
    data = np.random.randn(num_elements).astype(np.float32)
    print("Data generation complete.")
    return data

def benchmark_gpu_transfer(src_gpu, dst_gpu, data_size_gb, data, num_transfers=3):
    print(f"\nStarting transfer benchmark from GPU {src_gpu} to GPU {dst_gpu} with {data_size_gb} GB of data.")
    src_data = torch.from_numpy(data).cuda(src_gpu)

    transfer_times = []
    for _ in range(num_transfers):
        start_time = time.time()
        _ = src_data.clone().cuda(dst_gpu)
        torch.cuda.synchronize()
        transfer_time = round(time.time() - start_time, 1)
        transfer_times.append(transfer_time)

    print(f"Transfer from GPU {src_gpu} to GPU {dst_gpu} completed. Times: {transfer_times}")
    return np.mean(transfer_times)

def create_heatmap(matrix, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="viridis")
    plt.title(title)
    plt.xlabel('Destination GPU')
    plt.ylabel('Source GPU')
    plt.savefig(filename)
    print(f"Saved heatmap as {filename}")

def main():
    data_size_gb = 10
    num_elements = int(data_size_gb * (1024**3) // 4)  # 4 bytes per float32
    data = generate_data(num_elements)

    pex_gpus = list(range(3, 7))
    pcie_gpus = list(range(0, 3)) + [7]
    pex_matrix = np.zeros((len(pex_gpus), len(pex_gpus)))
    pcie_matrix = np.zeros((len(pcie_gpus), len(pcie_gpus)))

    print("Benchmarking GPUs on PEX Interconnect:")
    for i in pex_gpus:
        for j in pex_gpus:
            if i != j:
                pex_matrix[i-3][j-3] = benchmark_gpu_transfer(i, j, data_size_gb, data)

    print("\nBenchmarking GPUs on PCIe:")
    for i in pcie_gpus:
        for j in pcie_gpus:
            if i != j:
                pcie_matrix[pcie_gpus.index(i)][pcie_gpus.index(j)] = benchmark_gpu_transfer(i, j, data_size_gb, data)

    print("\nBenchmark completed.")
    create_heatmap(pex_matrix, "PEX Interconnect GPU Transfer Benchmark", "pex_benchmark.png")
    create_heatmap(pcie_matrix, "PCIe GPU Transfer Benchmark", "pcie_benchmark.png")

if __name__ == "__main__":
    main()


