import torch
import time
import itertools

def get_max_matrix_size(device_id):
    total_memory = torch.cuda.get_device_properties(device_id).total_memory
    free_memory = torch.cuda.memory_reserved(device_id) - torch.cuda.memory_allocated(device_id)
    safe_memory_use = free_memory * 0.7  # Use 70% of free memory to be safer
    bytes_per_element = 4  # float32 has 4 bytes
    num_elements = safe_memory_use // bytes_per_element // 3  # Divide by 3 for A, B, and result
    matrix_size = int(num_elements ** 0.5)  # Square root for square matrix
    return matrix_size

def compute_test(device_id, num_iterations):
    device = torch.device(f'cuda:{device_id}')
    matrix_size = get_max_matrix_size(device_id)
    A = torch.rand(matrix_size, matrix_size, device=device, dtype=torch.float32)
    B = torch.rand(matrix_size, matrix_size, device=device, dtype=torch.float32)

    start_time = time.time()
    for _ in range(num_iterations):
        A = torch.matmul(A, B)
    torch.cuda.synchronize(device=device)
    total_time = time.time() - start_time

    return total_time

def transfer_test(device_id, data_size_gb):
    device = torch.device(f'cuda:{device_id}')
    num_elements = (data_size_gb * (1024**3)) // 4
    data = torch.randn(num_elements, dtype=torch.float32)

    start_time = time.time()
    data_gpu = data.to(device)
    torch.cuda.synchronize(device=device)
    gpu_to_cpu_time = time.time() - start_time

    start_time = time.time()
    data_cpu = data_gpu.to('cpu')
    torch.cuda.synchronize(device=device)
    cpu_to_gpu_time = time.time() - start_time

    return gpu_to_cpu_time, cpu_to_gpu_time

def gpu_to_gpu_transfer_test(src_device_id, dst_device_id, data_size_gb):
    src_device = torch.device(f'cuda:{src_device_id}')
    dst_device = torch.device(f'cuda:{dst_device_id}')
    num_elements = (data_size_gb * (1024**3)) // 4
    data = torch.randn(num_elements, dtype=torch.float32, device=src_device)

    start_time = time.time()
    data.to(dst_device)
    torch.cuda.synchronize(dst_device)
    transfer_time = time.time() - start_time

    return transfer_time

def main():
    num_iterations = 50  # Iterations for compute test
    data_size_gb = 10  # Data size for transfer test
    compute_results = {}
    transfer_results = {}
    gpu_to_gpu_transfer_results = {}

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        for device_id in range(torch.cuda.device_count()):
            compute_time = compute_test(device_id, num_iterations)
            compute_results[device_id] = compute_time
            gpu_to_cpu_time, cpu_to_gpu_time = transfer_test(device_id, data_size_gb)
            transfer_results[device_id] = (gpu_to_cpu_time, cpu_to_gpu_time)

            print(f"Device {device_id} - Compute Time: {compute_time:.4f}s, "
                  f"GPU to CPU Transfer Time: {gpu_to_cpu_time:.4f}s, "
                  f"CPU to GPU Transfer Time: {cpu_to_gpu_time:.4f}s")
        # GPU-to-GPU transfer test for all pairs
        for src_id, dst_id in itertools.combinations(range(torch.cuda.device_count()), 2):
            transfer_time = gpu_to_gpu_transfer_test(src_id, dst_id, 1)  # 1 GB for quick test
            gpu_to_gpu_transfer_results[(src_id, dst_id)] = transfer_time
            print(f"Transfer from GPU {src_id} to GPU {dst_id}: {transfer_time:.4f}s")

        # Displaying fastest and slowest GPUs for compute and transfer
        fastest_gpu_compute = min(compute_results, key=compute_results.get)
        slowest_gpu_compute = max(compute_results, key=compute_results.get)
        print(f"\nFastest GPU (Compute): Device {fastest_gpu_compute}")
        print(f"Slowest GPU (Compute): Device {slowest_gpu_compute}")

        fastest_gpu_transfer = min(transfer_results, key=lambda x: sum(transfer_results[x]))
        slowest_gpu_transfer = max(transfer_results, key=lambda x: sum(transfer_results[x]))
        print(f"Fastest GPU (Transfer): Device {fastest_gpu_transfer}")
        print(f"Slowest GPU (Transfer): Device {slowest_gpu_transfer}")

    else:
        print("CUDA is not available or only one GPU detected.")

if __name__ == "__main__":
    main()
