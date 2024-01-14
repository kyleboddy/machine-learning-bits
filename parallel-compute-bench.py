import torch
import time
import itertools
import multiprocessing

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
    
    # Increase the matrix size for a longer compute time
    matrix_size = 5000  # Example size, adjust as needed

    A = torch.rand(matrix_size, matrix_size, device=device, dtype=torch.float32)
    B = torch.rand(matrix_size, matrix_size, device=device, dtype=torch.float32)

    start_time = time.time()
    for _ in range(num_iterations):
        # Increase the number of iterations for a longer compute time
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

    # Calculate the number of elements correctly
    bytes_per_element = 4  # Assuming float32, which has 4 bytes
    num_elements = int(data_size_gb * (1024**3) / bytes_per_element)

    # Creating a 1-D tensor of the specified size
    data = torch.randn((num_elements,), dtype=torch.float32, device=src_device)

    start_time = time.time()
    data.to(dst_device)
    torch.cuda.synchronize(dst_device)
    transfer_time = time.time() - start_time

    return transfer_time

def run_gpu_to_gpu_transfer_test(src_device_id, dst_device_id, data_size_gb, results_queue):
    transfer_time = gpu_to_gpu_transfer_test(src_device_id, dst_device_id, data_size_gb)
    results_queue.put((src_device_id, dst_device_id, transfer_time))

def run_tests_on_device(device_id, num_iterations, data_size_gb, results_queue):
    torch.cuda.set_device(device_id)

    print(f"Starting tests on Device {device_id}")

    compute_time = compute_test(device_id, num_iterations)
    print(f"[Device {device_id}] Compute test completed in {compute_time:.4f}s")

    gpu_to_cpu_time, cpu_to_gpu_time = transfer_test(device_id, data_size_gb)
    print(f"[Device {device_id}] Transfer test completed: GPU to CPU {gpu_to_cpu_time:.4f}s, CPU to GPU {cpu_to_gpu_time:.4f}s")

    results_queue.put((device_id, compute_time, gpu_to_cpu_time, cpu_to_gpu_time))
    print(f"Results updated for Device {device_id}")

def main():
    num_iterations = 50
    data_size_gb = 10

    # Set start method to 'spawn' for multiprocessing
    multiprocessing.set_start_method('spawn')

    print("Checking CUDA availability and device count...")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"CUDA is available with {torch.cuda.device_count()} devices. Starting tests...")
        processes = []
        results_queue = multiprocessing.Queue()

        for device_id in range(torch.cuda.device_count()):
            p = multiprocessing.Process(target=run_tests_on_device, args=(device_id, num_iterations, data_size_gb, results_queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        results = {}
        while not results_queue.empty():
            device_id, compute_time, gpu_to_cpu_time, cpu_to_gpu_time = results_queue.get()
            results[device_id] = {
                "compute_time": compute_time,
                "gpu_to_cpu_time": gpu_to_cpu_time,
                "cpu_to_gpu_time": cpu_to_gpu_time
            }

        # Output results after all processes complete
        print("Final Results:")
        for device_id, result in results.items():
            print(f"Device {device_id} - Compute Time: {result['compute_time']:.4f}s, "
                  f"GPU to CPU Transfer Time: {result['gpu_to_cpu_time']:.4f}s, "
                  f"CPU to GPU Transfer Time: {result['cpu_to_gpu_time']:.4f}s")

        # Starting GPU-to-GPU transfer tests in parallel
        print("Starting GPU-to-GPU transfer tests in parallel...")
        gpu_transfer_processes = []
        transfer_results_queue = multiprocessing.Queue()
        data_size_gb_for_transfer = 1.5  # Example: Increase the data size to 5GB

        for src_id, dst_id in itertools.combinations(range(torch.cuda.device_count()), 2):
            p = multiprocessing.Process(target=run_gpu_to_gpu_transfer_test, args=(src_id, dst_id, data_size_gb_for_transfer, transfer_results_queue))
            gpu_transfer_processes.append(p)
            p.start()

        for p in gpu_transfer_processes:
            p.join()

        while not transfer_results_queue.empty():
            src_id, dst_id, transfer_time = transfer_results_queue.get()
            print(f"Transfer from GPU {src_id} to GPU {dst_id} completed in {transfer_time:.4f}s")

        # Analyzing fastest and slowest GPUs for compute and transfer
        print("Analyzing fastest and slowest GPUs for compute and transfer...")
        fastest_gpu_compute = min(results, key=lambda x: results[x]['compute_time'])
        slowest_gpu_compute = max(results, key=lambda x: results[x]['compute_time'])
        print(f"Fastest GPU (Compute): Device {fastest_gpu_compute} with time {results[fastest_gpu_compute]['compute_time']:.4f}s")
        print(f"Slowest GPU (Compute): Device {slowest_gpu_compute} with time {results[slowest_gpu_compute]['compute_time']:.4f}s")

        fastest_gpu_transfer = min(results, key=lambda x: results[x]['gpu_to_cpu_time'] + results[x]['cpu_to_gpu_time'])
        slowest_gpu_transfer = max(results, key=lambda x: results[x]['gpu_to_cpu_time'] + results[x]['cpu_to_gpu_time'])
        print(f"Fastest GPU (Transfer): Device {fastest_gpu_transfer} with total transfer time {results[fastest_gpu_transfer]['gpu_to_cpu_time'] + results[fastest_gpu_transfer]['cpu_to_gpu_time']:.4f}s")
        print(f"Slowest GPU (Transfer): Device {slowest_gpu_transfer} with total transfer time {results[slowest_gpu_transfer]['gpu_to_cpu_time'] + results[slowest_gpu_transfer]['cpu_to_gpu_time']:.4f}s")

    else:
        print("CUDA is not available or only one GPU detected.")




if __name__ == "__main__":
    main()
