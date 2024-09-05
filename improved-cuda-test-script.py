import torch
import time
import torch.nn as nn

def test_cuda_availability():
    """
    Test CUDA availability and print device information.
    """
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(torch.cuda.current_device())}")
    else:
        print("CUDA is not available. Please check your CUDA installation.")

def test_matmul(size=10000, iterations=10):
    """
    Test matrix multiplication performance.

    Args:
    size (int): Size of the square matrices
    iterations (int): Number of times to repeat the multiplication
    """
    print(f"\nTesting CUDA matrix multiplication (size: {size}x{size}, iterations: {iterations})...")
    
    A = torch.randn(size, size, device='cuda')
    B = torch.randn(size, size, device='cuda')
    
    start_time = time.time()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()  # Ensure all CUDA operations are completed
    end_time = time.time()
    
    print(f"Matrix multiplication on GPU took {end_time - start_time:.6f} seconds")

def test_basic_operations(size=10000000, iterations=100):
    """
    Test basic CUDA operations.

    Args:
    size (int): Size of the vectors
    iterations (int): Number of times to repeat the operations
    """
    print(f"\nTesting basic CUDA operations (vector size: {size}, iterations: {iterations})...")
    
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    
    start_time = time.time()
    for _ in range(iterations):
        z = x + y
        z = x * y
        z = torch.sin(x)
        z = torch.exp(y)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Basic operations on GPU took {end_time - start_time:.6f} seconds")

def test_convolution(size=1024, channels=64, kernel_size=3, iterations=1000):
    """
    Test 2D convolution performance.

    Args:
    size (int): Size of the square input
    channels (int): Number of input and output channels
    kernel_size (int): Size of the convolution kernel
    iterations (int): Number of times to repeat the convolution
    """
    print(f"\nTesting CUDA 2D convolution (size: {size}x{size}, channels: {channels}, kernel: {kernel_size}x{kernel_size}, iterations: {iterations})...")
    
    input_tensor = torch.randn(1, channels, size, size, device='cuda')
    conv_layer = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2).cuda()
    
    start_time = time.time()
    for _ in range(iterations):
        output = conv_layer(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"2D convolution on GPU took {end_time - start_time:.6f} seconds")

def test_lstm(seq_length=1000, input_size=512, hidden_size=512, num_layers=2, batch_size=64, iterations=100):
    """
    Test LSTM performance.

    Args:
    seq_length (int): Length of the input sequence
    input_size (int): Size of input features
    hidden_size (int): Size of hidden state
    num_layers (int): Number of LSTM layers
    batch_size (int): Batch size
    iterations (int): Number of times to repeat the forward pass
    """
    print(f"\nTesting CUDA LSTM (seq_length: {seq_length}, input_size: {input_size}, hidden_size: {hidden_size}, layers: {num_layers}, batch_size: {batch_size}, iterations: {iterations})...")
    
    input_tensor = torch.randn(seq_length, batch_size, input_size, device='cuda')
    lstm = nn.LSTM(input_size, hidden_size, num_layers).cuda()
    
    start_time = time.time()
    for _ in range(iterations):
        output, _ = lstm(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"LSTM forward pass on GPU took {end_time - start_time:.6f} seconds")

def main():
    torch.cuda.empty_cache()  # Clear GPU cache before starting tests
    
    test_cuda_availability()
    test_matmul()
    test_basic_operations()
    test_convolution()
    test_lstm()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()
