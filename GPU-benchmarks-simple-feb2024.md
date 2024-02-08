# Aeternum Tests Overview

## Server Specifications
- **CPU**: Ryzen Threadripper 2970WX CPU
- **RAM**: 128GB DDR4 RAM in XMP1, JEDEC maximum
- **Storage**: EVO 980 PRO 2TB NVME m2
- **GPU Configuration**:
  - GPU 0 + 1: EVGA RTX 3090, 420W triple power connector
  - GPU 2 + 3: Founder Edition RTX 3090s
  - NVLink Configuration:
    - No NVLink on GPU 0 (x8), GPU 1 (x16)
    - NVLink on GPU 2 (x8) + GPU 3 (x16)
  - All GPUs are at gen3 speeds, using either ROG Strix or Cirrascale Custom Riser Cables (short run)
- **Models**: LoneStriker_miqu-1-70b-sf-4.25bpw-h6-exl2 running on text-generation-webui

---

## Test One: x8 Speeds with 70B Model
- **Subject**: x8 speeds with 70B model
- **Loader**: ExLlamav2_HF
- **GPU Topology**: GPU 0 (x8), GPU 2 (x8)
- **GPU-Split Specs**: 20,0,20,0
- **Model Loading Time**: 8.51 seconds
- **Output Generation Time**: 16.84 seconds
- **Output Speed**: 16.03 tokens/s

---

## Test Two: x16 Speeds with 70B Model
- **Subject**: x16 speeds with 70B model
- **Loader**: ExLlamav2_HF
- **GPU Topology**: GPU 1 (x16), GPU 3 (x16)
- **GPU-Split Specs**: 0,20,0,20
- **Model Loading Time**: 7.81 seconds
- **Output Generation Time**: 18.65 seconds
- **Output Speed**: 15.93 tokens/s

---

## Test Three: Mixed PCIe Lanes with 70B Model
- **Subject**: Mixed PCIe lanes with 70B model
- **Loader**: ExLlamav2_HF
- **GPU Topology**: GPU 1 (x16), GPU 2 (x8)
- **GPU-Split Specs**: 0,20,20,0
- **Model Loading Time**: 8.53 seconds
- **Output Generation Time**: 10.05 seconds
- **Output Speed**: 15.42 tokens/s

---

## Test Four: Mixed PCIe Lanes + NVLink Enabled with 70B Model
- **Subject**: Mixed PCIe lanes + NVLink enabled with 70B model
- **Loader**: ExLlamav2_HF
- **GPU Topology**: GPU 2 (x8), GPU 3 (x16) -> [NVLink bridge]
- **GPU-Split Specs**: 0,0,20,20
- **Model Loading Time**: 8.39 seconds
- **Output Generation Time**: 15.00 seconds
- **Output Speed**: 16.14 tokens/s

---

## Test Five: All Four Mixed PCIe Lanes + NVLink Enabled with 70B Model
- **Subject**: All four mixed PCIe lanes + NVLink enabled with 70B model
- **Loader**: ExLlamav2_HF
- **GPU Topology**: GPU 0 (x8), GPU 1 (x16), GPU 2 (x8), GPU 3 (x16) -> [NVLink bridge on 2 + 3]
- **GPU-Split Specs**: 8,8,8,8
- **Model Loading Time**: 8.63 seconds
- **Output Generation Time**: 9.56 seconds
- **Output Speed**: 15.37 tokens/s
