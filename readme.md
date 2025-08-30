# Harness MLOps Exploration

This repository is an experimental project to explore how **Harness** can be leveraged for **MLOps** workflows — from model training to deployment and monitoring.

## Hardware Setup

The experiments are running on my personal gaming rig with the following specs:

- **GPU:** NVIDIA RTX 3070 Ti  
- **CPU:** AMD Ryzen 7 5800X3D  
- **RAM:** 32 GB  

## Cluster Setup

To simulate a production-like environment locally:

1. Installed **K3s** (lightweight Kubernetes distribution) on the local system.  
2. Installed the **NVIDIA GPU Operator** on top of K3s to provide GPU kernel access for workloads.  

The NVIDIA GPU Operator allows Kubernetes pods to seamlessly use the GPU for ML training and inference.

For more details, refer to the official [NVIDIA GPU Operator documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html#rancher-kubernetes-engine-2).

---

🚀 The goal is to iterate on this setup and test how Harness can streamline MLOps pipelines on GPU-backed Kubernetes clusters.
