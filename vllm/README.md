# Deploying a Chat Completion Model with vLLM Truss

This repository provides two approaches for deploying OpenAI-compatible chat completion models using vLLM and Truss. Select the option that best suits your use case.

---

## Deployment Options

### 1. **vLLM Server via `vllm serve` (Strongly Recommended)**

**Overview:**
Leverage the built-in vLLM server for an OpenAI-compatible, codeless deployment. This is the recommended method for most users who want a fast, production-ready setup.

**How to Use:**
- See the [`vllm_server`](./vllm_server) directory for more details and instructions.

**Why use this?**
- Minimal setup, codeless solution
- OpenAI-compatible

---

### 2. **vLLM with Truss Server**

**Overview:**
For advanced users who need custom inference logic, additional pre/post-processing, or further flexibility.

**How to Use:**
- Refer to the [`truss_server`](./truss_server) directory for details and configuration examples.

**Why use this?**
- Fully customizable inference and server logic
- OpenAI-compatible with minimal client changes
