# Datadog Integration for vLLM on Baseten

This guide explains how to integrate Datadog monitoring with vLLM models deployed on Baseten. By following this guide, you'll be able to monitor your vLLM model's performance metrics, trace requests, and gain deep insights into your LLM infrastructure.

## Overview

This integration allows you to:
- Monitor vLLM performance metrics (token throughput, latency, cache usage, etc.)
- Collect custom metrics and send them to Datadog
- Track model performance over time
- Set up alerts for performance degradation

## Prerequisites

- Baseten account
- Datadog account with API key
- Hugging Face access token (for model downloads)

## Architecture

The solution consists of two main components:

1. **Datadog Configuration**: Configuration files for the Datadog agent and vLLM integration
2. **Baseten Deployment**: Configuration to deploy your monitored model on Baseten with build commands that install the Datadog agent

## Step 1: Create the vLLM Configuration File

Create a `vllm_conf.yaml` file inside a `data/` directory to configure how Datadog collects vLLM metrics:

```bash
mkdir -p data
cat > data/vllm_conf.yaml << 'EOF'
instances:
  - openmetrics_endpoint: http://localhost:8000/metrics
    namespace: vllm
    metrics:
      - vllm.*
    tags:
      - env:production
      - service:truss-vllm
EOF
```

**Configuration Details:**
- `openmetrics_endpoint`: vLLM exposes Prometheus metrics at `/metrics`
- `namespace`: Prefix for all metrics (they'll appear as `vllm.*` in Datadog)
- `metrics`: Pattern to collect all vLLM metrics
- `tags`: Custom tags to organize your metrics in Datadog

**Note:** The `vllm_conf.yaml` must be placed in the `data/` directory so it gets copied to the correct location during the build process.

## Step 2: Create Baseten Configuration

Create a `config.yaml` file for your Baseten deployment:

```yaml
base_image:
  image: vllm/vllm-openai:v0.11.0

build_commands:
  - apt-get update
  - apt-get -y install apt-transport-https curl gnupg
  - curl -fsSL https://keys.datadoghq.com/DATADOG_APT_KEY_CURRENT.public | gpg --dearmor -o /usr/share/keyrings/datadog-archive-keyring.gpg
  - sh -c "echo 'deb [signed-by=/usr/share/keyrings/datadog-archive-keyring.gpg] https://apt.datadoghq.com/ stable 7' > /etc/apt/sources.list.d/datadog.list"
  - apt-get update
  - apt-get install -y datadog-agent
  - mkdir -p /etc/datadog-agent/conf.d/vllm.d
  - cp data/vllm_conf.yaml /etc/datadog-agent/conf.d/vllm.d/conf.yaml
  - rm -rf /etc/datadog-agent/conf.d/kubelet.d
  - mkdir -p /tmp/datadog-agent /var/log/datadog
  - chmod -R 777 /tmp/datadog-agent /var/log/datadog

docker_server:
  liveness_endpoint: /health
  predict_endpoint: /v1/chat/completions
  readiness_endpoint: /health
  server_port: 8000
  start_command: sh -c "export DD_API_KEY=$(cat /secrets/dd_api_key | tr -d '\n\r' | xargs) && mkdir -p /tmp/datadog-agent /var/log/datadog && /opt/datadog-agent/bin/agent/agent run 2>&1 & sleep 3 && HF_TOKEN=$(cat /secrets/hf_access_token) vllm serve Qwen/Qwen2.5-3B-Instruct --enable-prefix-caching --enable-chunked-prefill"

environment_variables:
  DD_SITE: "us5.datadoghq.com"  # Change to your Datadog site (e.g., datadoghq.com, datadoghq.eu)
  DD_HOSTNAME: "truss-vllm-server"
  DD_SERVICE: "truss-vllm"
  DD_ENV: "production"
  DD_RUN_PATH: "/tmp/datadog-agent"
  DD_AUTH_TOKEN_FILE_PATH: "/tmp/datadog-agent/auth_token"
  DD_INVENTORIES_CHECKS_ENABLED: "false"
  DD_OTLP_CONFIG_RECEIVER_PROTOCOLS_GRPC_ENDPOINT: ""
  DD_CLOUD_PROVIDER_METADATA: "[]"
  VLLM_LOGGING_LEVEL: WARNING

model_metadata:
  repo_id: Qwen/Qwen2.5-3B-Instruct
  example_model_input:
    messages:
      - role: system
        content: "You are a helpful assistant."
      - role: user
        content: "What does Tongyi Qianwen mean?"
    stream: false
    model: "Qwen/Qwen2.5-3B-Instruct"
    max_tokens: 512
    temperature: 0.6

model_name: your-model-name

resources:
  accelerator: L4
  use_gpu: true

runtime:
  predict_concurrency: 256

secrets:
  dd_api_key: null
  hf_access_token: null
```

### Configuration Breakdown

**Build Commands:**
- Use the official vLLM OpenAI-compatible image as the base
- Install the Datadog agent at build time (same steps as the previous Dockerfile approach)
- Copy `vllm_conf.yaml` from the `data/` directory to the agent config location
- Remove kubelet checks that aren't needed
- Create writable directories for the agent

**Base Image:**
- Uses the official vLLM OpenAI image directly from Docker Hub

**Start Command:**
- Reads Datadog API key from secrets
- Starts Datadog agent in the background
- Waits 3 seconds for agent initialization
- Starts vLLM server

**Environment Variables:**
- `DD_SITE`: Your Datadog site URL (check your Datadog account settings)
- `DD_HOSTNAME`: Identifier for this deployment in Datadog
- `DD_SERVICE`: Service name for grouping metrics
- `DD_ENV`: Environment tag (production, staging, dev)
- Path variables: Configure where the agent stores runtime data
- Disable unnecessary features to reduce log noise

**Secrets:**
- `dd_api_key`: Your Datadog API key (set in Baseten UI)
- `hf_access_token`: Your Hugging Face token (set in Baseten UI)

## Step 3: Set Up Secrets in Baseten

Before deploying, configure your secrets in the Baseten dashboard:

1. Navigate to your Baseten project settings
2. Go to the "Secrets" section
3. Add two secrets:
   - `dd_api_key`: Your Datadog API key (found in Datadog under Organization Settings → API Keys)
   - `hf_access_token`: Your Hugging Face access token

## Step 4: Deploy to Baseten

Deploy your model using Truss:

```bash
# Install or upgrade Truss with required dependencies
pip3 install --upgrade truss 'pydantic>=2.0.0'

# Deploy your model
truss push
```

This will package your configuration and deploy it to Baseten. Follow the prompts to select your Baseten workspace.

Alternatively, you can deploy through the Baseten web UI by creating a new deployment and uploading your `config.yaml` file.

## Step 5: Verify the Integration

### Check Deployment Logs

Monitor your deployment logs in Baseten to verify:

1. Datadog agent starts successfully
2. vLLM server starts and exposes metrics
3. No critical errors in the logs

Look for log messages like:
```
2025-11-12 11:25:45 PST | CORE | INFO | starting forwarder with 1 endpoints
2025-11-12 11:25:45 PST | CORE | INFO | domain 'https://app.us5.datadoghq.com.' has 1 keys: ********
```

### Check Datadog Dashboard

1. Log into your Datadog account
2. Navigate to Infrastructure → Host Map
3. Look for your host with the name you specified in `DD_HOSTNAME`
4. Navigate to Metrics → Explorer
5. Search for metrics starting with `vllm.*`

You should see metrics like:
- `vllm.e2e_request_latency.seconds`
- `vllm.time_to_first_token.seconds`
- `vllm.generation_tokens.count`
- `vllm.gpu_cache_usage_perc`
- `vllm.num_requests.running`

## Available vLLM Metrics

The integration collects various performance metrics including:

### Latency Metrics
- `vllm.e2e_request_latency.seconds` - End-to-end request latency
- `vllm.time_to_first_token.seconds` - Time to first token (TTFT)
- `vllm.time_per_output_token.seconds` - Time per output token

### Throughput Metrics
- `vllm.avg.generation_throughput.toks_per_s` - Average generation throughput
- `vllm.avg.prompt.throughput.toks_per_s` - Average prefill throughput
- `vllm.generation_tokens.count` - Number of generation tokens processed
- `vllm.prompt_tokens.count` - Number of prefill tokens processed

### Resource Metrics
- `vllm.gpu_cache_usage_perc` - GPU KV-cache usage percentage
- `vllm.cpu_cache_usage_perc` - CPU KV-cache usage percentage

### Queue Metrics
- `vllm.num_requests.running` - Number of requests currently running
- `vllm.num_requests.waiting` - Number of requests waiting
- `vllm.num_requests.swapped` - Number of requests swapped to CPU
- `vllm.num_preemptions.count` - Cumulative number of preemptions

## Testing Your Deployment

Create a test script to verify your deployment works:

```python
import requests

client = requests.Session()

resp = client.post(
    "https://model-YOUR_MODEL_ID.api.baseten.co/development/predict",
    headers={"Authorization": "Api-Key YOUR_API_KEY"},
    json={
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What does Tongyi Qianwen mean?'}
        ],
        'max_tokens': 512,
        'temperature': 0.6,
        'stream': False
    },
)

print(resp.json())
```

## Advanced Configuration

### Using Reasoning Models

For models with reasoning capabilities (like DeepSeek-R1 or Qwen3-30B-A3B), modify the start command:

```yaml
start_command: sh -c "export DD_API_KEY=$(cat /secrets/dd_api_key | tr -d '\n\r' | xargs) && mkdir -p /tmp/datadog-agent /var/log/datadog && /opt/datadog-agent/bin/agent/agent run 2>&1 & sleep 3 && HF_TOKEN=$(cat /secrets/hf_access_token) vllm serve Qwen/Qwen3-30B-A3B --reasoning-parser deepseek_r1 --served-model-name qwen30b --port 8000"
```

**Note:** In vLLM v0.10.0+, the `--enable-reasoning` flag was deprecated. Simply use `--reasoning-parser` which automatically enables reasoning mode.

### Customizing Datadog Tags

Add custom tags to help organize and filter your metrics:

```yaml
# In vllm_conf.yaml
instances:
  - openmetrics_endpoint: http://localhost:8000/metrics
    namespace: vllm
    metrics:
      - vllm.*
    tags:
      - env:production
      - service:truss-vllm
      - model:qwen-2.5-3b
      - team:ml-platform
      - region:us-west
```

### Adjusting Log Levels

To reduce log verbosity, adjust the Datadog log level:

```yaml
# In config.yaml environment_variables
DD_LOG_LEVEL: "error"  # Only show ERROR logs (options: debug, info, warn, error)
```

## Troubleshooting

### API Key Invalid Error

If you see errors like "API Key invalid":
1. Verify your API key is correct in Baseten secrets
2. Ensure there are no trailing spaces or newlines
3. Check that you're using the correct Datadog site (DD_SITE)

### Metrics Not Appearing in Datadog

1. Check that the Datadog agent started successfully in logs
2. Verify the vLLM `/metrics` endpoint is accessible
3. Confirm your API key has the correct permissions
4. Check that metrics match the pattern in `vllm_conf.yaml`

### Permission Errors

If you see "Permission denied" errors:
- Ensure the directories `/tmp/datadog-agent` and `/var/log/datadog` are writable
- The build commands create these directories with proper permissions

### High Memory Usage

If the Datadog agent uses too much memory:
- Disable unnecessary features via environment variables
- Reduce metric collection frequency
- Use more selective metric patterns in `vllm_conf.yaml`

## Best Practices

1. **Monitor Agent Health**: Set up Datadog monitors to alert on agent issues
2. **Tag Consistently**: Use consistent tagging across all deployments
3. **Start Simple**: Begin with default metrics, then customize as needed
4. **Use Secrets Management**: Never hardcode API keys in configuration files

## Cost Considerations

- Datadog charges based on custom metrics and hosts
- vLLM exposes many metrics by default
- Consider filtering metrics to only those you need
- Use metric wildcards carefully to avoid unexpected costs

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Datadog vLLM Integration](https://docs.datadoghq.com/integrations/vllm/)
- [Baseten Documentation](https://docs.baseten.co/)
- [Datadog Agent Configuration](https://docs.datadoghq.com/agent/configuration/)

## Summary

You now have a fully monitored vLLM deployment with Datadog integration! The metrics will help you:
- Track model performance over time
- Identify bottlenecks and optimization opportunities
- Set up alerts for performance degradation
- Make data-driven decisions about scaling

Happy monitoring!
