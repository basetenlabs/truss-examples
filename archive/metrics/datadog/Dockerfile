FROM vllm/vllm-openai:v0.11.0

# Set environment variable to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get -y install apt-transport-https curl gnupg

# Download and install the Datadog GPG key
RUN sh -c "curl -fsSL https://keys.datadoghq.com/DATADOG_APT_KEY_CURRENT.public | gpg --dearmor -o /usr/share/keyrings/datadog-archive-keyring.gpg"

# Add the Datadog repository
RUN sh -c "echo 'deb [signed-by=/usr/share/keyrings/datadog-archive-keyring.gpg] https://apt.datadoghq.com/ stable 7' > /etc/apt/sources.list.d/datadog.list"

RUN apt-get update
RUN apt-get install -y datadog-agent

# Create the vllm.d directory and copy integration config
RUN mkdir -p /etc/datadog-agent/conf.d/vllm.d
COPY vllm_conf.yaml /etc/datadog-agent/conf.d/vllm.d/conf.yaml

# Completely disable kubelet check
RUN rm -rf /etc/datadog-agent/conf.d/kubelet.d

# Create writable directories for Datadog agent
RUN mkdir -p /tmp/datadog-agent /var/log/datadog && \
    chmod -R 777 /tmp/datadog-agent /var/log/datadog
