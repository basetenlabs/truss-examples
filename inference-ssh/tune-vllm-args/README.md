# Tune vLLM args over SSH

This example demonstrates an in-container development loop over SSH. It deploys a small
vLLM server with SSH access enabled, then shows how to SSH into a running replica, edit
files in place, and restart the app to pick up the changes without redeploying or
recreating the container.

The server is launched by [`entr`](https://eradman.com/entrproject/), which watches
the files in `/app/data` and relaunches vLLM whenever one of them changes. Every
vLLM flag lives in [`data/vllm_args.txt`](data/vllm_args.txt), and the assistant
persona lives in [`data/chat_template.jinja`](data/chat_template.jinja). See
[`config.yaml`](config.yaml) for how it is wired together. SSH into the replica, edit
either file, and `entr` restarts the vLLM engine in place.

It serves [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
on a single L4 GPU.

## 1. Set up SSH

Set up SSH once per machine, using the
[Truss CLI](https://docs.baseten.co/reference/cli/truss/overview) or the
[Baseten CLI](https://docs.baseten.co/reference/cli/baseten/overview). Install
whichever you prefer by following its docs, then run its setup below. For background
on SSH access, see the [SSH docs](https://docs.baseten.co/inference/ssh).

**Using the Truss CLI**

```sh
truss login
truss ssh setup
```

When `truss login` prompts for an authentication method, choose the API key option.
SSH signing needs the API key stored locally, so browser login alone will not work
for `truss ssh`.

**Using the Baseten CLI**

```sh
baseten auth login
baseten ssh setup
```

> **Note:** Baseten CLI SSH is not released yet, so the Baseten CLI must be
> [built from source](https://github.com/basetenlabs/baseten-cli#building) and put on
> PATH to use it.

## 2. Deploy

Push with the Truss CLI or the Baseten CLI. Both accept `--wait` to block until the
deployment is `ACTIVE`.

**Using the Truss CLI**

```sh
truss push inference-ssh/tune-vllm-args --wait
```

**Using the Baseten CLI**

```sh
baseten model push --dir inference-ssh/tune-vllm-args --wait
```

Note the **model ID** and **deployment ID** from the output. You will use them to
call the model and to SSH in.

The first deploy pulls the vLLM base image and does a full vLLM startup, which takes
several minutes. Restarting vLLM later over SSH is much faster.

## 3. Keep a replica running

In the Baseten UI, open the deployment and set both **min and max replicas to 1**.
This keeps exactly one replica up while you tweak. An active SSH session does not by
itself prevent scale-down, so without a pinned replica your session would be
terminated when the deployment scales to zero.

## 4. Call the model

Set the model ID from the push output as an environment variable.

```sh
export MY_MODEL_ID=<model_id>
```

**Using curl**

Set your Baseten API key too.

```sh
export MY_API_KEY=<your-api-key>
```

Then run the request.

```sh
curl -s https://model-$MY_MODEL_ID.api.baseten.co/environments/production/sync/v1/chat/completions \
  -H "Authorization: Bearer $MY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What is the meaning of life?"}]
  }' | jq -r '.choices[0].message.content'
```

**Using the Baseten CLI**

```sh
baseten model predict --model-id $MY_MODEL_ID --data '{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "messages": [{"role": "user", "content": "What is the meaning of life?"}]
}' --jq '.choices[0].message.content'
```

The example ships with a pirate persona in
[`data/chat_template.jinja`](data/chat_template.jinja), so the reply comes back in
pirate voice:

> Oh matey! The answer to that's an arrrrly question! Life ain't no game for us, matey!
> It's a quest, matey! To find the meaning, matey, we gotta dive deep into the sea and
> seek the depths of the ocean!

## 5. Tune over SSH and call again

**Connect to the replica**

Set the model ID and deployment ID from the push output as environment variables.

```sh
export MY_MODEL_ID=<model_id>
export MY_DEPLOYMENT_ID=<deployment_id>
```

Then connect over SSH.

```sh
ssh model-$MY_MODEL_ID-$MY_DEPLOYMENT_ID.ssh.baseten.co
```

You can also connect from your editor. With the
[VS Code Remote - SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
or Cursor's SSH remote, connect to the same
`model-<model_id>-<deployment_id>.ssh.baseten.co` host and edit the files directly.

**Change the persona**

Edit the persona line near the top of
[`data/chat_template.jinja`](data/chat_template.jinja) and save.

```sh
vi /app/data/chat_template.jinja
```

`nano` is also installed if you prefer it.

For example, swap the pirate for Shakespeare:

```diff
-{%- set persona = "You are a swashbuckling pirate. Answer every question in boisterous pirate slang, packed with 'arrr', 'matey', and nautical metaphors." -%}
+{%- set persona = "You are William Shakespeare. Answer every question in florid Elizabethan English, in iambic pentameter where the meter allows." -%}
```

`entr` picks up the change and restarts vLLM in place. Watch the deployment logs in the
Baseten dashboard to see it restart, and wait for the restart to finish. Once it does,
the assistant adopts the new persona.

The reload reinitializes the engine and reloads the Python stack, so it takes a bit, but
it is much faster than the first deploy because the image, weights, and JIT caches are
already warm.

> **Note:** The same works for any vLLM flag. Edit
> [`data/vllm_args.txt`](data/vllm_args.txt) to change `--max-model-len` or add
> another argument, and saving restarts the engine the same way.

**Call the model again**

Re-run the call from step 4 and compare the response. With the Shakespeare persona, the
same prompt comes back as:

> The meaning of life, my lord, is to know and love; it's a quest for wisdom, for truth,
> and for joy; it's not just about surviving but to live fully, to be happy, and to
> serve our Maker with all we have.
