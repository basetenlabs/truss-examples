# gRPC server dev loop over SSH

This example demonstrates an in-container development loop over SSH for a Python
gRPC server. It deploys a small gRPC server with SSH access enabled, then shows
how to SSH into a running replica, edit the server code in place, and restart it
to pick up the change without redeploying or recreating the container.

The server implements a `Greeter` service plus the standard gRPC health check
service that Baseten uses to decide the server is healthy. The code lives in
[`project/`](project), a self-contained [uv](https://docs.astral.sh/uv/) project
whose `grpc_greeter` package holds the server, client, and generated stubs. In
the container the server runs under [`entr`](https://eradman.com/entrproject/),
which relaunches it whenever a package file changes; see
[`config.yaml`](config.yaml) for the wiring.

It runs on CPU, since a greeter needs no GPU.

## 1. Set up SSH

Set up SSH once per machine with the
[Baseten CLI](https://docs.baseten.co/reference/cli/baseten/overview). For
background on SSH access, see the [SSH docs](https://docs.baseten.co/inference/ssh).

```sh
baseten auth login
baseten ssh setup
```

## 2. Deploy

Push with the Baseten CLI. Pass `--wait` to block until the deployment is
`ACTIVE`.

```sh
baseten model push --dir inference-ssh/grpc-server --wait
```

Note the **model ID** and **deployment ID** from the output. You will use them to
call the model and to SSH in.

The first deploy pulls the base image and installs dependencies, which takes a
few minutes. Restarting the server later over SSH is much faster.

## 3. Keep a replica running

In the Baseten UI, open the deployment and set both **min and max replicas to 1**.
This keeps exactly one replica up while you tweak. An active SSH session does not
by itself prevent scale-down, so without a pinned replica your session would be
terminated when the deployment scales to zero.

## 4. Call the model

The client is part of the `grpc_greeter` project. Set the model ID and API key in
the environment, then run it with uv from the project directory.

```sh
export MY_MODEL_ID=<model_id>
export MY_API_KEY=<your-api-key>

cd inference-ssh/grpc-server/project
uv run grpc-greeter-client --name World
```

The client opens a secure channel to `model-$MY_MODEL_ID.grpc.api.baseten.co:443`
and passes the API key and model ID as gRPC metadata. The reply comes back as:

> Hello, World!

## 5. Edit over SSH and call again

**Connect to the replica**

Set the deployment ID from the push output as an environment variable, then
connect over SSH.

```sh
export MY_DEPLOYMENT_ID=<deployment_id>
ssh model-$MY_MODEL_ID-$MY_DEPLOYMENT_ID.ssh.baseten.co
```

You can also connect from your editor. With the
[VS Code Remote - SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
or Cursor's SSH remote, connect to the same
`model-<model_id>-<deployment_id>.ssh.baseten.co` host and edit the files
directly.

**Change the greeting**

Edit the `GREETING` constant near the top of the server and save.

```sh
vi /packages/src/grpc_greeter/server.py
```

`nano` is also installed if you prefer it.

```diff
-GREETING = "Hello"
+GREETING = "Howdy"
```

`entr` picks up the change and restarts the server in place. Watch the deployment
logs in the Baseten dashboard to see it restart, and wait for the restart to
finish.

**Call the model again**

Re-run the call from step 4 and compare the response:

> Howdy, World!

> **Note:** The server runs the generated stubs, not
> [`project/greeter.proto`](project/greeter.proto) directly. Changing the contract
> takes effect only when you regenerate the stubs with `uv run grpc-greeter-codegen`;
> that rewrites the package code, which `entr` picks up and restarts. The client
> and server share the stubs, so you may want to regenerate and redeploy, or
> regenerate on both sides to avoid a redeploy.
