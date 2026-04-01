# VS Code Remote Tunnel - Custom Server

This `docker_server` Truss that runs a FastAPI server alongside a VS Code remote tunnel. This allows rapid in-IDE development.

On startup, the container:

1. Starts a FastAPI server with uvicorn (auto-reloads on file changes)
2. Prompts you to authenticate via Microsoft device login
3. Opens a VS Code tunnel you can connect to from your local editor or browser

Once connected, you can edit server code, see changes reload automatically, and use the integrated terminal - all running on the deployed container's hardware.

## Prerequisites

- [Truss CLI](https://docs.baseten.co/quickstart) installed (or just use `uvx truss` in place of `truss` below)
- [Remote - Tunnels](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-server) VS Code extension installed locally

## Usage

Deploy the server and watch the logs from inside this directory:

```sh
truss push --tail
```

Watch logs looking for green login prompt. Visit the URL following it and use device code to authenticate.

Once authenticated, the logs will print two connection URLs:
- **Web** - opens VS Code in your browser via `vscode.dev`
- **Desktop** - opens VS Code on your machine via `vscode://` link (requires Remote - Tunnels extension)

Click either link to connect. The tunnel opens to `/app/data` where the server code lives. Can also manually find the remote tunnel from a running vscode instance.

Now send a predict request, e.g. via the playground for the deployment in the UI:

```json
{
  "foo": "bar"
}
```

And see the response of:

```json
{
  "message": "hello from vscode-tunnel truss",
  "input": {
    "foo": "bar"
  }
}
```

Now alter the `server.py` to append `, second version` to the `message`, and notice how the logs will restart as soon as file save occurs. Now the same predict call will return:

```json
{
  "message": "hello from vscode-tunnel truss, second version",
  "input": {
    "foo": "bar"
  }
}
```

## Notes

* This is just an example. See [data/server-start.sh](data/server-start.sh) for the basic commands to install VS Code standalone CLI to incorporate into any existing container/setup.
* The Python extension is installed on tunnel create, but can install any others needed, including at runtime from inside the VS Code.
* The terminal provides full shell into the container for general purpose use.
* There is a task to completely restart uvicorn inside [data/.vscode/tasks.json](data/.vscode/tasks.json), use `Ctrl+Shift+B` to run it.
* Deployments can scale to zero after several minutes of inactivity. To keep the container alive, run a keepalive loop from your local machine: `while true; do curl -s <your-predict-url> > /dev/null; sleep 30; done`

## Files

- `config.yaml` - Truss configuration (base image, build commands, start command)
- `data/server-start.sh` - Startup script that launches uvicorn, authenticates, and starts the tunnel
- `data/server.py` - FastAPI server with `/health` and `/predict` endpoints
- `data/pyproject.toml` / `data/uv.lock` - Python dependencies managed by uv
- `data/.vscode/tasks.json` - VS Code task for restarting the server
