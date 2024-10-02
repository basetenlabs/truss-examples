## Deploying on Baseten with `start_command`

If you've deployed on [Baseten](https://baseten.co) before, you're likely familiar with using [Truss](https://docs.baseten.co/quickstart) to do so.

This process typically involves creating a `model.py` file, which contains the code for your model, and packaging it for deployment via an API endpoint managed by Baseten.

However, there are situations where you might want to deploy a model that's already wrapped in an API. A common example is the `vLLM` OpenAI-Compatible Server, which provides its own HTTP endpoint. Another scenario is the "bring your own image" approach, where you have a Docker image that can handle HTTP requests directly.

In these cases, setting up an HTTP endpoint through Truss or writing a `model.py` file can introduce unnecessary overhead. To streamline this, we've introduced the *docker_server.start_command* option, allowing you to specify an alternative `docker start` command, avoiding the need for additional setup.
