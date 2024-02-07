"""Script for local testing of model.py."""

import asyncio
import os
import shutil
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages"))

import yaml
from model import model
from packages import helpers

if __name__ == "__main__":
    wdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(wdir, "config.yaml"), "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    shutil.copytree(
        src=os.path.join(wdir, "packages", "triton_model_repo"),
        dst=os.path.join("/", "packages", "triton_model_repo"),
        dirs_exist_ok=True,
    )

    async def predict(model_instance: model.Model, streaming: bool):
        request = helpers.GenerationRequest(
            prompt="Once upon",
            max_num_generated_tokens=60,
            streaming=streaming,
        )

        if streaming:
            print("Streaming results:")
            async for part in await model_instance.predict(request.dict()):
                print(part, end="")
            print("\n")
        else:
            print("Non-Streaming results:")
            print(await model_instance.predict(request.dict()))

        return

    async def main():
        print("Loading model.")
        model_instance = model.Model(data_dir="", config=config, secrets={})
        model_instance.load()
        print("Model loaded.")

        await predict(model_instance, streaming=False)
        await predict(model_instance, streaming=True)

        print("Shutting down.")
        model_instance.shutdown()
        print("Done.")

    asyncio.run(main())
