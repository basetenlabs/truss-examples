"""Script for local testing of model.py."""

import asyncio
import os
import shutil
import sys

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

    async def predict(model_instance: model.Model, streaming: bool, use_draft: bool):
        request = helpers.GenerationRequest(
            prompt="Once upon",
            max_num_generated_tokens=60,
            streaming=streaming,
        )
        if not use_draft:
            request.num_draft_tokens = 0

        result = await model_instance.predict(request.dict())

        if isinstance(result, str):
            print("Non-Streaming results:")
            print(result)
        else:
            print("Streaming results:")
            async for part in result:
                print(part, end="")
            print("\n")

        return

    async def main():
        print("Loading model.")
        model_instance = model.Model(data_dir="", config=config, secrets={})
        model_instance.load()
        print("Model loaded.")

        await predict(model_instance, streaming=False, use_draft=True)
        await predict(model_instance, streaming=True, use_draft=True)
        await predict(model_instance, streaming=True, use_draft=False)

        print("Shutting down.")
        model_instance.shutdown()
        print("Done.")

    asyncio.run(main())
