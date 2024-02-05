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

    async def main(streaming: bool):
        model_instance = model.Model(data_dir="", config=config, secrets={})
        model_instance.load()

        request = helpers.GenerationRequest(
            prompt="Once upon",
            max_num_generated_tokens=60,
            streaming=streaming,
        )

        if streaming:
            async for part in await model_instance.predict(request.dict()):
                print(part)
        else:
            print(await model_instance.predict(request.dict()))

        model_instance.shutdown()
        return

    asyncio.run(main(streaming=False))
    asyncio.run(main(streaming=True))

    time.sleep(2)
    print("Done")
