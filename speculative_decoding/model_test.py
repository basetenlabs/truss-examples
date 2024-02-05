import asyncio
import os
import time

import yaml
from model import helpers, model

if __name__ == "__main__":
    wdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(wdir, "config.yaml"), "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    async def main():
        model_instance = model.Model(config=config, secrets={})
        model_instance.load()

        request = helpers.GenerationRequest(
            # prompt="Once upon a time there was",
            prompt="Once upon",
            max_num_generated_tokens=60,
            request_id="123",
        )

        result = await model_instance.predict(request.dict())
        print(result.get_current_text())
        return

    asyncio.run(main())

    time.sleep(2)
    print("Done")
