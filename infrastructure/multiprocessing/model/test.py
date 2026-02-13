import asyncio
from typing import Any

import model


async def call_fn(fn, payload: Any) -> Any:
    return await fn(payload)


async def test():
    m = model.Model(data_dir="", config="", secrets="")
    body = {"n": 100}

    payload = await call_fn(m.preprocess, body)
    response = await call_fn(m.predict, payload)

    return response


if __name__ == "__main__":
    x = asyncio.run(test())
    print(x)
