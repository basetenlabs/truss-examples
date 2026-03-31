import asyncio

import httpx

BASETEN_API_KEY = "bsKlILqX...."


async def send_post_request(url, data, task_id, headers=None):
    async with httpx.AsyncClient() as client:
        data["task_id"] = task_id
        response = await client.post(url, json=data, headers=headers)
        returned_task_id = response.json()["task_id"]
        assert returned_task_id == task_id
        return response


async def send_concurrent_requests(url, data, headers=None):
    tasks = [send_post_request(url, data, i, headers=headers) for i in range(4)]
    responses = await asyncio.gather(*tasks)
    return responses


# Example usage (Uncomment and replace the URL and data as needed)

asyncio.run(
    send_concurrent_requests(
        "https://model-2qjdzkpq.api.baseten.co/development/predict",
        {},
        headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"},
    )
)
