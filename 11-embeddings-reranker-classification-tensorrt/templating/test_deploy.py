def test_deploy(deploy_id: str = "03ykpnkw"):
    import os

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["BASETEN_API_KEY"],
        base_url=f"https://model-{deploy_id}.api.baseten.co/environments/production/sync/v1",
    )

    # Default completion
    response_completion = client.completions.create(
        model="not_required",
        prompt="Q: Tell me everything about Baseten.co! A:",
        temperature=0.3,
        max_tokens=100,
    )
    assert "baseten" in response_completion.choices[0].text.lower()

    # Chat completion
    response_chat = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": "Tell me everything about Baseten.co!"}],
        temperature=0.3,
        max_tokens=100,
    )
    assert "baseten" in response_chat.choices[0].message.content.lower()
    # Structured output
    from pydantic import BaseModel

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    completion = client.beta.chat.completions.parse(
        model="not_required",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        response_format=CalendarEvent,
    )

    event = completion.choices[0].message.parsed
    assert "science" in event.name.lower()
    print(f"✅ All tests passed for deployment {deploy_id}")


if __name__ == "__main__":
    test_deploy()
