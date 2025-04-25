import batched
import torch
import asyncio

sem = asyncio.Semaphore(100)


def example_snac():
    from snac import SNAC

    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    audio = torch.randn(
        1, 1, 32000
    ).cuda()  # placeholder for actual audio with shape (B, 1, T)

    with torch.inference_mode():
        codes = model.encode(audio)
        audio_cecode = model.quantizer.from_codes(codes)
        audio_hat = model.decoder(audio_cecode)
        print(audio_hat)
        autio_stack = torch.cat([audio_cecode, audio_cecode])
        audio_hat2 = model.decoder(autio_stack)
        print("audio_hat2,", audio_hat2)
        model.decoder = model.decoder.to(torch.bfloat16)
        audio_hat3 = model.decoder(audio_cecode.to(torch.bfloat16))
        print(audio_hat3)


@batched.dynamically(batch_size=64)
def hello_audio(audio: list[str]):
    """
    Dummy function to demonstrate the use of batched decorator.
    This function will be called with a batch size of 64.
    """
    print(len(audio), audio[0])
    return audio


async def run_audio(i):
    """
    Dummy function to demonstrate the use of async function.
    This function will be called asynchronously.
    """
    async with sem:
        await hello_audio.acall(f"hello world {i}")


async def run():
    """
    Dummy function to demonstrate the use of run function.
    This function will be called synchronously.
    """
    return await asyncio.gather(*[run_audio(i) for i in range(200)])


if __name__ == "__main__":
    asyncio.run(run())
    example_snac()
