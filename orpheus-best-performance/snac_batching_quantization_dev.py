import batched
import torch
import asyncio

sem = asyncio.Semaphore(100)


def example_2():
    pass


from snac import SNAC

# force inference mode during the lifetime of the script
inference_mode_raii_guard = torch._C._InferenceMode(True)

snac_device = "cuda"


class SnacModelBatched:
    def __init__(self):
        self.dtype_decoder = torch.float32
        snac_torch_compile = False

        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        model = model.to(snac_device)

        model.decoder = model.decoder.to(self.dtype_decoder)
        if snac_torch_compile:
            model.decoder = torch.compile(model.decoder, dynamic=True)
            model.quantizer = torch.compile(model.quantizer, dynamic=True)
        for bs_size in [1, 256]:
            audio = torch.randn(bs_size, 1, 4096).to(snac_device)
            with torch.inference_mode():
                codes = model.encode(audio)
                intermed = model.quantizer.from_codes(codes)
                model.decoder(intermed.to(self.dtype_decoder))
        # model.encoder = torch.nn.Identity()
        self.snac_model = model
        self.stream = torch.Stream()

    @batched.dynamically(batch_size=256, timeout_ms=10)
    def batch_snac_model(
        self, items: list[dict[str, list[torch.Tensor]]]
    ) -> list[torch.Tensor]:
        # Custom processing logic here
        # return [model.decode(item["codes"]) for item in items]
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            stacked_z_q = torch.cat(  # codes is list[torch.Tensor]
                [
                    self.snac_model.quantizer.from_codes(codes["codes"])
                    for codes in items
                ],
                dim=0,
            )
            output_batched = self.snac_model.decoder(
                stacked_z_q.to(self.dtype_decoder)
            ).to(torch.float32)
            out = output_batched.split(
                1, dim=0
            )  # unbatch the output into len(items) tensors of shape (1, 1, x)
            self.stream.synchronize()  # make sure the results are ready
            return out


model_snac = SnacModelBatched()

dummy = torch.randn(1, 1, 4096).to(snac_device)

a = model_snac.batch_snac_model(
    {"codes": model_snac.snac_model.encode(dummy)}  # shape (1, 1, 4096)
)

print(a)


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
