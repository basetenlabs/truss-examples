# model/model.py
import copy
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse, StreamingResponse
import asyncio

Message = Dict[str, str]  # {"role": "...", "content": "..."}


class Model:
    # model class implementing fanout via suffix_messages
    def __init__(self, trt_llm, **kwargs) -> None:
        self._secrets = kwargs["secrets"]
        self._engine = trt_llm["engine"]

    async def predict(self, model_input: Dict[str, Any], request: Request) -> Any:
        if not isinstance(model_input, dict):
            raise HTTPException(
                status_code=400, detail="Request body must be a JSON object."
            )

        # Enforce non-streaming
        if bool(model_input.get("stream", False)):
            raise HTTPException(
                status_code=400,
                detail="stream=true is not supported here; set stream=false.",
            )

        n = model_input.pop("n", 1)
        if n != 1:
            raise HTTPException(
                status_code=400,
                detail="n>1 is not supported here; use suffix_messages for multi-generation fanout.",
            )

        prompt_key, base_messages = self._get_base_messages(model_input)
        n, suffix_tasks = self._parse_fanout(model_input)

        # Build a reusable request skeleton (don’t forward n/suffix_messages to engine)
        base_req = copy.deepcopy(model_input)
        base_req.pop("suffix_messages", None)

        # Run sequential generations
        per_gen_payloads: List[Any] = []

        async def run_generation(i: int) -> Any:
            msgs_i = copy.deepcopy(base_messages)
            if suffix_tasks is not None:
                msgs_i.extend(suffix_tasks[i])
            base_req[prompt_key] = msgs_i
            print(f"Running generation {i} with messages: {msgs_i}")

            resp = await self._engine.chat_completions(
                request=request, model_input=base_req
            )

            # You said stream=false returns a full Response object.
            # Still: fail loudly if a stream ever appears.
            if isinstance(resp, StreamingResponse) or hasattr(resp, "body_iterator"):
                raise HTTPException(
                    status_code=400,
                    detail="Engine returned streaming but stream=false was requested.",
                )

            return resp

        # (michaelfeil): get first response, then asyncio.gather the rest.
        payload = await run_generation(0)
        per_gen_payloads.append(payload)
        if n > 1:
            # run n-1 other tasks concurrently
            results = await asyncio.gather(*(run_generation(i) for i in range(1, n)))
            per_gen_payloads.extend(results)

        # Convert to OpenAI-ish multi-choice response
        out = self._to_openai_choices(per_gen_payloads)
        return JSONResponse(content=out.model_dump())

    # ---------------- helpers ----------------

    def _get_base_messages(
        self, model_input: Dict[str, Any]
    ) -> Tuple[str, List[Message]]:
        if "prompt" in model_input:
            raise HTTPException(
                status_code=400,
                detail='Use "messages" instead of "prompt" for chat models.',
            )
        if "messages" not in model_input:
            raise HTTPException(
                status_code=400, detail='Request must include "messages" field.'
            )
        key = "messages"
        msgs = model_input.get(key)
        if not isinstance(msgs, list):
            raise HTTPException(
                status_code=400, detail=f'"{key}" must be a list of messages.'
            )

        for m in msgs:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise HTTPException(
                    status_code=400,
                    detail=f'Each item in "{key}" must have role+content.',
                )
        return key, msgs  # type: ignore[return-value]

    def _parse_fanout(
        self, model_input: Dict[str, Any]
    ) -> Tuple[int, Optional[List[List[Message]]]]:
        suffix = model_input.get("suffix_messages", None)

        if not isinstance(suffix, list) or any(not isinstance(t, list) for t in suffix):
            raise HTTPException(
                status_code=400,
                detail='"suffix_messages" must be a list of tasks (each task is a list of messages).',
            )
        if len(suffix) < 1 or len(suffix) > 256:
            raise HTTPException(
                status_code=400,
                detail='"suffix_messages" must have between 1 and 256 tasks.',
            )

        for task in suffix:
            for m in task:
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    raise HTTPException(
                        status_code=400,
                        detail="Each suffix message must have role+content.",
                    )

        k = len(suffix)

        return k, suffix  # type: ignore[return-value]

    def _to_openai_choices(
        self, payloads: List[Any]
    ) -> Any:
        """
        payloads: list of openai.types.chat.chat_completion.ChatCompletion objects
        """
        base = payloads[0]

        # If it's an OpenAI ChatCompletion object, patch it in-place.
        if hasattr(base, "choices") and hasattr(base, "model_dump"):
            print("Patching OpenAI ChatCompletion response for fanout...")
            new_choices = []
            for i, p in enumerate(payloads):
                c0 = p.choices[0]
                # Ensure index matches OpenAI n semantics
                try:
                    c0.index = i
                except Exception:
                    # some objects may be frozen; fallback to dumping/editing dict
                    c0 = c0.model_copy(update={"index": i})
                new_choices.append(c0)

            base.choices = new_choices
            return base  # return object; caller should serialize it

        raise HTTPException(
            status_code=500, detail=f"Unsupported engine response type for fanout. {type(base)}"
        )
