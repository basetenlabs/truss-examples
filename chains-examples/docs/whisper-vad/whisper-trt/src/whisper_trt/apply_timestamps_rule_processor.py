from typing import Optional

import torch
import torch.nn.functional as F

EOT_PROB_THRESHOLD = 0.95
END_OF_CHUNK_MARGIN_SECS = 1.0
BLANK_LOGIT = -float("inf")


class ApplyTimestampsRule:
    def __init__(
        self,
        num_beams: int,
        batch_size: int,
        duration_secs: list[float],
        notimestamps_id: int,
        timestamp_begin_id: int,
        eot_id: int,
        input_prompt_ids_offset: int,
        # TODO: propogate device in properly
        device=torch.device("cuda"),
    ) -> None:
        self._num_beams = num_beams
        self._batch_size = batch_size
        self._duration_secs_by_batch = torch.tensor(
            duration_secs, dtype=torch.float, device=device
        )
        self._notimestamps_id = notimestamps_id
        self._eot_id = eot_id
        self._timestamp_begin_id = timestamp_begin_id
        self._input_prompt_ids_offset = input_prompt_ids_offset

        # Initialize tensors
        self._last_ts_by_batch = torch.full(
            (batch_size,), float("-inf"), dtype=torch.float, device=device
        )
        min_int = torch.iinfo(torch.int).min
        self._last_ts_token_id_by_batch = torch.full(
            (batch_size,), min_int, dtype=torch.int, device=device
        )
        self._last_token_id_by_batch = torch.full(
            (batch_size,), min_int, dtype=torch.int, device=device
        )
        self._penultimate_token_id_by_batch = torch.full(
            (batch_size,), min_int, dtype=torch.int, device=device
        )

    def __call__(self, step, input_ids, scores):
        _track_info(
            step=step,
            input_ids=input_ids,
            timestamp_begin_id=self._timestamp_begin_id,
            batch_size=self._batch_size,
            num_beams=self._num_beams,
            prompt_ids_offset=self._input_prompt_ids_offset,
            device=scores.device,
            last_ts_token_id_by_batch=self._last_ts_token_id_by_batch,
            last_token_id_by_batch=self._last_token_id_by_batch,
            penultimate_token_id_by_batch=self._penultimate_token_id_by_batch,
            last_ts_by_batch=self._last_ts_by_batch,
        )

        self._try_avoid_early_termination_via_prob_threshold_batch(
            scores=scores,
            last_ts=self._last_ts_by_batch,
            duration=self._duration_secs_by_batch,
        )

        self._process_batch(
            step=step,
            scores=scores,
            timestamp_begin_id=self._timestamp_begin_id,
            notimestamps_id=self._notimestamps_id,
        )

        return scores

    def _process_batch(
        self,
        step: int,
        scores: torch.Tensor,
        timestamp_begin_id: int,
        notimestamps_id: int,
    ):
        scores[:, notimestamps_id] = BLANK_LOGIT

        if step == 0:
            scores[:, : self._notimestamps_id] = BLANK_LOGIT

        if step <= 1:
            return

        ts_log_prob_sum = torch.logsumexp(scores[:, timestamp_begin_id:], dim=1)
        max_non_ts_prob, _ = scores[:, :notimestamps_id].max(dim=1)
        mask = ts_log_prob_sum > max_non_ts_prob
        scores[mask, :timestamp_begin_id] = BLANK_LOGIT

    def _process_batch_beam(
        self,
        step: int,
        scores: torch.Tensor,
        last_token_id: Optional[int],
        penultimate_token_id: Optional[int],
        last_ts_token_id: Optional[int],
        timestamp_begin_id: int,
        notimestamps_id: int,
    ):
        scores[notimestamps_id] = BLANK_LOGIT

        # suppress non-timestamps at the beginning
        if step == 0:
            scores[:notimestamps_id] = BLANK_LOGIT

        def force_ts_based_on_prob():
            # If sum of ts log prob > max non-ts log prob then force ts
            ts_log_prob_sum = torch.logsumexp(scores[timestamp_begin_id:], dim=0)
            max_non_ts_prob = scores[:notimestamps_id].max()
            if ts_log_prob_sum > max_non_ts_prob:
                scores[:timestamp_begin_id] = BLANK_LOGIT

        if last_token_id >= 0 and penultimate_token_id >= 0:
            if last_token_id >= timestamp_begin_id:
                if penultimate_token_id >= timestamp_begin_id:
                    # ts--ts, should be non-ts
                    # disallowing eot for batch and bm because of ts pair
                    scores[timestamp_begin_id:] = BLANK_LOGIT
                else:
                    # non_ts-ts, can be ts or eot, can't be text
                    force_ts_based_on_prob()
                    # TODO(pankaj) This seems to increase miss_block_count very slightly,
                    # figure out if that's a real issue. If not then add this line back.
                    # scores[:eot_id] = blank
                    scores[timestamp_begin_id:last_token_id] = BLANK_LOGIT
            else:
                # non_ts prev token
                force_ts_based_on_prob()
                if last_ts_token_id is not None:
                    scores[timestamp_begin_id:last_ts_token_id] = BLANK_LOGIT

    def _try_avoid_early_termination_via_prob_threshold_batch(
        self,
        scores: torch.Tensor,
        last_ts: torch.Tensor,
        duration: torch.Tensor,
    ):
        duration_mask = duration - last_ts > END_OF_CHUNK_MARGIN_SECS

        probs = F.softmax(scores, dim=1)
        eot_prob = probs[:, self._eot_id]
        eot_prob_mask = eot_prob < EOT_PROB_THRESHOLD

        mask = duration_mask.repeat_interleave(self._num_beams) & eot_prob_mask
        scores[mask, self._eot_id] = BLANK_LOGIT


def _track_info(
    step,
    input_ids,
    timestamp_begin_id,
    batch_size,
    num_beams,
    prompt_ids_offset,
    device,
    last_ts_token_id_by_batch,
    last_token_id_by_batch,
    penultimate_token_id_by_batch,
    last_ts_by_batch,
):
    if step > 0:
        # Select relevant batches and steps
        batch_indices = torch.arange(
            0, batch_size * num_beams, num_beams, device=device
        )
        relevant_input_ids = input_ids[batch_indices]

        # Get last token ids
        last_token_ids = relevant_input_ids[:, step + prompt_ids_offset - 1]
        last_token_id_by_batch[:] = last_token_ids

        if step > 1:
            penultimate_token_ids = relevant_input_ids[:, step + prompt_ids_offset - 2]
            penultimate_token_id_by_batch[:] = penultimate_token_ids

        # Find last timestamp token ids
        ts_mask = last_token_ids >= timestamp_begin_id
        if ts_mask.any():
            last_ts_token_id_by_batch[ts_mask] = last_token_ids[ts_mask]
            last_ts_by_batch[ts_mask] = 0.02 * (
                last_token_ids[ts_mask].float() - timestamp_begin_id
            )
