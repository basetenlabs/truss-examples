import torch
import torch.nn.functional as F

BLANK_LOGIT = -float("inf")


class LanguageDetectionRules:
    def __init__(
        self,
        num_beams: int,
        batch_size: int,
        nospeech_id: int,
        lang_token_ids: list[int],
        # TODO: propogate device in properly
        no_speech_threshold: int = 0.5,
        device=torch.device("cuda"),
    ) -> None:
        self._num_beams = num_beams
        self._batch_size = batch_size
        self._no_speech_id = nospeech_id
        self._lang_token_ids = lang_token_ids
        self._no_speech_threshold = no_speech_threshold

    def __call__(self, step, input_ids, scores):
        mask = torch.ones(scores.shape[-1], dtype=torch.bool)
        mask[self._lang_token_ids + [self._no_speech_id]] = False
        # Don't allow any tokens that are not language tokens or nospeed
        scores[:, mask] = BLANK_LOGIT

        # Check for nospeechpercentage being above threshold
        probs = F.softmax(scores, dim=1)
        no_speech_prob = probs[:, self._no_speech_id]
        no_speech_prob_max = no_speech_prob < self._no_speech_threshold
        scores[no_speech_prob_max, self._no_speech_id] = BLANK_LOGIT

        return scores
