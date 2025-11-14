import re, time
from config import SILENCE_END_MS, PUNCT_TRIGGER


class SentenceAssembler:
    def __init__(self):
        self.buf = []  # list of {text,start,end,temp_speaker}
        self.last_voice_ms = int(time.time() * 1000)

    def update_voice_time(self):
        self.last_voice_ms = int(time.time() * 1000)

    def add_words(self, words):
        """
        words: list of {text, start, end, speaker}
        Returns: list of finalized sentence dicts.
        """
        finals = []
        for w in words:
            self.buf.append(w)
            if PUNCT_TRIGGER and re.search(r"[\.!\?]$", w["text"]):
                finals.append(self._flush())
        return finals

    def maybe_flush_on_silence(self):
        now = int(time.time() * 1000)
        if self.buf and now - self.last_voice_ms > SILENCE_END_MS:
            return self._flush()
        return None

    def _flush(self):
        sent = {
            "text": " ".join(x["text"] for x in self.buf).strip(),
            "start": self.buf[0]["start"],
            "end": self.buf[-1]["end"],
            "temp_speaker": self._majority_speaker(),
        }
        self.buf.clear()
        return sent

    def _majority_speaker(self):
        if not self.buf:
            return "A"
        cnt = {}
        for x in self.buf:
            sp = x.get("speaker", "A")
            cnt[sp] = cnt.get(sp, 0) + 1
        return max(cnt, key=cnt.get)
