from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import requests


DEFAULT_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "8413073290:AAEtREozkPdAgfrPtMA5eKUSuqZPfd7n1w4")
DEFAULT_CHAT_ID = os.getenv("TG_CHAT_ID", "1657628015")


class TelegramNotifier:
    def __init__(
        self,
        *,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.bot_token = str(bot_token or DEFAULT_BOT_TOKEN or "").strip()
        self.chat_id = str(chat_id or DEFAULT_CHAT_ID or "").strip()
        self.log = log
        self._lock = threading.Lock()
        self._last_sent_at: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, text: str, *, key: Optional[str] = None, cooldown_seconds: float = 5.0) -> None:
        if not self.enabled:
            return
        dedupe_key = str(key or text).strip()
        now = time.monotonic()
        with self._lock:
            last_sent_at = self._last_sent_at.get(dedupe_key, 0.0)
            if cooldown_seconds > 0 and (now - last_sent_at) < cooldown_seconds:
                return
            self._last_sent_at[dedupe_key] = now
        worker = threading.Thread(
            target=self._send_sync,
            args=(text,),
            name="TelegramNotifier",
            daemon=True,
        )
        worker.start()

    def _send_sync(self, text: str) -> None:
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
                timeout=(3.05, 10.0),
            )
            response.raise_for_status()
        except Exception as exc:
            if self.log is not None:
                try:
                    self.log(f"Telegram alert failed: {exc}")
                except Exception:
                    pass
