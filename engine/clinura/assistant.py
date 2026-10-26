from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

GROUND_RULES = """You work inside Clinura, a contactless vital-sign monitor that estimates heart rate, \
heart-rate variability (RMSSD/SDNN) and respiration from a camera using remote \
photoplethysmography. Every number you see is a camera-derived estimate, not a validated medical \
measurement. Rules:
- Use only the telemetry JSON you are given. Never invent values, times, medications or events.
- Quote exact clock times (HH:MM or HH:MM:SS) and values from the data when you refer to them.
- If the data cannot answer something, say so plainly and say what would be needed.
- Events marked as coinciding with movement are likely artifacts; say so when relevant.
- Eyes-closed rest periods come from eye-blink tracking and are only a proxy for sleep.
- Notes with source "manual" were entered by staff and are authoritative for what they state.
- Be concise and clinical. Do not give treatment advice or diagnoses."""

HANDOVER_PROMPT = """Write a shift handover note from the telemetry below. Use this exact structure \
with these headings, plain text, and "- " bullets where useful:

## Summary
One or two sentences: who/where (use the bed and patient labels if provided), the window covered, \
and the overall picture.

## Vitals
Heart rate (mean ± SD, range with times), HRV (RMSSD), respiration. Mention coverage when below 70%.

## Notable events
Chronological bullets with clock times, values and likely cause (e.g. movement artifact). Write \
"None detected" if there are none.

## Staff notes
Manual notes with times, or "None logged".

## Assessment
A cautious one-to-three sentence impression grounded in the data, for example whether camera-derived \
vitals stayed within typical resting ranges. State the main data-quality caveat."""


class AssistantError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class AssistantConfig:
    api_key: str | None
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> AssistantConfig:
        return cls(
            api_key=os.environ.get("DEEPSEEK_API_KEY") or None,
            model=os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL,
            base_url=(os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


class Assistant:
    def __init__(self, config: AssistantConfig | None = None) -> None:
        self.config = config or AssistantConfig.from_env()

    def status(self) -> dict[str, object]:
        return {"configured": self.config.configured, "model": self.config.model}

    async def _chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        if not self.config.configured:
            raise AssistantError(
                "DeepSeek is not configured. Add DEEPSEEK_API_KEY to the .env file and restart.",
                status_code=503,
            )
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                response = await client.post(
                    f"{self.config.base_url}/chat/completions", json=body, headers=headers
                )
        except httpx.TimeoutException as exc:
            raise AssistantError("DeepSeek did not respond in time. Try again.", 504) from exc
        except httpx.HTTPError as exc:
            raise AssistantError(f"Could not reach DeepSeek ({exc.__class__.__name__}).") from exc

        if response.status_code == 401:
            raise AssistantError("DeepSeek rejected the API key. Check DEEPSEEK_API_KEY.", 502)
        if response.status_code == 402:
            raise AssistantError("The DeepSeek account has insufficient balance.", 502)
        if response.status_code == 429:
            raise AssistantError("DeepSeek rate limit reached. Wait a moment and retry.", 429)
        if response.status_code >= 400:
            detail = response.text[:300]
            logger.warning("DeepSeek error %s: %s", response.status_code, detail)
            raise AssistantError(f"DeepSeek returned an error ({response.status_code}).")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AssistantError("DeepSeek returned an unexpected response.") from exc
        text = str(content or "").strip()
        if not text:
            raise AssistantError("DeepSeek returned an empty answer.")
        return text

    async def handover(
        self, context: dict[str, object], bed: str | None, patient: str | None
    ) -> str:
        labels = {"bed": bed or "not specified", "patient": patient or "not specified"}
        messages = [
            {"role": "system", "content": GROUND_RULES},
            {
                "role": "user",
                "content": f"{HANDOVER_PROMPT}\n\nLabels: {json.dumps(labels)}\n\n"
                f"Telemetry:\n{json.dumps(context, separators=(',', ':'))}",
            },
        ]
        return await self._chat(messages, max_tokens=1200)

    async def ask(
        self, question: str, history: list[dict[str, str]], context: dict[str, object]
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": f"{GROUND_RULES}\n\nYou answer questions from the attending team about "
                "the patient's monitored state. Lead with the direct answer, then the supporting "
                "times and values. Keep answers under 150 words unless asked for detail.\n\n"
                f"Telemetry:\n{json.dumps(context, separators=(',', ':'))}",
            }
        ]
        for turn in history[-8:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])[:4000]})
        messages.append({"role": "user", "content": question})
        return await self._chat(messages, max_tokens=700)


def local_handover(context: dict[str, object], bed: str | None, patient: str | None) -> str:
    window = context["window"]
    coverage = context["coverage"]
    hr = context.get("heart_rate_bpm")
    rmssd = context.get("rmssd_ms")
    resp = context.get("respiration_bpm")
    peak = context.get("heart_rate_peak")
    trough = context.get("heart_rate_lowest")
    events = [e for e in context.get("events", []) if e.get("source") != "manual"]
    notes = [e for e in context.get("events", []) if e.get("source") == "manual"]
    who = ", ".join(part for part in (bed and f"Bed {bed}", patient) if part) or "Monitored patient"

    lines = ["## Summary"]
    if not window["seconds_recorded"]:
        lines.append(f"{who}: no telemetry was recorded in the last {window['hours']} h.")
        return "\n".join(lines)
    lines.append(
        f"{who}, {window['hours']} h window ending {str(window['end'])[11:16]}. "
        f"Patient in view {coverage['patient_in_view_pct']}% of the time; confirmed heart rate "
        f"available {coverage['confirmed_hr_pct']}% of the time."
    )
    lines += ["", "## Vitals"]
    if hr:
        text = f"- Heart rate {hr['mean']} ± {hr['sd']} BPM"
        if peak and trough:
            text += (
                f", range {trough['bpm']:.0f} BPM ({trough['time']}) to "
                f"{peak['bpm']:.0f} BPM ({peak['time']})"
            )
        lines.append(text)
    else:
        lines.append("- Heart rate: no confirmed readings")
    lines.append(
        f"- HRV (RMSSD) mean {rmssd['mean']} ms" if rmssd else "- HRV: not enough clean data"
    )
    lines.append(
        f"- Respiration mean {resp['mean']} breaths/min" if resp else "- Respiration: not available"
    )
    lines += ["", "## Notable events"]
    lines += [
        f"- {e['clock']} {e['title']}" + (f" ({e['detail']})" if e.get("detail") else "")
        for e in events
    ] or ["- None detected"]
    lines += ["", "## Staff notes"]
    lines += [
        f"- {e['clock']} {e['title']}" + (f": {e['detail']}" if e.get("detail") else "")
        for e in notes
    ] or ["- None logged"]
    lines += [
        "",
        "## Assessment",
        "Generated locally without AI from camera-derived estimates. Review the events above; "
        "configure DEEPSEEK_API_KEY for a narrative summary.",
    ]
    return "\n".join(lines)
