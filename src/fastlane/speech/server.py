"""Browser voice demo (Phase 7.4): mic -> Silero VAD -> STT -> FastLane agent -> sentence-streamed TTS.

    fastlane voice                      # then open http://127.0.0.1:8765
    fastlane voice --system B1 --stt faster-whisper:small.en --tts kokoro:hf_alpha

One WebSocket per call. The page sends 16 kHz 16-bit PCM from the mic (binary frames) or typed text
(the fallback); the server sends JSON events and the reply audio (16 kHz PCM, one binary frame per sentence).
Half duplex: the page stops sending mic audio while the agent speaks (no echo, no barge-in).

Each turn is logged to data/runs/voice/<call>.jsonl with the stage times:
  vad_wait_ms   silence the detector waits before it ends the turn (fixed, = --silence-ms)
  stt_ms        final transcription
  agent_ms      FastLane turn (models + tools; trace has the split)
  tts_first_ms  synthesis of the first sentence
  voice_ms      end of turn detected -> first reply audio sent (what the caller waits, plus vad_wait_ms)
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # module level: FastAPI resolves the annotations below
from fastapi.responses import HTMLResponse

from fastlane.config import DATA_DIR
from fastlane.speech.audio import RATE, float_to_pcm16, pcm16_to_float
from fastlane.speech.stt import STT, domain_prompt
from fastlane.speech.tts import TTS, sentences
from fastlane.speech.vad import TurnDetector

PAGE = Path(__file__).with_name("page.html")
VOICE_RUNS = DATA_DIR / "runs" / "voice"


@dataclass
class VoiceTurn:
    turn: int
    mode: str  # "voice" | "text"
    caller: str
    agent: str = ""
    partials: list[str] = field(default_factory=list)
    audio_s: float = 0.0
    vad_wait_ms: float = 0.0
    stt_ms: float = 0.0
    agent_ms: float = 0.0
    model_ms: float = 0.0
    handled_by: str | None = None
    escalation_reason: str | None = None
    tts_first_ms: float = 0.0
    tts_total_ms: float = 0.0
    voice_ms: float = 0.0


class VoiceServer:
    def __init__(self, system: str = "fastlane", phone: str = "9876543210", stt: STT | None = None, tts: TTS | None = None,
                 silence_ms: int = 600, use_prompt: bool = True):
        self.system, self.phone = system, phone
        self.stt, self.tts = stt, tts
        self.silence_ms = silence_ms
        self.prompt = domain_prompt() if use_prompt else None
        self.stt_lock = threading.Lock()  # one transcription at a time (partials and finals share the model)
        self.tts_lock = threading.Lock()

    # -------------------------------------------------------------- blocking helpers (run in threads)
    def _transcribe(self, audio: np.ndarray):
        with self.stt_lock:
            return self.stt.transcribe(audio, prompt=self.prompt)

    def _speak(self, text: str):
        with self.tts_lock:
            return self.tts.speak(text)

    # -------------------------------------------------------------- app
    def app(self) -> FastAPI:
        app = FastAPI(title="FastLane voice demo")

        @app.get("/")
        def index():
            return HTMLResponse(PAGE.read_text())

        @app.websocket("/ws")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            call = Call(self, websocket)
            try:
                await call.run()
            except WebSocketDisconnect:
                pass
            finally:
                call.close()

        return app


class Call:
    """One phone call over one WebSocket."""

    def __init__(self, server: VoiceServer, websocket):
        from fastlane.cli import build_agent

        self.s, self.ws = server, websocket
        # The agent's practice world is a SQLite connection, usable only from the thread that opened it:
        # every call gets one worker thread that builds the agent and runs all of its turns.
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="call")
        self.agent = self.pool.submit(build_agent, server.system, server.phone).result()
        self.det = TurnDetector(silence_ms=server.silence_ms)
        self.speaking = False
        self.mic_s = self.mic_reported = self.mic_peak = 0.0
        self.turns: list[VoiceTurn] = []
        self.partials: list[str] = []
        self.partial_task: asyncio.Task | None = None
        VOICE_RUNS.mkdir(parents=True, exist_ok=True)
        self.log = VOICE_RUNS / f"{datetime.now():%Y%m%d-%H%M%S}-{server.system}.jsonl"

    async def on_agent_thread(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.pool, fn, *args)

    async def send(self, kind: str, **data) -> None:
        await self.ws.send_text(json.dumps({"type": kind, **data}, ensure_ascii=False))

    async def run(self) -> None:
        a = self.agent
        await self.send("ready", system=self.s.system, small=getattr(a.small, "name", None), large=getattr(a.large, "name", None),
                        stt=self.s.stt.name, tts=self.s.tts.name, phone=self.s.phone)
        await self.say(await self.on_agent_thread(a.greet), None)
        while True:
            msg = await self.ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                if not self.speaking and not a.ended:
                    await self.on_audio(pcm16_to_float(msg["bytes"]))
                continue
            data = json.loads(msg.get("text") or "{}")
            if data.get("type") == "playback_done":
                self.speaking = False
                self.det.reset()
            elif data.get("type") == "text" and data.get("text", "").strip() and not a.ended:
                await self.turn(data["text"].strip(), VoiceTurn(len(self.turns) + 1, "text", data["text"].strip()))

    async def on_audio(self, chunk: np.ndarray) -> None:
        self.mic_s += len(chunk) / RATE
        self.mic_peak = max(self.mic_peak, float(np.abs(chunk).max(initial=0.0)))
        if self.mic_s - self.mic_reported >= 2.0:  # tell the page what arrives, so a silent mic is easy to spot
            self.mic_reported = self.mic_s
            await self.send("mic", seconds=round(self.mic_s, 1), peak=round(self.mic_peak, 3))
            self.mic_peak = 0.0
        for ev in self.det.feed(chunk):
            if ev.kind == "start":
                self.partials = []
                await self.send("listening")
            elif ev.kind == "partial" and (self.partial_task is None or self.partial_task.done()):
                self.partial_task = asyncio.create_task(self.partial(ev.audio))  # skipped while one is still running
            elif ev.kind == "end":
                await self.final(ev.audio)

    async def partial(self, audio: np.ndarray) -> None:
        t = await asyncio.to_thread(self.s._transcribe, audio)
        if t.text and not self.speaking:
            self.partials.append(t.text)
            self.agent.early.on_partial(t.text)  # policy questions start their lookup while the caller is still talking
            await self.send("partial", text=t.text)

    async def final(self, audio: np.ndarray) -> None:
        if self.partial_task is not None and not self.partial_task.done():
            self.partial_task.cancel()
        self.speaking = True  # stop taking mic audio from here until the reply has been played
        t0 = time.perf_counter()
        t = await asyncio.to_thread(self.s._transcribe, audio)
        if not t.text.strip():
            self.speaking = False
            self.det.reset()
            return
        vt = VoiceTurn(len(self.turns) + 1, "voice", t.text, partials=list(self.partials), audio_s=round(len(audio) / RATE, 2),
                       vad_wait_ms=self.s.silence_ms, stt_ms=round(t.ms, 1))
        await self.send("final", text=t.text, stt_ms=round(t.ms))
        await self.turn(t.text, vt, t0)

    async def turn(self, text: str, vt: VoiceTurn, t0: float | None = None) -> None:
        t0 = t0 or time.perf_counter()
        self.speaking = True
        ta = time.perf_counter()
        reply = await self.on_agent_thread(self.agent.respond, text)  # partials were already fed to early lookup live
        tr = self.agent.traces[-1]
        vt.agent, vt.agent_ms, vt.model_ms = reply, round((time.perf_counter() - ta) * 1000, 1), tr.model_ms
        vt.handled_by, vt.escalation_reason = tr.handled_by, tr.escalation_reason
        await self.send("agent", text=reply, handled_by=tr.handled_by, reason=tr.escalation_reason, agent_ms=round(vt.agent_ms))
        await self.say(reply, vt, t0)
        self.turns.append(vt)
        with self.log.open("a") as f:
            f.write(json.dumps(asdict(vt), ensure_ascii=False) + "\n")
        await self.send("timings", **{k: v for k, v in asdict(vt).items() if k.endswith("_ms") or k in ("turn", "handled_by")})
        if self.agent.ended:
            await self.send("ended")

    async def say(self, text: str, vt: VoiceTurn | None, t0: float | None = None) -> None:
        """Speak a reply sentence by sentence; the page plays each chunk as it arrives."""
        self.speaking = True
        total = 0.0
        for i, sentence in enumerate(sentences(text)):
            sp = await asyncio.to_thread(self.s._speak, sentence)
            total += sp.ms
            await self.ws.send_bytes(float_to_pcm16(sp.audio))
            if i == 0 and vt is not None:
                vt.tts_first_ms = round(sp.ms, 1)
                vt.voice_ms = round((time.perf_counter() - t0) * 1000, 1) if t0 else 0.0
        if vt is not None:
            vt.tts_total_ms = round(total, 1)
        await self.send("spoken")  # the page answers "playback_done" when the audio has finished playing

    def close(self) -> None:
        self.pool.submit(self.agent.close).result()
        self.pool.shutdown()


def serve(system: str = "fastlane", stt: str = "mlx", tts: str = "kokoro", phone: str = "9876543210",
          host: str = "127.0.0.1", port: int = 8765, silence_ms: int = 600, use_prompt: bool = True) -> None:
    import uvicorn

    from fastlane.speech.stt import get_stt
    from fastlane.speech.tts import get_tts

    print(f"loading STT {stt} and TTS {tts} ...", flush=True)
    s, t = get_stt(stt), get_tts(tts)
    s.transcribe(np.zeros(RATE // 2, dtype=np.float32))  # warm up (first call compiles kernels)
    t.speak("Hello.")
    server = VoiceServer(system, phone, s, t, silence_ms, use_prompt)
    print(f"FastLane voice demo ({system}) on http://{host}:{port}", flush=True)
    uvicorn.run(server.app(), host=host, port=port, log_level="warning")
