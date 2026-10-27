import numpy as np
import pytest

pytest.importorskip("soundfile")

from fastlane.speech.audio import RATE, float_to_pcm16, load, pcm16_to_float, resample, wav_bytes  # noqa: E402
from fastlane.speech.tts import for_speech, sentences  # noqa: E402


def test_sentences_stream_in_order_and_keep_short_bits_together():
    reply = "Just to confirm: 2 prime tickets for Rocket Raja, total ₹520. Shall I book it?"
    assert sentences(reply) == ["Just to confirm:", "2 prime tickets for Rocket Raja, total ₹520.", "Shall I book it?"]
    assert sentences("Note: closed Mondays.") == ["Note: closed Mondays."]  # nothing long after the colon
    assert sentences("Yes. Done, your booking is confirmed.") == ["Yes. Done, your booking is confirmed."]
    assert sentences("The fee is Rs. 30 per ticket. Anything else?") == ["The fee is Rs. 30 per ticket.", "Anything else?"]
    assert sentences("You get **25 %** back.") == ["You get 25 % back."]


def test_for_speech_reads_rupees_and_round_hours():
    assert for_speech("total ₹900 at 7:00") == "total rupees 900 at 7"


def test_pcm_round_trip_and_resample():
    x = np.sin(np.linspace(0, 200, RATE)).astype(np.float32) * 0.5
    y = pcm16_to_float(float_to_pcm16(x))
    assert np.abs(x - y).max() < 1e-3
    assert len(resample(x, RATE, 24_000)) == 24_000
    assert np.abs(load(wav_bytes(x)) - x).max() < 1e-3


def test_wer_normalisation_puts_numbers_in_digits():
    from fastlane.speech.wer import normalise

    assert normalise("Sravani Agarwal, forty-two, female.") == "sravani agarwal 42 female"
    assert normalise("Next Friday, 23rd October at 7 p.m.") == "next friday 23 october at 7 pm"
    assert normalise("twenty third of October, ₹1,500") == "23 of october rupees 1500"


def test_key_details_are_found_and_scored(world_path):
    from fastlane.speech.wer import Entities, detail_hits, normalise

    e = Entities.from_world(["Sravani Agarwal"], world_path)
    found = e.find(normalise("Kaala Samudram at Galaxy Cinemas Kukatpally on Friday 23rd October for Sravani Agarwal, 42"))
    assert found["films"] == ["kaala samudram"] and "kukatpally" in found["places"] and found["numbers"] == ["23", "42"]
    hits = detail_hits(found, normalise("Kala Samudram at Galaxy Cinemas Kukatpally on Friday 23 October for Sravani Agarwal 42"))
    assert hits["films"] == (0, 1) and hits["names"] == (1, 1) and hits["numbers"] == (2, 2)
