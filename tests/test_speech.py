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


def test_roundtrip_keeps_only_transcripts_with_every_key_detail(world_path):
    from fastlane.speech.wer import Entities
    from fastlane.training.roundtrip import lost_details, people_names

    e = Entities.from_world(people_names(), world_path)
    line = "Ji, the movie is Kaala Samudram, we are 2 people, on October 28"
    assert lost_details(line, "ji the movie is Kaala Samudram we are two people on October 28th", e) == []
    assert lost_details(line, "the movie is kala samudram we are to people on october 28", e) == ["films:kaala samudram", "numbers:2"]
    assert lost_details("No, make it 3", "Now make it three", e) == ["yes/no"]  # a "no" lost would confirm a read-back
    assert lost_details("Haan, Chennai Nights movie na", "Han, Chennai Nights, maybe not.", e) == ["yes/no"]


def test_roundtrip_noise_replaces_caller_lines_only():
    import random

    from fastlane.training.roundtrip import RoundTripNoise, Trip

    trips = [Trip("two tickets", 0, "hf_alpha", 1.0, 20.0, "Two tickets.", 900.0),
             Trip("at Galaxy", 0, "af_heart", 1.0, 20.0, "at gallery", 900.0, lost=["places:galaxy"])]
    noise = RoundTripNoise(trips)
    content = "CONVERSATION:\nAgent: How many tickets?\nCaller: two tickets\nCaller: at Galaxy"
    assert noise.user_content(content, random.Random(0)) == (
        "CONVERSATION:\nAgent: How many tickets?\nCaller: Two tickets.\nCaller: at Galaxy")  # lost a detail: kept clean


def test_phone_channel_adds_noise_at_the_asked_level():
    from fastlane.training.roundtrip import _band, phone_channel

    t = np.arange(RATE) / RATE
    tone = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    out = phone_channel(tone, np.random.default_rng(0), snr_db=20.0)
    assert len(out) == len(tone) + int(0.5 * RATE) and np.abs(out).max() <= 1.0
    low = _band((0.3 * np.sin(2 * np.pi * 100 * t)).astype(np.float32))  # below the telephone band
    assert np.abs(low).max() < 0.01


def test_roundtrip_booking_reference_must_survive(world_path):
    from fastlane.speech.wer import Entities
    from fastlane.training.roundtrip import booking_refs, lost_details

    e = Entities.from_world([], world_path)
    assert booking_refs("the 6 AM AC seater bus") == [] and booking_refs("reference is MV GVPUZ") == ["MVGVPUZ"]
    assert lost_details("reference is MV GVPUZ", "Reference is MVGVPUS.", e) == ["reference:MVGVPUZ"]
    assert lost_details("it's MV X9K63", "It's MVX 9K63.", e) == []
    assert lost_details("Haan, the 17:00 show", "Han, the seventeen show.", e) == []
    assert lost_details("Sari, for 26 Oct ji", "sari, for 26 Arc Jai.", e) == ["short:oct"]
