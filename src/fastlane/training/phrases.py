"""Phrase bank for generated training dialogues: how callers say things, how the agent asks.

Everything here is for TRAINING dialogues only. The policy questions and red flags are different
from the ones in the 360 test calls (fastlane.eval.scenarios) so test calls stay unseen.
"""

from __future__ import annotations

import random
from datetime import date

from fastlane.gateway.readback import spoken_time

FILLERS = ["", "", "", "Haan, ", "Okay, ", "Ji, ", "See, ", "Yes, ", "Hmm, ", "Sari, "]
POLITE_TAIL = ["", "", "", " please", " please.", " ji", " na"]


def say_date(iso: str, rng: random.Random) -> str:
    d = date.fromisoformat(iso)
    suffix = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    return rng.choice([f"{d.day}{suffix} {d.strftime('%B')}", f"{d.strftime('%B')} {d.day}", f"the {d.day}{suffix}",
                       f"{d.day} {d.strftime('%b')}", f"{d.strftime('%A')} the {d.day}{suffix}"])


def say_time(hhmm: str, rng: random.Random) -> str:
    h, m = map(int, hhmm.split(":"))
    h12 = h % 12 or 12
    forms = [spoken_time(hhmm), hhmm if rng.random() < 0.3 else spoken_time(hhmm)]
    if m == 30:
        forms.append(f"half past {h12}" + (" in the evening" if h >= 17 else ""))
    if m == 0:
        forms.append(f"{h12} o'clock" + (" in the morning" if h < 12 else ""))
    return rng.choice(forms)


def say_cinema(label: str, rng: random.Random) -> str:
    brand, area = label.split()[0], label.split()[-1]
    return rng.choice([label, brand, f"{brand} {area}", f"the {area} one", f"{brand} in {area}"])


def say_restaurant(name: str, rng: random.Random) -> str:
    area = name.replace("Tandoor Tales ", "")
    return rng.choice([name, area, f"the {area} branch", f"Tandoor Tales {area.split()[0]}"])


CITY_SAY = {"Bengaluru": ["Bengaluru", "Bangalore"], "Hyderabad": ["Hyderabad", "Hyd"], "Vijayawada": ["Vijayawada", "Bezawada"],
            "Chennai": ["Chennai", "Chennai"]}

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def say_number(n: int, rng: random.Random) -> str:
    return rng.choice([str(n), NUMBER_WORDS[n]]) if n < len(NUMBER_WORDS) else str(n)


def say_passengers(pax: list[dict], rng: random.Random) -> str:
    g = {"M": ["male", "male", "M"], "F": ["female", "female", "F"], "O": ["other"]}
    parts = [f"{p['name']}, {p['age']}, {rng.choice(g[p['gender']])}" for p in pax]
    return rng.choice(["; ".join(parts), " and ".join(parts), ". ".join(parts)])


def say_ref(ref: str, rng: random.Random) -> str:
    return rng.choice([ref, " ".join(ref), f"{ref[:2]} {ref[2:]}"])


def say_phone(phone: str, rng: random.Random) -> str:
    return rng.choice([phone, f"{phone[:5]} {phone[5:]}", " ".join(phone)])


# ----------------------------------------------------------------- agent asks

ASK = {
    "movie": ["Which film would you like to watch?", "Sure. Which movie?", "Which film are you looking for?"],
    "date": ["Which date would you like?", "For which day?", "On which date?"],
    "cinema": ["Which cinema would you prefer? We have Galaxy Kukatpally, Starlight Gachibowli, Regal Ameerpet and Moonbeam Secunderabad.",
               "Which cinema - Galaxy, Starlight, Regal or Moonbeam?", "At which cinema?"],
    "time": ["Which show time would you like?", "What time would suit you?", "Which show - morning, afternoon or evening?"],
    "seat_category": ["Classic, prime or recliner seats?", "Which seats would you like - classic, prime or recliner?"],
    "quantity": ["How many tickets?", "How many tickets do you need?"],
    "name": ["May I have the name for the booking?", "Under which name should I book?", "Your name for the booking, please?"],
    "destination": ["Where would you like to travel to?", "From where to where are you travelling?", "Which city are you going to?"],
    "bus_time": ["Which bus timing would you like?", "What departure time would you prefer?"],
    "passengers": ["Please tell me each passenger's name, age and gender.", "May I have the name, age and gender of every passenger?"],
    "restaurant": ["Which branch - Jubilee Hills, Banjara Hills or Hitech City?", "Which Tandoor Tales branch would you like?"],
    "res_time": ["What time would you like the table?", "For what time?"],
    "party_size": ["How many guests?", "For how many people?", "Table for how many?"],
    "booking_id": ["Sure. May I have the booking reference?", "Please tell me your booking reference.", "Could you give me the booking reference number?"],
    "new_time": ["Which time would you like to move it to?", "What new time would you like?"],
    "new_date": ["Which date would you like to change it to?", "To which date?"],
    "new_party": ["How many guests will there be now?"],
    "problem": ["I'm sorry to hear that. Could you tell me what happened?", "Sure, please tell me the problem."],
}

GIVE = {
    "movie": ["{v}", "{v} please", "I want to watch {v}", "the movie is {v}", "{v} movie"],
    "date": ["{v}", "on {v}", "for {v}", "{v} please"],
    "cinema": ["{v}", "at {v}", "{v} is fine", "{v} please"],
    "time": ["the {v} show", "{v}", "{v} show please"],
    "seat_category": ["{v}", "{v} seats", "{v} please"],
    "quantity": ["{v} tickets", "{v}", "we are {v} people", "{v} tickets please"],
    "name": ["{v}", "my name is {v}", "book it under {v}", "it's {v}", "name is {v}"],
    "destination": ["to {v}", "{v}", "I'm going to {v}"],
    "route": ["from {o} to {v}", "{o} to {v}"],
    "bus_time": ["the {v} bus", "{v}", "{v} one"],
    "passengers": ["{v}", "passengers are {v}", "{v}, that's all"],
    "restaurant": ["{v}", "{v} please", "at {v}"],
    "party_size": ["{v} people", "{v} of us", "table for {v}", "we are {v}"],
    "booking_id": ["it's {v}", "{v}", "reference is {v}", "booking number {v}"],
}

CONFIRM = ["Yes, correct.", "Yes, please go ahead.", "Haan, that's right.", "Yes.", "Correct, book it.", "Theek hai, go ahead.",
           "Yes yes, all correct.", "Avunu, correct.", "That's right, please confirm.", "Okay, go ahead."]
BYE = ["No, that's all. Thank you!", "Thanks, bye.", "Nothing else, thank you.", "Okay, thank you. Bye.", "That's it, thanks!"]
ACCEPT_ALT = ["Okay, the {v} one is fine.", "Fine, give me {v}.", "Okay, {v} then.", "Alright, {v} works."]

# ------------------------------------------------- training policy Q&A (not the test questions)

POLICY_QA = {
    "movies": [
        ("How many tickets can I book in one go?", "You can book up to 10 tickets in one booking. For bigger groups our staff can help."),
        ("Until when can I change my movie booking?", "You can change it up to 4 hours before the show, subject to availability."),
        ("Do children need their own ticket?", "Children aged 3 and above need a ticket. Children below 3 can sit on a parent's lap, except for A-certified films."),
        ("Can I choose my exact seat numbers?", "Not by phone - we assign the best block of seats together in your chosen category."),
        ("Is there an interval in the film?", "Yes, films longer than 100 minutes have a 15-minute interval."),
        ("Do you give student discounts?", "Student and senior-citizen discounts are only at the box office with a valid ID, not on phone bookings."),
        ("Which metro station is near Moonbeam Cinemas?", "The nearest metro station is Paradise."),
        ("Are subtitles there for Telugu films?", "Yes, all non-English films have English subtitles."),
    ],
    "buses": [
        ("How many seats can I book at once?", "You can book up to 6 seats in one booking."),
        ("If I cancel a day and a half before, how much do I get back?", "Cancelling between 24 and 72 hours before departure gives you 75% of the fare back."),
        ("Can I cancel an hour before the bus leaves?", "No, tickets can't be cancelled within 4 hours of departure."),
        ("Is food served on the bus?", "No food is served on board. AC buses give water, and long routes have one meal stop."),
        ("Which ID should I carry on the bus?", "Please carry a government photo ID like Aadhaar, driving licence, passport or voter ID."),
        ("Is there a toilet on the bus?", "No, but the bus stops for a comfort break every 3 to 4 hours."),
        ("Where does the Chennai bus board in Hyderabad?", "The Chennai bus boards at Ameerpet Metro in Hyderabad."),
        ("Can I change the passenger name later?", "Spelling corrections are allowed up to 12 hours before departure, but a ticket can't be given to a different person."),
    ],
    "restaurants": [
        ("What time does dinner start?", "Dinner seatings start at 7 PM. The last seating is 10 PM, or 9:30 PM at Hitech City."),
        ("How long will you hold the table if we are late?", "We hold the table for 15 minutes after the seating time."),
        ("Do you have a kids' menu?", "Yes, a child's plate is Rs 220, and high chairs are available at all branches."),
        ("Is there outdoor seating at Jubilee Hills?", "Yes, Jubilee Hills has outdoor terrace tables."),
        ("Is the butter chicken nut free?", "No, the butter chicken gravy has cashew in it."),
        ("Which breads don't have egg?", "Tandoori roti, jowar roti, lachha paratha and missi roti are egg-free; the naan dough has egg."),
        ("Do you have a lunch buffet?", "There's a weekday lunch buffet at Hitech City only, Rs 499 per person."),
        ("Can I pay with a meal card?", "Yes, cards, UPI, Sodexo and Pluxee meal cards are accepted at the restaurant."),
    ],
}

RED_FLAGS = [
    ("The popcorn counter staff were very rude to me yesterday, I want to complain.", "complaint"),
    ("My money got deducted twice for my ticket, I need it back.", "payment"),
    ("I want tickets for 25 students from our school.", "large_group"),
    ("Can you book me a cab to the cinema?", "out_of_scope"),
    ("I left my spectacles in the hall yesterday.", "other"),
    ("I want to speak to your manager right now.", "caller_request"),
    ("The food we got last night was stale, I want to complain.", "complaint"),
    ("My refund for last week's cancellation still hasn't come.", "payment"),
    ("Please book a flight to Chennai for me.", "out_of_scope"),
    ("Can you arrange catering for 50 people at my house?", "out_of_scope"),
    ("The bus conductor misbehaved with my mother, this is unacceptable.", "complaint"),
    ("Can I talk to a real person please?", "caller_request"),
]
