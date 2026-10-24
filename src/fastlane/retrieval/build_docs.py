"""Generate the fact-heavy policy documents (menus, fares, calendar, venues) from the world data.

Hand-written policies live next to them in docs/. Generating the rest from the same constants the
world is built from keeps documents and database consistent. Run: `fastlane docs build`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from fastlane.config import DOCS_DIR, FESTIVALS, TODAY
from fastlane.gateway import rules
from fastlane.gateway.readback import spoken_time
from fastlane.world.seed import (
    BASE_PRICES,
    CINEMAS,
    MOVIES,
    RESTAURANTS,
    ROUTES,
    SCREEN_LAYOUTS,
    SHOW_SLOTS,
    TABLE_LAYOUTS,
    TRIP_TEMPLATES,
)

ALLERGENS = {
    "nuts": "tree nuts (cashew, almond, pistachio)",
    "peanut": "peanuts",
    "gluten": "wheat / gluten",
    "dairy": "milk and milk products (ghee, paneer, cream, butter)",
    "egg": "egg",
    "shellfish": "prawns and crab",
    "fish": "fish",
    "soy": "soy",
    "sesame": "sesame",
    "mustard": "mustard",
}


@dataclass(frozen=True)
class Dish:
    name: str
    section: str
    price: int
    diet: str  # veg | nonveg | vegan
    allergens: tuple[str, ...]
    spice: int  # 0-3
    jain: bool  # can be made without onion/garlic
    description: str


DISHES = [
    # Veg starters
    Dish("Paneer Tikka", "Starters - Vegetarian", 320, "veg", ("dairy", "mustard"), 2, True, "Cottage cheese cubes marinated in yoghurt and spices, charred in the tandoor."),
    Dish("Hara Bhara Kebab", "Starters - Vegetarian", 280, "veg", ("nuts", "gluten"), 1, False, "Spinach, peas and potato patties with cashew, shallow fried."),
    Dish("Tandoori Aloo", "Starters - Vegetarian", 260, "vegan", ("mustard",), 2, False, "Baby potatoes stuffed with spiced filling, roasted in the tandoor. Vegan."),
    Dish("Dahi Ke Kebab", "Starters - Vegetarian", 290, "veg", ("dairy", "gluten"), 1, True, "Hung-curd patties with green chilli, crumb-coated."),
    Dish("Malai Broccoli", "Starters - Vegetarian", 300, "veg", ("dairy", "nuts"), 0, True, "Broccoli florets in a cream, cheese and cashew marinade."),
    Dish("Chilli Baby Corn", "Starters - Vegetarian", 270, "vegan", ("soy", "gluten"), 3, False, "Crisp baby corn tossed with soy, chilli and spring onion."),
    Dish("Peanut Chaat", "Starters - Vegetarian", 220, "vegan", ("peanut",), 2, True, "Roasted peanuts with onion, tomato, lime and chaat masala."),
    # Non-veg starters
    Dish("Murgh Malai Tikka", "Starters - Non-Vegetarian", 380, "nonveg", ("dairy", "nuts"), 0, False, "Chicken in cream, cheese and cashew marinade, mild."),
    Dish("Tandoori Chicken (half)", "Starters - Non-Vegetarian", 360, "nonveg", ("dairy", "mustard"), 2, False, "Bone-in chicken in red yoghurt marinade."),
    Dish("Chicken 65", "Starters - Non-Vegetarian", 340, "nonveg", ("egg", "gluten"), 3, False, "Hyderabad-style fried chicken with curry leaves and red chilli."),
    Dish("Mutton Seekh Kebab", "Starters - Non-Vegetarian", 420, "nonveg", (), 2, False, "Minced mutton skewers with ginger and green chilli."),
    Dish("Apollo Fish", "Starters - Non-Vegetarian", 410, "nonveg", ("fish", "egg", "gluten"), 3, False, "Boneless fish fried and tossed in a spicy yoghurt-chilli sauce."),
    Dish("Tandoori Prawns", "Starters - Non-Vegetarian", 520, "nonveg", ("shellfish", "dairy", "mustard"), 2, False, "Jumbo prawns in ajwain and yoghurt marinade."),
    Dish("Egg Bhurji Pav", "Starters - Non-Vegetarian", 240, "nonveg", ("egg", "gluten", "dairy"), 2, False, "Spiced scrambled egg with buttered pav."),
    # Veg mains
    Dish("Dal Makhani", "Mains - Vegetarian", 300, "veg", ("dairy",), 1, False, "Black lentils slow-cooked overnight with butter and cream. Contains no nuts."),
    Dish("Dal Tadka", "Mains - Vegetarian", 240, "vegan", (), 2, True, "Yellow lentils tempered with cumin and garlic; vegan when made with oil (ask)."),
    Dish("Paneer Butter Masala", "Mains - Vegetarian", 340, "veg", ("dairy", "nuts"), 1, True, "Paneer in a tomato, butter and cashew gravy."),
    Dish("Palak Paneer", "Mains - Vegetarian", 320, "veg", ("dairy",), 1, True, "Paneer in a spiced spinach puree."),
    Dish("Kadai Vegetable", "Mains - Vegetarian", 290, "vegan", (), 2, False, "Mixed vegetables with peppers and kadai masala. Vegan."),
    Dish("Malai Kofta", "Mains - Vegetarian", 330, "veg", ("dairy", "nuts", "gluten"), 0, False, "Paneer and potato dumplings in a rich cashew gravy."),
    Dish("Chana Masala", "Mains - Vegetarian", 260, "vegan", (), 2, False, "Chickpeas in an onion-tomato masala. Vegan."),
    Dish("Bagara Baingan", "Mains - Vegetarian", 280, "vegan", ("peanut", "sesame"), 2, False, "Hyderabadi baby aubergines in a peanut, sesame and coconut gravy."),
    Dish("Mirchi Ka Salan", "Mains - Vegetarian", 260, "vegan", ("peanut", "sesame"), 3, False, "Long green chillies in a tangy peanut-sesame gravy."),
    # Non-veg mains
    Dish("Butter Chicken", "Mains - Non-Vegetarian", 420, "nonveg", ("dairy", "nuts"), 1, False, "Tandoori chicken in a tomato, butter and cashew gravy."),
    Dish("Chicken Chettinad", "Mains - Non-Vegetarian", 400, "nonveg", (), 3, False, "Chicken with black pepper, fennel and coconut."),
    Dish("Mutton Rogan Josh", "Mains - Non-Vegetarian", 480, "nonveg", ("dairy",), 2, False, "Kashmiri-style mutton curry with yoghurt and red chilli."),
    Dish("Goan Fish Curry", "Mains - Non-Vegetarian", 450, "nonveg", ("fish", "mustard"), 2, False, "Fish in coconut and kokum curry."),
    Dish("Prawn Masala", "Mains - Non-Vegetarian", 520, "nonveg", ("shellfish",), 2, False, "Prawns in an onion-tomato masala."),
    Dish("Haleem (seasonal)", "Mains - Non-Vegetarian", 380, "nonveg", ("gluten", "dairy", "nuts"), 1, False, "Slow-cooked wheat and mutton porridge with fried onion and cashew. Weekends only."),
    # Breads
    Dish("Tandoori Roti", "Breads", 50, "vegan", ("gluten",), 0, True, "Whole-wheat bread from the tandoor. Vegan without butter."),
    Dish("Butter Naan", "Breads", 70, "veg", ("gluten", "dairy", "egg"), 0, True, "Leavened bread with butter. The dough contains egg."),
    Dish("Garlic Naan", "Breads", 80, "veg", ("gluten", "dairy", "egg"), 0, False, "Naan topped with garlic and coriander."),
    Dish("Lachha Paratha", "Breads", 75, "veg", ("gluten", "dairy"), 0, True, "Layered whole-wheat bread with ghee."),
    Dish("Jowar Roti", "Breads", 60, "vegan", (), 0, True, "Sorghum flatbread. Gluten-free and vegan."),
    Dish("Missi Roti", "Breads", 65, "vegan", ("gluten",), 1, False, "Gram-flour and wheat bread with spices."),
    # Rice and biryani
    Dish("Hyderabadi Chicken Dum Biryani", "Rice and Biryani", 420, "nonveg", ("dairy", "nuts"), 2, False, "Kacchi-style biryani with saffron, fried onion and cashew; served with salan and raita."),
    Dish("Mutton Dum Biryani", "Rice and Biryani", 520, "nonveg", ("dairy", "nuts"), 2, False, "Mutton biryani, dum-cooked; served with salan and raita."),
    Dish("Vegetable Dum Biryani", "Rice and Biryani", 340, "veg", ("dairy", "nuts"), 2, False, "Seasonal vegetables and basmati, dum-cooked; served with salan and raita."),
    Dish("Jeera Rice", "Rice and Biryani", 180, "vegan", (), 0, True, "Basmati rice with cumin."),
    Dish("Curd Rice", "Rice and Biryani", 190, "veg", ("dairy", "mustard"), 0, True, "Rice with yoghurt, tempered with mustard seeds and curry leaves."),
    # Desserts
    Dish("Double Ka Meetha", "Desserts", 180, "veg", ("gluten", "dairy", "nuts"), 0, True, "Hyderabadi bread pudding with saffron milk and dry fruits."),
    Dish("Qubani Ka Meetha", "Desserts", 220, "veg", ("dairy", "nuts"), 0, True, "Stewed apricots with cream and almonds."),
    Dish("Gulab Jamun", "Desserts", 150, "veg", ("dairy", "gluten"), 0, True, "Milk dumplings in rose syrup."),
    Dish("Kulfi Falooda", "Desserts", 200, "veg", ("dairy", "nuts", "gluten"), 0, True, "Pistachio kulfi with vermicelli and basil seeds."),
    Dish("Fruit Bowl", "Desserts", 160, "vegan", (), 0, True, "Seasonal cut fruit. Vegan and gluten-free."),
    # Beverages
    Dish("Masala Chaas", "Beverages", 90, "veg", ("dairy",), 1, True, "Spiced buttermilk."),
    Dish("Sweet Lassi", "Beverages", 120, "veg", ("dairy",), 0, True, "Thick sweet yoghurt drink."),
    Dish("Fresh Lime Soda", "Beverages", 100, "vegan", (), 0, True, "Sweet, salted or mixed."),
    Dish("Irani Chai", "Beverages", 60, "veg", ("dairy",), 0, True, "Hyderabadi tea with condensed milk."),
    Dish("Tender Coconut Water", "Beverages", 110, "vegan", (), 0, True, "Served fresh."),
]

BRANCH_SPECIALS = {
    "tandoor-jubilee": [
        Dish("Raan-e-Sikandari", "Chef's Specials", 1450, "nonveg", ("dairy", "nuts"), 2, False, "Whole leg of lamb, slow-roasted; serves 4. Order 24 hours ahead."),
        Dish("Truffle Malai Paneer", "Chef's Specials", 460, "veg", ("dairy",), 0, True, "Paneer in a mild truffle-cream sauce."),
    ],
    "tandoor-banjara": [
        Dish("Nalli Nihari", "Chef's Specials", 540, "nonveg", ("gluten",), 2, False, "Slow-cooked lamb shank stew, Old City style."),
        Dish("Pathar Ka Gosht", "Chef's Specials", 560, "nonveg", ("dairy",), 2, False, "Mutton cooked on a hot stone."),
    ],
    "tandoor-hitech": [
        Dish("Jain Thali", "Chef's Specials", 380, "veg", ("dairy", "gluten"), 1, True, "Complete Jain meal with no onion, garlic or root vegetables. Lunch only."),
        Dish("Millet Khichdi Bowl", "Chef's Specials", 300, "vegan", (), 1, True, "Foxtail millet and moong dal khichdi. Vegan and gluten-free."),
    ],
}
# Banjara Hills has no seafood licence; dishes with fish/shellfish are not served there.
BRANCH_EXCLUDES = {"tandoor-banjara": ("fish", "shellfish")}


def _front(doc_id: str, title: str, domain: str, version: str = "2026-10", effective: str = "2026-10-01") -> str:
    return f"---\ndoc_id: {doc_id}\ntitle: {title}\ndomain: {domain}\nversion: {version}\neffective_from: {effective}\n---\n\n# {title}\n\n"


def _diet_label(d: Dish) -> str:
    label = {"veg": "Vegetarian", "nonveg": "Non-vegetarian", "vegan": "Vegan"}[d.diet]
    if d.jain:
        label += "; Jain option available"
    return label


def menu_doc(rid: str, name: str) -> str:
    excluded = BRANCH_EXCLUDES.get(rid, ())
    dishes = [d for d in DISHES if not set(d.allergens) & set(excluded)] + BRANCH_SPECIALS[rid]
    out = _front(f"menu_{rid.split('-')[1]}", f"{name} Menu with Allergens", "restaurants")
    out += (f"Prices in rupees, inclusive of GST. Spice level 0 (mild) to 3 (hot). Allergens are listed for every dish; "
            f"the kitchen handles nuts, gluten, dairy, egg, fish, shellfish, sesame, soy and mustard, so traces are possible "
            f"in any dish. Guests with severe allergies should tell the staff on arrival.\n\n")
    if excluded:
        out += f"{name} does not serve fish or seafood.\n\n"
    sections: dict[str, list[Dish]] = {}
    for d in dishes:
        sections.setdefault(d.section, []).append(d)
    for section, items in sections.items():
        out += f"## {section}\n\n"
        for d in items:
            allergens = ", ".join(d.allergens) if d.allergens else "none of the listed allergens"
            out += f"- **{d.name}** - Rs {d.price}. {d.description} {_diet_label(d)}. Spice {d.spice}/3. Allergens: {allergens}.\n"
        out += "\n"
    return out


def allergen_guide() -> str:
    out = _front("allergen_guide", "Tandoor Tales Allergen and Dietary Guide", "restaurants")
    out += ("This guide lists which dishes are safe for common dietary needs across all branches. Branch menus give the full "
            "list with prices. Dishes marked 'Jain option' are made without onion, garlic, potato or other root vegetables on request.\n\n")
    groups = {
        "Nut-free dishes (no tree nuts or peanuts)": [d for d in DISHES if not {"nuts", "peanut"} & set(d.allergens)],
        "Gluten-free dishes": [d for d in DISHES if "gluten" not in d.allergens],
        "Vegan dishes": [d for d in DISHES if d.diet == "vegan"],
        "Dishes with a Jain option": [d for d in DISHES if d.jain],
        "Dairy-free dishes": [d for d in DISHES if "dairy" not in d.allergens],
        "Egg-free breads": [d for d in DISHES if d.section == "Breads" and "egg" not in d.allergens],
    }
    for title, items in groups.items():
        out += f"## {title}\n\n" + ", ".join(d.name for d in items) + ".\n\n"
    out += ("## Allergen definitions\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in ALLERGENS.items()) + "\n\n")
    out += ("## Notes\n\nThe naan dough contains egg; for an egg-free bread choose tandoori roti, jowar roti, lachha paratha "
            "or missi roti. Biryanis are garnished with cashew; a nut-free biryani can be made on request with 30 minutes' notice "
            "if ordered when booking. The salan served with biryani contains peanuts and sesame.\n")
    return out


def festival_doc() -> str:
    out = _front("festival_calendar", "Festival Calendar and Holiday Arrangements 2026", "general", "2026-10", "2026-01-01")
    out += ("Dates below are the calendar used for all bookings this year. When a caller names a festival, the booking date is "
            "the date in this table.\n\n| Festival | Date | Day |\n|---|---|---|\n")
    for name, d in sorted(FESTIVALS.items(), key=lambda kv: kv[1]):
        out += f"| {name} | {d.day} {d.strftime('%B %Y')} | {d.strftime('%A')} |\n"
    out += """
## Festival pricing

On festival days the weekend pricing applies: movie tickets cost Rs 40 more in Classic and Prime and Rs 50 more
in Recliner, and bus fares are 15% higher.

## Diwali arrangements

- **Diwali Dhamaka**, the big Diwali release, opens on Friday 6 November. Opening-day shows sell out fast.
- All Tandoor Tales branches take reservations on Diwali (8 November) but dinner seatings are usually fully booked
  well in advance; lunch is easier to get.
- Buses from Hyderabad to Bengaluru, Chennai and Vijayawada on the evening of Friday 6 November are in heavy demand.

## Dussehra arrangements

Dussehra falls on Tuesday 20 October. Evening buses from Hyderabad to Vijayawada on Monday 19 October (the day before)
fill early. Cinemas run normal show times.

## Saddula Bathukamma

Sunday 18 October. Cinemas and restaurants are open as usual; traffic near Tank Bund is diverted in the evening, so
guests travelling to Secunderabad should allow extra time.
"""
    return out


def movie_fares_doc() -> str:
    out = _front("movie_fares", "Movie Ticket Prices and Show Times", "movies")
    times = ", ".join(spoken_time(f"{s // 60:02d}:{s % 60:02d}") for s in SHOW_SLOTS)
    out += f"Every screen runs four shows a day at {times}. Ticket prices per seat, including GST:\n\n"
    out += "| Category | Weekday | Weekday morning (10 AM) | Fri-Sun and festivals |\n|---|---|---|---|\n"
    for cat, base in BASE_PRICES.items():
        morning = base - 30 if cat != "recliner" else base + 10
        peak = base + 40 if cat != "recliner" else base + 50
        weekday = base if cat != "recliner" else base + 10
        out += f"| {cat.title()} | Rs {weekday} | Rs {morning} | Rs {peak} |\n"
    out += ("\nMorning-show discounts apply to Classic and Prime only, on every day (so a Saturday 10 AM Classic ticket is "
            f"Rs {BASE_PRICES['classic'] + 10}). "
            f"Cancellation fee: Rs {rules.MOVIE_CANCEL_FEE_PER_TICKET} per ticket.\n\n## Screens\n\n")
    for sname, rows, per_row, cats in SCREEN_LAYOUTS:
        desc = "; ".join(f"{k} rows {'-'.join([v[0], v[-1]]) if len(v) > 1 else v}" for k, v in cats.items())
        out += f"- **{sname}**: {len(rows) * per_row} seats ({len(rows)} rows of {per_row}). {desc}.\n"
    return out


def films_doc() -> str:
    out = _front("films_now_showing", "Films Now Showing and Coming Soon", "movies")
    blurbs = {
        "kaala-samudram": "A coastal smuggling saga set in 1980s Kakinada, with spectacular sea action sequences.",
        "dil-ki-dastak": "A romance between a Lucknow poet and a Mumbai startup founder, with music by a new composer.",
        "last-monsoon": "A quiet family drama about three sisters returning to their Kerala home during the monsoon.",
        "rocket-raja": "A comedy about a Bihar village mechanic who builds a rocket for the national science fair.",
        "godavari-express": "A thriller on an overnight train from Hyderabad to Rajahmundry where a passenger vanishes.",
        "shadow-protocol": "An English-language spy action film with an Indian intelligence officer in Istanbul.",
        "little-champs": "Animated adventure in which four school children accidentally launch themselves to the Moon.",
        "chennai-nights": "A neo-noir thriller about a night-shift taxi driver in Chennai who witnesses a murder.",
        "diwali-dhamaka": "A family comedy about a joint family's chaotic Diwali reunion. The big festive release.",
    }
    for m in MOVIES:
        status = "Now showing" if m.release <= TODAY else f"Releases {m.release.day} {m.release.strftime('%B')}"
        out += (f"## {m.title}\n\n{status}. {m.language}, {m.genre}, certificate {m.certificate}, "
                f"{m.runtime_min // 60} h {m.runtime_min % 60} min. {blurbs[m.id]} "
                "Subtitles: English subtitles on all non-English films.\n\n")
    return out


def cinemas_doc() -> str:
    out = _front("cinemas_guide", "Cinema Locations, Parking and Facilities", "movies")
    extra = {
        "galaxy-kukatpally": "Mall parking (Rs 60 for 4 hours). Food court on the same floor. Nearest metro: KPHB Colony.",
        "starlight-gachibowli": "Free basement parking for 3 hours with a ticket. Cafe in the lobby. Nearest metro: none; cabs recommended.",
        "regal-ameerpet": "No dedicated parking; use the metro parking lot opposite (Rs 40). Single-level, older building with lifts. Nearest metro: Ameerpet.",
        "moonbeam-secunderabad": "Paid parking (Rs 50). Recliner lounge with food service in Screen 1. Nearest metro: Paradise.",
    }
    for cid, name, area, address in CINEMAS:
        out += f"## {name} {area}\n\nAddress: {address}, Hyderabad. Three screens. {extra[cid]}\n\n"
    return out


def buses_doc() -> str:
    out = _front("bus_routes_and_fares", "Bus Routes, Timings, Fares and Boarding Points", "buses")
    out += ("Fares are per seat or berth on weekdays (Monday-Thursday); Friday to Sunday and festival fares are 15% higher, "
            "rounded to the nearest Rs 10.\n\n")
    pair_names = {"BLR": ("Hyderabad", "Bengaluru"), "VJA": ("Hyderabad", "Vijayawada"), "CHE": ("Hyderabad", "Chennai")}
    for pair, trips in TRIP_TEMPLATES.items():
        a, b = pair_names[pair]
        r1 = next(r for r in ROUTES if r[1] == a and r[2] == b)
        r2 = next(r for r in ROUTES if r[1] == b and r[2] == a)
        out += f"## {a} - {b}\n\nDistance {r1[3]} km, journey about {r1[4] // 60} h {r1[4] % 60:02d} min. "
        out += f"Boarding: {r1[5]} (from {a}); {r2[5]} (from {b}). Same timings in both directions.\n\n"
        out += "| Departs | Bus type | Weekday fare |\n|---|---|---|\n"
        for dep, btype, fare in trips:
            out += f"| {spoken_time(f'{dep // 60:02d}:{dep % 60:02d}')} | {btype.replace('_', ' ').replace('nonac', 'non-AC').replace('ac ', 'AC ')} | Rs {fare} |\n"
        out += "\n"
    out += """## Bus types

- **AC sleeper**: 30 berths (15 lower, 15 upper), blanket and water bottle provided, charging point at every berth.
- **Non-AC sleeper**: 30 berths, no blanket.
- **AC seater**: 40 push-back seats in 2+2 layout; window seats are 1, 4, 5, 8 and so on.
- **Non-AC seater**: 40 seats, 2+2.

Lower berths cannot be guaranteed but the system assigns them first unless an upper berth is asked for.

## Luggage and travel rules

Each passenger may carry one suitcase up to 15 kg plus one small bag. Pets are not allowed, except service dogs.
Carry a government photo ID (Aadhaar, driving licence, passport, voter ID); the conductor may check it against the
passenger name. Buses stop once for a 20-minute meal break on routes longer than 6 hours. Smoking and alcohol are
not allowed on board.
"""
    return out


def restaurants_doc() -> str:
    out = _front("restaurants_guide", "Tandoor Tales Branches, Hours and Seating", "restaurants")
    zones = {"indoor": "indoor", "outdoor": "outdoor terrace", "private": "private dining room"}
    for rid, name, area, address, max_party, dep_min, dep_pp, lunch, dinner, closed in RESTAURANTS:
        tables = TABLE_LAYOUTS[rid]
        seating = "; ".join(f"{n} x {s}-seat {zones[z]}" for s, z, n in tables)
        closed_txt = "Closed on Mondays." if closed == 0 else "Open every day."
        out += (f"## {name}\n\nAddress: {address}, Hyderabad. {closed_txt} Lunch seatings "
                f"{spoken_time(f'{lunch[0] // 60:02d}:{lunch[0] % 60:02d}')} to {spoken_time(f'{lunch[1] // 60:02d}:{lunch[1] % 60:02d}')}, "
                f"dinner seatings {spoken_time(f'{dinner[0] // 60:02d}:{dinner[0] % 60:02d}')} to {spoken_time(f'{dinner[1] // 60:02d}:{dinner[1] % 60:02d}')}. "
                f"Tables: {seating}. Phone bookings up to {max_party} guests; deposit of Rs {dep_pp} per guest for parties of "
                f"{dep_min} or more. Valet parking available in the evening (Rs 100).\n\n")
    return out


SNACKS = [
    ("Salted Popcorn (regular)", 180, "vegan", ()),
    ("Salted Popcorn (large)", 260, "vegan", ()),
    ("Caramel Popcorn (large)", 300, "veg", ("dairy",)),
    ("Cheese Popcorn (large)", 320, "veg", ("dairy",)),
    ("Peri-Peri Fries", 190, "vegan", ()),
    ("Veg Puff", 90, "veg", ("gluten", "dairy")),
    ("Chicken Puff", 110, "nonveg", ("gluten", "egg", "dairy")),
    ("Paneer Tikka Sandwich", 220, "veg", ("gluten", "dairy", "mustard")),
    ("Chicken Club Sandwich", 260, "nonveg", ("gluten", "egg", "dairy")),
    ("Nachos with Cheese", 240, "veg", ("dairy",)),
    ("Samosa (2 pcs)", 120, "vegan", ("gluten",)),
    ("Chocolate Brownie", 160, "veg", ("gluten", "dairy", "egg", "nuts")),
    ("Cold Coffee", 180, "veg", ("dairy",)),
    ("Soft Drink (regular)", 120, "vegan", ()),
    ("Mineral Water", 40, "vegan", ()),
]
COMBOS = [
    ("Couple Combo", "2 regular salted popcorn + 2 soft drinks", 420),
    ("Family Combo", "2 large popcorn (any flavour) + 4 soft drinks + nachos", 990),
    ("Kids Combo", "regular salted popcorn + juice box + chocolate", 260),
]


def snacks_doc() -> str:
    out = _front("cinema_snacks", "Cinema Snacks, Combos and Allergens", "movies")
    out += ("Snack counters at all four cinemas sell the items below. Moonbeam Cinemas Screen 1 (recliners) offers seat "
            "service: press the call button. Allergens are listed per item; popcorn is popped in coconut oil.\n\n## Snacks\n\n")
    for name, price, diet, allergens in SNACKS:
        label = {"veg": "Vegetarian", "nonveg": "Non-vegetarian", "vegan": "Vegan"}[diet]
        out += f"- **{name}** - Rs {price}. {label}. Allergens: {', '.join(allergens) or 'none listed'}.\n"
    out += "\n## Combos\n\n" + "".join(f"- **{n}** - {d}, Rs {p}.\n" for n, d, p in COMBOS)
    out += ("\n## Pre-ordering snacks\n\nSnacks cannot be added to a phone booking. They are bought at the counter or ordered "
            "from the seat in recliner screens.\n")
    return out


STOPS = {
    "Hyderabad-Bengaluru": ["Ameerpet Metro (start)", "Kukatpally Y Junction (+20 min)", "Shamshabad toll plaza (+55 min)",
                            "Jadcherla (+1 h 50 min, meal stop at Highway Dhaba)", "Kurnool bypass (+3 h 45 min)",
                            "Anantapur (+6 h 10 min)", "Hebbal, Bengaluru (+9 h 15 min)", "Majestic Bus Stand (end)"],
    "Hyderabad-Vijayawada": ["LB Nagar Ring Road (start)", "Hayathnagar (+15 min)", "Choutuppal (+45 min)",
                             "Suryapet (+2 h 10 min, meal stop)", "Nandigama (+4 h 10 min)", "Pandit Nehru Bus Station (end)"],
    "Hyderabad-Chennai": ["Ameerpet Metro (start)", "Mehdipatnam (+25 min)", "Shamshabad (+55 min)", "Kurnool (+3 h 50 min, meal stop)",
                          "Kadapa (+6 h 30 min)", "Tirupati bypass (+8 h 15 min)", "Koyambedu, Chennai (end)"],
}


def bus_stops_doc() -> str:
    out = _front("bus_stops", "Bus Stops and Pick-up Points", "buses")
    out += ("Buses pick up and drop at the points below, in order. Times are approximate offsets from departure. "
            "Return buses stop at the same points in reverse order. Tell the assistant your pick-up point when booking.\n\n")
    for route, stops in STOPS.items():
        out += f"## {route}\n\n" + "\n".join(f"{i}. {stop}" for i, stop in enumerate(stops, 1)) + "\n\n"
    out += ("## Boarding point facilities\n\nAmeerpet Metro: covered waiting area under the metro station, washrooms in the "
            "station. LB Nagar: open lay-by near the ring road, a tea stall. Majestic, Koyambedu and Pandit Nehru Bus Station: "
            "platform numbers are sent by SMS 2 hours before departure.\n")
    return out


def seat_maps_doc() -> str:
    out = _front("seat_maps", "Cinema Seat Maps", "movies")
    out += ("All four cinemas use the same three screen layouts. Row A is closest to the screen. Seat 1 is on the left "
            "when facing the screen.\n\n")
    for sname, rows, per_row, cats in SCREEN_LAYOUTS:
        out += f"## {sname} layout\n\n{len(rows)} rows ({rows[0]}-{rows[-1]}), {per_row} seats per row, {len(rows) * per_row} seats in all.\n\n"
        for cat, cat_rows in cats.items():
            out += f"- {cat.title()}: rows {', '.join(cat_rows)} ({len(cat_rows) * per_row} seats)\n"
        out += f"- Aisles after seat {per_row // 2}; wheelchair spaces at the end of row A (Screens 1 and 2).\n\n"
    return out


def build_docs(out_dir: Path = DOCS_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = {
        "gen_allergen_guide.md": allergen_guide(),
        "gen_festival_calendar.md": festival_doc(),
        "gen_movie_fares.md": movie_fares_doc(),
        "gen_films_now_showing.md": films_doc(),
        "gen_cinemas_guide.md": cinemas_doc(),
        "gen_bus_routes_and_fares.md": buses_doc(),
        "gen_restaurants_guide.md": restaurants_doc(),
        "gen_cinema_snacks.md": snacks_doc(),
        "gen_bus_stops.md": bus_stops_doc(),
        "gen_seat_maps.md": seat_maps_doc(),
    }
    for rid, name, *_ in RESTAURANTS:
        docs[f"gen_menu_{rid.split('-')[1]}.md"] = menu_doc(rid, name)
    paths = []
    for fname, text in docs.items():
        path = out_dir / fname
        path.write_text(text)
        paths.append(path)
    return paths
