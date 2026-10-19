"""Indian names, areas and phone numbers for synthetic callers (all fictional)."""

from __future__ import annotations

import random

MALE_FIRST = [
    "Aarav", "Aditya", "Ajay", "Akash", "Amit", "Anand", "Anil", "Arjun", "Arvind", "Ashok",
    "Bhaskar", "Chandrasekhar", "Deepak", "Dinesh", "Ganesh", "Gopal", "Harish", "Imran", "Jagadeesh",
    "Karthik", "Kiran", "Krishna", "Mahesh", "Manoj", "Mohammed", "Murali", "Naveen", "Nikhil",
    "Pavan", "Prakash", "Pranav", "Rahul", "Rajesh", "Rakesh", "Ramesh", "Ravi", "Rohit", "Sai",
    "Sandeep", "Sanjay", "Santosh", "Siddharth", "Srinivas", "Sudheer", "Suresh", "Syed", "Tarun",
    "Uday", "Varun", "Venkatesh", "Vijay", "Vikram", "Vinod", "Vishal", "Yash",
]
FEMALE_FIRST = [
    "Aishwarya", "Ananya", "Anjali", "Anusha", "Bhavana", "Deepika", "Divya", "Fatima", "Gayatri",
    "Harini", "Ishita", "Jyothi", "Kavya", "Keerthi", "Lakshmi", "Madhuri", "Meena", "Meghana",
    "Nandini", "Neha", "Nikitha", "Padma", "Pooja", "Priya", "Radhika", "Ramya", "Rekha", "Roshni",
    "Sahithi", "Sana", "Shalini", "Shruti", "Sindhu", "Sneha", "Sowmya", "Sravani", "Swathi",
    "Tejaswini", "Usha", "Vaishnavi", "Varsha", "Vidya", "Zoya",
]
SURNAMES = [
    "Reddy", "Rao", "Naidu", "Chowdary", "Sharma", "Verma", "Gupta", "Agarwal", "Iyer", "Iyengar",
    "Nair", "Menon", "Pillai", "Kulkarni", "Deshpande", "Patil", "Joshi", "Mishra", "Pandey",
    "Tiwari", "Chaturvedi", "Tyagi", "Jacob", "Thomas", "D'Souza", "Fernandes", "Khan", "Ahmed",
    "Hussain", "Siddiqui", "Banerjee", "Chatterjee", "Mukherjee", "Das", "Bose", "Singh", "Kaur",
    "Malhotra", "Kapoor", "Mehta", "Shah", "Patel", "Varma", "Raju", "Goud", "Yadav", "Kavuri",
    "Bhat", "Hegde", "Shetty", "Subramanian", "Krishnan", "Venkataraman",
]
HYDERABAD_AREAS = [
    "Kukatpally", "Gachibowli", "Madhapur", "Kondapur", "Ameerpet", "Begumpet", "Secunderabad",
    "Dilsukhnagar", "LB Nagar", "Mehdipatnam", "Tolichowki", "Banjara Hills", "Jubilee Hills",
    "Miyapur", "Uppal", "Habsiguda", "Manikonda", "Attapur", "Kompally", "Nizampet",
]


def person(rng: random.Random) -> tuple[str, str]:
    """Return (full name, gender code 'M'/'F')."""
    if rng.random() < 0.5:
        return f"{rng.choice(MALE_FIRST)} {rng.choice(SURNAMES)}", "M"
    return f"{rng.choice(FEMALE_FIRST)} {rng.choice(SURNAMES)}", "F"


def phone(rng: random.Random, taken: set[str]) -> str:
    while True:
        number = rng.choice("6789") + "".join(rng.choice("0123456789") for _ in range(9))
        if number not in taken:
            taken.add(number)
            return number
