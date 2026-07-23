"""
preprocess.py
-------------
Shared text utilities used by BOTH training and the Streamlit app.

Keeping these in one place is what fixes the "scattered" problem in the
original code: the app and the trainer now clean text in exactly the same
way, so what the model learns matches what it sees at prediction time.

It provides three things:
  1. clean_text()            -> light, safe normalisation
  2. detect_language()       -> 'urdu' | 'roman_urdu' | 'english'
  3. rule_based_scam_score() -> a 0..1 score + the signals that fired,
                                working in English, Roman Urdu and Urdu script
"""

import re

# ---------------------------------------------------------------------------
# 1. TEXT CLEANING
# ---------------------------------------------------------------------------
# NOTE: we deliberately keep numbers, links and symbols. In scam detection
# those ARE the signal (₨, http, OTP codes, phone numbers). We only lower-case
# and collapse whitespace.

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = _WHITESPACE.sub(" ", text)
    return text


# ---------------------------------------------------------------------------
# 2. LANGUAGE DETECTION
# ---------------------------------------------------------------------------
# Urdu (Arabic) script lives in these Unicode blocks.
_URDU_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

# Very common Roman-Urdu function words. If several appear, the message is
# almost certainly Urdu written with English letters.
_ROMAN_URDU_WORDS = {
    "hai", "hain", "nahi", "nhi", "kya", "kia", "main", "mein", "tum", "aap",
    "ap", "ka", "ki", "ko", "se", "ye", "yeh", "wo", "woh", "kar", "karo",
    "karein", "karen", "kiya", "acha", "theek", "thek", "bhai", "paisa",
    "paise", "rupay", "rupaye", "rupee", "inaam", "inam", "mubarak", "jeeta",
    "jeet", "jeeti", "gaye", "gaya", "bhej", "bhejo", "bhejein", "jaldi",
    "abhi", "muft", "lakh", "crore", "number", "account", "aur", "hum", "mera",
    "meri", "apka", "apki", "apna", "raha", "rahi", "diya", "milega", "milegi",
}


def detect_language(text: str) -> str:
    """Return 'urdu' (script), 'roman_urdu', or 'english'."""
    if not isinstance(text, str) or not text.strip():
        return "english"

    # Any real amount of Urdu script -> treat as Urdu.
    urdu_chars = len(_URDU_SCRIPT.findall(text))
    if urdu_chars >= 3:
        return "urdu"

    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return "english"
    hits = sum(1 for t in tokens if t in _ROMAN_URDU_WORDS)
    # A couple of Roman-Urdu markers, or a decent density, flips it.
    if hits >= 2 or (hits / len(tokens)) >= 0.20:
        return "roman_urdu"
    return "english"


# ---------------------------------------------------------------------------
# 3. RULE-BASED SCAM SIGNALS  (language-independent safety net)
# ---------------------------------------------------------------------------
# The ML model is trained mostly on ENGLISH spam. These rules give the system
# real teeth in Urdu / Roman Urdu, where there is little labelled training
# data. Each list is a "signal"; the more signals fire, the higher the score.

_KEYWORDS = {
    "prize_or_winning": [
        # English
        "congratulations", "congrats", "you won", "you have won", "winner",
        "you are selected", "lucky draw", "lottery", "claim your prize",
        "gift card", "reward", "cash prize",
        # Roman Urdu
        "mubarak", "inaam", "inam", "jeeta", "jeet gaye", "lucky", "scheme",
        # Urdu script
        "مبارک", "انعام", "جیت", "لاٹری",
    ],
    "urgency": [
        "urgent", "immediately", "act now", "expires", "expiry", "last chance",
        "within 24 hours", "hurry", "limited time",
        "jaldi", "abhi", "foran", "turant",
        "جلدی", "فوری", "ابھی",
    ],
    "credential_or_otp": [
        "otp", "one time password", "pin code", "verify your account",
        "verification code", "cvv", "password", "login", "confirm your",
        "account suspended", "account blocked", "account band",
        "code bhejein", "code bhejo", "pin bhejo", "verify karein",
        "اکاؤنٹ", "پاس ورڈ", "کوڈ",
    ],
    "money_request": [
        "send money", "transfer", "fee", "processing fee", "deposit",
        "easypaisa", "jazzcash", "bank account", "paisa bhejo", "paise bhejein",
        "rupay", "rupee", "lakh", "crore", "$", "₨", "rs.", "rs ",
        "پیسے", "رقم", "اکاؤنٹ نمبر",
    ],
    "suspicious_link": [
        "click here", "click the link", "click below", "tap here", "bit.ly",
        "tinyurl", "link par click", "click karein",
    ],
    "impersonation": [
        # commonly impersonated in Pakistani scams
        "benazir", "ehsaas", "bisp", "helpline", "govt scheme", "government",
        "prize bond", "telenor", "jazz", "zong", "ufone",
    ],
}

# Regex signals (structure, not words)
_URL_RE = re.compile(r"(https?://|www\.|bit\.ly|tinyurl|\b\w+\.(?:com|net|xyz|info|link|tk|ml)\b)")
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-]{8,}\d)")
_MONEY_RE = re.compile(r"(₨|rs\.?\s?\d|\$\s?\d|\d+\s?(lakh|crore|million|k\b))")


def rule_based_scam_score(text: str):
    """
    Return (score, signals) where:
      score   : float in 0..1  (higher = more scam-like)
      signals : list of human-readable reasons that fired
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, []

    low = text.lower()
    signals = []
    weight = 0.0

    for label, words in _KEYWORDS.items():
        if any(w in low for w in words):
            signals.append(label.replace("_", " "))
            weight += 1.0

    if _URL_RE.search(low):
        signals.append("suspicious link")
        weight += 1.0
    if _PHONE_RE.search(low):
        signals.append("phone number present")
        weight += 0.5
    if _MONEY_RE.search(low):
        signals.append("money amount mentioned")
        weight += 0.5

    # de-duplicate signal names while preserving order
    seen = set()
    signals = [s for s in signals if not (s in seen or seen.add(s))]

    # squash the weight into 0..1. ~3 strong signals -> already very high.
    score = 1 - (0.5 ** weight)  # 0 sig=0, 1=0.5, 2=0.75, 3=0.875 ...
    return round(score, 3), signals


if __name__ == "__main__":
    # quick self-test
    samples = [
        "Congratulations! You won a $1000 gift card. Click link to claim now!",
        "Mubarak ho! Aap ne 50 lakh ka inaam jeeta hai. OTP code bhejein.",
        "مبارک ہو آپ نے انعام جیتا ہے",
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
    ]
    for s in samples:
        lang = detect_language(s)
        score, sig = rule_based_scam_score(s)
        print(f"[{lang:10}] score={score:<5} signals={sig}\n   {s}\n")
