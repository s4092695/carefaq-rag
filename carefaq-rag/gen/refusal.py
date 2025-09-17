EMERGENCY = ["chest pain","severe","bleeding","unconscious","difficulty breathing"]
CLINICAL  = ["dosage","diagnose","prescribe","side effect","should i take","treat"]
REFUSAL_TEXT = ("I can’t provide medical advice. For urgent symptoms call 000 or visit an emergency department. "
                "For non-urgent issues, please book with a GP.")
def needs_refusal(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in EMERGENCY+CLINICAL)
