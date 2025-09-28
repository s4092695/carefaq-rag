# gen/unanswerable.py
EMERGENCY = ["chest pain","severe bleeding","unconscious","difficulty breathing","stroke","heart attack"]
CLINICAL  = ["dosage","diagnose","prescribe","side effect","should I take","treat","treatment","symptom"]

REFUSAL_TEXT = ("I can’t provide medical advice. For urgent symptoms call 000 or visit an emergency department. "
                "For non-urgent issues, please book with a GP.")

def needs_refusal(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in EMERGENCY+CLINICAL)

def low_confidence(scores, threshold=0.2):
    return (max(scores) if scores else 0.0) < threshold
