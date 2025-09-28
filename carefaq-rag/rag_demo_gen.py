from gen.lc_rag import answer_lc

qs = [
  "Do you bulk bill pensioners?",
  "What are the clinic opening hours?",
  "Do you offer dental services?"
]

for q in qs:
    out = answer_lc(q, k=5)
    print("Q:", q)
    print("A:", out.get("answer") or "<no answer>")
    print("Sources:", out.get("sources", []))
    print("-"*50)
