import time
from transformers import pipeline, set_seed

set_seed(42)

chatbot = pipeline("text2text-generation", model="google/flan-t5-base")

print("🩺 Medical Chatbot (Educational Use Only)")
print("Type 'exit' to quit.\n")

examples = """
Q: I have a cold and a fever. What should I do?
A: You should rest, stay hydrated, and take over-the-counter fever medicine like paracetamol.

Q: I have a skin allergy. What should I do?
A: Avoid any known allergens, apply a soothing lotion like calamine, and consult a dermatologist if it gets worse.
"""

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Take care! 👋")
        break

    if not user_input:
        print("Chatbot: Please enter your symptoms.")
        continue

    if "eye infection" in user_input.lower():
        print("Chatbot: You may be experiencing conjunctivitis. Keep your eyes clean, avoid rubbing them, and consult a doctor if symptoms persist.")
        continue

    prompt = f"{examples}\nQ: {user_input}\nA:"
    response = chatbot(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2
    )

    print("Chatbot:", response[0]["generated_text"].strip())
    time.sleep(1)
