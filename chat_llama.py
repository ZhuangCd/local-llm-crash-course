from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q2_K.gguf"
)


def get_prompt(instruction: str) -> str:
    system = "You are an Ai assistant that gives helpful anwers. You answer in short and concise way"
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print(prompt)
    return prompt


question = "Which city is the capital of India?"

for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)

print()
