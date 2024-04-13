
from kimchima.utils import chat_summary

conversation_model="gpt2"
summarization_model="sshleifer/distilbart-cnn-12-6"
msg = "why Melbourne is a good place to travel?"
prompt = "Melbourne is often considered one of the most livable cities globally, offering a high quality of life."

res = chat_summary(
    conversation_model=conversation_model,
    summarization_model=summarization_model,
    messages=msg,
    prompt=prompt
    )

print(res)