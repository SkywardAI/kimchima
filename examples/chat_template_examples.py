from kimchima import (
    ChatTemplateFactory, 
    TokenizerFactory
)


pretrained_model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
messages =[{"role": "user", "content": "Hello, how are you?"},
           {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
           {"role": "user", "content": "I'd like to show off how chat templating works!"}]

non_tokenized_prompt =  ChatTemplateFactory.prompt_generation(
                model=pretrained_model_name_or_path,
                messages=messages,
                tokenize=False
                )
print(non_tokenized_prompt)

tokenized_prompt =  ChatTemplateFactory.prompt_generation(
                model=pretrained_model_name_or_path,
                messages=messages,
                tokenize=True
                )

print(tokenized_prompt)
tokenizer = TokenizerFactory.auto_tokenizer(pretrained_model_name_or_path=pretrained_model_name_or_path)
for word in tokenized_prompt:
    print(tokenizer.decode(word))