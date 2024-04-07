from kimchima import (
    ModelFactory, 
    TokenizerFactory,
    StreamerFactory,
    EmbeddingsFactory,
    QuantizationFactory,
    PipelinesFactory,
    Devices
)


pretrained_model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"

model = ModelFactory.auto_model(pretrained_model_name_or_path=pretrained_model_name_or_path)
tokenizer= TokenizerFactory.auto_tokenizer(pretrained_model_name_or_path=pretrained_model_name_or_path)

# computing embeddings for single text
embeddings = EmbeddingsFactory.get_text_embeddings(
    model=model,
    tokenizer=tokenizer,
    prompt='Melbourne',
    device='cpu'
)
print(embeddings.shape)

# computing embeddings for multiple texts
embeddings = EmbeddingsFactory.get_text_embeddings(
    model=model,
    tokenizer=tokenizer,
    prompt=['Melbourne', 'Sydney'],
    device='cpu'
)
print(embeddings.shape)

# Checking the device: GPU, mps and CPU
device = Devices.get_device()
print(device)


# get capability of GPU(Nvidia)
capability = Devices.get_capability()
print(capability)


# streamer
model= ModelFactory.auto_model_for_causal_lm(pretrained_model_name_or_path="gpt2")
tokenizer= TokenizerFactory.auto_tokenizer(pretrained_model_name_or_path="gpt2")
streamer= StreamerFactory.text_streamer(tokenizer=tokenizer, skip_prompt=False, skip_prompt_tokens=False)


pipe=PipelinesFactory.text_generation(
    model=model, 
    tokenizer=tokenizer, 
    text_streamer=streamer
    )

pipe("Melbourne is the capital of Victoria")