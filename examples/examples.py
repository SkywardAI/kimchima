from kimchima import (
    ModelFactory, 
    TokenizerFactory,
    EmbeddingsFactory
)


from kimchima import(
    get_device, 
    get_capability)

pretrained_model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"

model = ModelFactory.auto_model(pretrained_model_name_or_path=pretrained_model_name_or_path)
tokenizer= TokenizerFactory.auto_tokenizer(pretrained_model_name_or_path=pretrained_model_name_or_path)

# computing embeddings for single text
embeddings = EmbeddingsFactory.auto_embeddings(
    model=model,
    tokenizer=tokenizer,
    prompt='Melbourne',
    device='cpu'
)
print(embeddings.shape)

# computing embeddings for multiple texts
embeddings = EmbeddingsFactory.auto_embeddings(
    model=model,
    tokenizer=tokenizer,
    prompt=['Melbourne', 'Sydney'],
    device='cpu'
)
print(embeddings.shape)

# Checking the device: GPU, mps and CPU
device = get_device()
print(device)


# get capability of GPU(Nvidia)
capability = get_capability()
print(capability)