from kimchima import Auto, get_device

model = Auto(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.get_embeddings(text="Hello, world!")
print(embeddings)


device = get_device()
print(device)