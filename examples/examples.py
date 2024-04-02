from kimchima import Auto, get_device, get_capability

model = Auto(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")

# computing embeddings for single text
embeddings = model.get_embeddings(text="Melbourne")
print(embeddings.shape)

# computing embeddings for multiple texts
embeddings = model.get_embeddings(text=["Melbourne", "Sydney"])
print(embeddings.shape)

# Checking the device: GPU, mps and CPU
device = get_device()
print(device)


# get capability of GPU(Nvidia)
capability = get_capability()
print(capability)