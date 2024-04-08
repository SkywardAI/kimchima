from kimchima.pkg import DownloadHub


file_path=DownloadHub.download_specific_file(
    repo_id="microsoft/Mistral-7B-v0.1-onnx",
    filename="README.md",
    folder_name="examples/readme",
    revision="main"
)

print(file_path)

repo_path=DownloadHub.download_repo(
    repo_id="openai-community/gpt2",
    folder_name="examples/repo",
    revision="main"
)

print(repo_path)