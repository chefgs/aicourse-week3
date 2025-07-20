from langchain_community.document_loaders.text import TextLoader

story = TextLoader("story.txt", encoding="utf-8")

story.load()

