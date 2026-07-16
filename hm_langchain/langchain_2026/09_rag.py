from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


def demo_01():
    text_loader = TextLoader(file_path='./asset/load/01-langchain-utf-8.txt', encoding='utf-8')
    docs = text_loader.load()
    print(docs)


def demo_02():
    loader = CSVLoader(file_path='./asset/load/02-load.csv')
    data = loader.load()
    print(data)
    print(type(data))
    print(type(data[0]))
    print(len(data))
    print(data[0].page_content)


if __name__ == '__main__':
    demo_02()
