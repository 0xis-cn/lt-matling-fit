from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document

from langchain_text_splitters import MarkdownTextSplitter
from typing import Optional, Any
from datetime import date

import dotenv
dotenv.load_dotenv()

from os import getenv
import os.path

vector_store = Milvus(
    connection_args={'uri': getenv("MILVUS_URI")},
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)


import subprocess
from pathlib import Path
import logging
import re


def current_version():
    repo_path = Path("./0xis-cn")
    return subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True
    ).stdout.strip()


repo_path = Path("./0xis-cn")
all_documents = []


def document_from_filepath(md_file) -> Document:
    relative_path = str(md_file.relative_to(repo_path))
    metadata = {'source': relative_path}
    if (match := re.match(r"content(?P<prefix>.+/)((?P<date>\d{4}-\d{2}-\d{2})-)?(?P<disp>[-\w]+)(\.(?P<lang>[-\w]+))?\.md", relative_path)):
        metadata["uri"] = ("/" + kasi if (kasi := match.group("lang")) else "") + (match.group("prefix") or "") + match.group('disp')
        if match.group('date'):
            metadata["time"] = date.fromisoformat(match.group('date'))
    with open(md_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        try:
            split = lines.index(lines[0], 1)
        except:
            split = 0
        content = "\n".join(lines[split+1:]).strip()
        for key in ['title', 'tags', 'permalink']:
            for line in lines[:split]:
                line = line.strip()
                if line.startswith(key):
                    match = re.match(key + r'\s?(:\s|=\s")(?P<val>.+)"?', line)
                    try:
                        metadata[key] = match.group("val")
                    except:
                        metadata[key] = line
    return Document(
        page_content=content,
        metadata=metadata,
    )


def updated_documents() -> list[Document]:
    repository_uri = "https://github.com/0xis-cn/0xis-cn.git"
    updated_md_files = []

    if not repo_path.exists():
        # select all .md files
        subprocess.run(
            ["git", "clone", repository_uri, str(repo_path)],
            check=True,
            capture_output=True,
            text=True
        )
        updated_md_files = list(repo_path.rglob("*.md"))
    else:
        # pull and select updated .md files
        old_head = current_version()

        subprocess.run(
            ["git", "-C", str(repo_path), "pull"],
            check=True,
            capture_output=True,
            text=True
        )

        new_head = current_version()

        if old_head == new_head:
            return []

        # --name-only: show file name only; --diff-filter=AM：Add & Modify
        diff_output = subprocess.run(
            ["git", "-C", str(repo_path), "diff", "--name-only", "--diff-filter=AM", old_head, new_head],
            check=True,
            capture_output=True,
            text=True
        )

        for file_path in diff_output.stdout.splitlines():
            if file_path.endswith(".md"):   # temporarily not prone so ignored
                full_path = repo_path / file_path
                if full_path.exists():  # not deleted
                    updated_md_files.append(full_path)

    documents = []
    for md_file in updated_md_files:
        try:
            documents.append(document_from_filepath(md_file))
        except Exception as e:
            logging.error(f"An error when processing {md_file}")
            logging.exception(e)

    return documents


from whoosh.fields import Schema, TEXT, ID
schema = Schema(path=ID(stored=True), content=TEXT(stored=True))
from whoosh.index import create_in, open_dir
if os.path.exists("index"):
    idx = open_dir(dirname="index", schema=schema)
else:
    os.mkdir("index")
    idx = create_in(dirname="index", schema=schema)

def create_in(documents: list[Document]):
    writer = idx.writer()
    for document in documents:
        writer.add_document(
            content=document.page_content,
            path=document.metadata.get('uri')
        )
    writer.commit()


def keyword_search(query: str):
    from whoosh import qparser
    parser = qparser.QueryParser('content', schema=schema)
    query = parser.parse(query)
    with idx.searcher() as searcher:
        results = searcher.search(query, limit=10)
        for i in results:
            print(i)
        return results


def semantic_search(query: str):
    return vector_store.similarity_search(query)


def list_archives(
    earliest_date: Optional[date] = None,
    latest_date: Optional[date] = None,
    tags: Optional[list[str]] = None,
) -> list[Document]:
    """
    Thin wrapper over worker.list_archives. Do not reimplement.
    All params are optional and forwarded as-is.
    """
    documents = all_documents
    if earliest_date:
        documents = (i for i in documents if i.metadata.get('time') and i.metadata.get('time') >= earliest_date)
    if latest_date:
        documents = (i for i in documents if i.metadata.get('time') and i.metadata.get('time') <= latest_date)
    if tags:
        documents = (i for i in documents if i.metadata.get('tags') and any((tag in i.metadata.get('tags')) for tag in tags))
    return list(documents)


if __name__ == "__main__":
    documents = updated_documents()
    version = current_version()
    ids = vector_store.add_documents(documents, ids=[f"{version}-{i.metadata['source']}" for i in documents])
    logging.info("\n".join(ids))
    create_in(documents)

    print(semantic_search("Remniscence"))
    print(keyword_search("Lanzhou"))
else:
    for md_file in repo_path.rglob("*.md"):
        try:
            all_documents.append(document_from_filepath(md_file))
        except Exception as e:
            logging.error(f"An error when processing {md_file}")
            logging.exception(e)
    # for kasi in all_documents:
    #     print(kasi.metadata)