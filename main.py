from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
import dotenv

from langchain_text_splitters import MarkdownTextSplitter

dotenv.load_dotenv()

from os import getenv

vector_store = Milvus(
    connection_args={'uri': getenv("MILVUS_URI")},
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)


import subprocess
from pathlib import Path
import logging

def updated_documents() -> list[Document]:
    repository_uri = "https://github.com/0xis-cn/0xis-cn.git"
    repository_path = "./0xis-cn"
    repo_path = Path(repository_path)
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
        old_head = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True
        ).stdout.strip()

        subprocess.run(
            ["git", "-C", str(repo_path), "pull"],
            check=True,
            capture_output=True,
            text=True
        )

        new_head = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True
        ).stdout.strip()

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
            if file_path.endswith(".md"):
                full_path = repo_path / file_path
                if full_path.exists():  # not deleted
                    updated_md_files.append(full_path)

    documents = []
    for md_file in updated_md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            relative_path = str(md_file.relative_to(repo_path))
            documents.append(Document(
                page_content=content,
                metadata={"source": relative_path}  # TODO: more metadata
            ))
        except Exception as e:
            logging.error(f"An error when processing {md_file}")
            logging.exception(e)

    return documents