from __future__ import annotations

import dataclasses
import os
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> List[str]:
    """
    Tokenize a bAbI sentence/question.

    - Lowercase
    - Split punctuation as separate tokens
    """
    return _TOKEN_RE.findall(text.lower())


def download_and_extract_babi(
    root_dir: str | os.PathLike,
    *,
    url: str = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz",
    extracted_dirname: str = "babi",
    force_download: bool = False,
) -> Path:
    """
    Download and extract the bAbI QA dataset (v1.1) into root_dir/extracted_dirname.

    Returns the extracted directory path.
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    out_dir = root / extracted_dirname
    if out_dir.exists() and any(out_dir.iterdir()) and not force_download:
        return out_dir

    tgz_path = root / "babi_tasks_1-20_v1-2.tar.gz"
    if force_download and tgz_path.exists():
        tgz_path.unlink()

    if not tgz_path.exists():
        print(f"Downloading bAbI dataset from {url} ...")
        urllib.request.urlretrieve(url, tgz_path)

    if out_dir.exists():
        # Clean partial extraction
        for p in out_dir.glob("**/*"):
            if p.is_file():
                p.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tgz_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=out_dir)

    return out_dir


def _find_task_files(
    babi_root: Path,
    *,
    task_id: int,
    subset: str = "en-10k",
) -> Tuple[Path, Optional[Path], Path]:
    """
    Return (train, valid, test) file paths for a given task id.
    """
    qa_dir = babi_root / "tasks_1-20_v1-2" / "en" / subset
    if not qa_dir.exists():
        # Some tarballs contain tasks_1-20_v1-2/en-10k directly
        qa_dir = babi_root / "tasks_1-20_v1-2" / subset
    if not qa_dir.exists():
        raise FileNotFoundError(f"Could not find bAbI QA directory under {babi_root}")

    # filenames look like:
    # qa1_single-supporting-fact_train.txt
    # qa1_single-supporting-fact_valid.txt
    # qa1_single-supporting-fact_test.txt
    pattern = f"qa{task_id}_*"
    matches = sorted(qa_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} in {qa_dir}")

    train = next((p for p in matches if p.name.endswith("_train.txt")), None)
    # Some bAbI distributions (notably en-10k) do not include a *_valid.txt split.
    valid = next((p for p in matches if p.name.endswith("_valid.txt")), None)
    test = next((p for p in matches if p.name.endswith("_test.txt")), None)
    if train is None or test is None:
        raise FileNotFoundError(f"Missing train/test for task {task_id} in {qa_dir}")

    return train, valid, test


@dataclasses.dataclass(frozen=True)
class QAExample:
    story: List[List[str]]  # list of sentences, each tokenized
    question: List[str]
    answer: str
    supporting: Optional[List[int]] = None  # indices of supporting facts (unused for weak supervision)


def read_babi_examples(path: str | os.PathLike) -> List[QAExample]:
    """
    Parse a bAbI QA file into QAExample objects.
    """
    examples: List[QAExample] = []

    story: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue

            idx_str, rest = raw.split(" ", 1)
            idx = int(idx_str)
            if idx == 1:
                story = []

            if "\t" in rest:
                # question line: question \t answer \t supporting_facts
                q, a, sup = rest.split("\t")
                q_tokens = tokenize(q.rstrip("?"))
                supporting = [int(x) - 1 for x in sup.split()] if sup else None
                examples.append(QAExample(story=list(story), question=q_tokens, answer=a.strip(), supporting=supporting))
            else:
                # statement line
                sent = rest.rstrip(".")
                story.append(tokenize(sent))

    return examples


def load_babi_tasks(
    babi_root: str | os.PathLike,
    task_ids: Sequence[int],
    *,
    subset: str = "en-10k",
) -> Tuple[List[QAExample], List[QAExample], List[QAExample]]:
    """
    Load train/valid/test examples for multiple tasks and concatenate them.
    """
    train_all: List[QAExample] = []
    valid_all: List[QAExample] = []
    test_all: List[QAExample] = []

    root = Path(babi_root)
    for tid in task_ids:
        train_path, valid_path, test_path = _find_task_files(root, task_id=tid, subset=subset)
        train_all.extend(read_babi_examples(train_path))
        if valid_path is not None:
            valid_all.extend(read_babi_examples(valid_path))
        test_all.extend(read_babi_examples(test_path))

    return train_all, valid_all, test_all


def build_vocab(
    examples: Iterable[QAExample],
    *,
    min_freq: int = 1,
    add_pad: bool = True,
    add_unk: bool = True,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build a token vocabulary from a stream of QA examples.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for ex in examples:
        for sent in ex.story:
            counter.update(sent)
        counter.update(ex.question)
        counter.update([ex.answer])

    tokens: List[str] = []
    if add_pad:
        tokens.append("<PAD>")
    if add_unk:
        tokens.append("<UNK>")

    for tok, c in counter.most_common():
        if c >= min_freq and tok not in tokens:
            tokens.append(tok)

    stoi = {t: i for i, t in enumerate(tokens)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def _pad_or_trunc(seq: Sequence[int], length: int, pad_id: int) -> List[int]:
    if len(seq) >= length:
        return list(seq[:length])
    return list(seq) + [pad_id] * (length - len(seq))


def vectorize_examples(
    examples: Sequence[QAExample],
    stoi: Dict[str, int],
    *,
    memory_size: int,
    sentence_size: int,
    question_size: int,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
    reverse_story: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorize examples into numpy arrays suitable for PyTorch.

    Returns:
      - memories: [N, memory_size, sentence_size]
      - questions: [N, question_size]
      - answers: [N]
    """
    pad_id = stoi[pad_token]
    unk_id = stoi[unk_token]

    N = len(examples)
    memories = np.full((N, memory_size, sentence_size), pad_id, dtype=np.int64)
    questions = np.full((N, question_size), pad_id, dtype=np.int64)
    answers = np.zeros((N,), dtype=np.int64)

    for n, ex in enumerate(examples):
        story = ex.story[-memory_size:]
        if reverse_story:
            story = list(reversed(story))

        for i, sent in enumerate(story):
            sent_ids = [stoi.get(tok, unk_id) for tok in sent]
            memories[n, i, :] = np.array(_pad_or_trunc(sent_ids, sentence_size, pad_id), dtype=np.int64)

        q_ids = [stoi.get(tok, unk_id) for tok in ex.question]
        questions[n, :] = np.array(_pad_or_trunc(q_ids, question_size, pad_id), dtype=np.int64)

        answers[n] = stoi.get(ex.answer.lower(), unk_id)

    return memories, questions, answers


def infer_max_sizes(
    examples: Sequence[QAExample],
    *,
    memory_cap: int = 50,
    question_cap: int = 20,
    sentence_cap: int = 20,
) -> Tuple[int, int, int]:
    """
    Infer reasonable tensor sizes from examples with caps.
    """
    max_story = 1
    max_sent = 1
    max_q = 1
    for ex in examples:
        max_story = max(max_story, len(ex.story))
        for sent in ex.story:
            max_sent = max(max_sent, len(sent))
        max_q = max(max_q, len(ex.question))

    return min(max_story, memory_cap), min(max_sent, sentence_cap), min(max_q, question_cap)

