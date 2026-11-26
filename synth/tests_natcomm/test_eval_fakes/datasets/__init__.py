"""Fake datasets.load_dataset for the aime_prep.py test: rows from the JSON file $FAKE_DATASETS_FIXTURE
({dataset name: [row, ...]}); every call (name, split, revision) is appended to $FAKE_DATASETS_LOG if set."""
import json, os


def load_dataset(name, split=None, revision=None, **kw):
    if os.environ.get("FAKE_DATASETS_LOG"):
        with open(os.environ["FAKE_DATASETS_LOG"], "a") as f:
            f.write(json.dumps(dict(name=name, split=split, revision=revision)) + "\n")
    with open(os.environ["FAKE_DATASETS_FIXTURE"]) as f:
        return list(json.load(f)[name])
