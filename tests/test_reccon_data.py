from __future__ import annotations

import json

from et_hae_reccon.reccon.data import load_reccon_qa


def test_load_reccon_qa_squad_style(tmp_path) -> None:
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps(
            [
                {
                    "context": "I won the prize ! Wow !",
                    "qas": [
                        {
                            "id": "x1",
                            "question": "What is the cause?",
                            "is_impossible": False,
                            "answers": [{"text": "won the prize", "answer_start": 2}],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    examples = load_reccon_qa(path)
    assert len(examples) == 1
    assert examples[0].example_id == "x1"
    assert examples[0].answers[0].text == "won the prize"
