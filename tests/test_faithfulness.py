from rag_knowledge_assistant.faithfulness import HeuristicJudge, calibrate, load_cases


def test_grounding_score_bounds():
    judge = HeuristicJudge()
    ctx = ["Cosine similarity measures the angle between two vectors."]
    # fully supported answer -> high grounding
    assert judge.score("Cosine similarity measures the angle between vectors.", ctx) >= 0.9
    # fabricated claim -> low grounding
    low = judge.score("Cosine similarity was invented by Euler in 1738 aboard a ship.", ctx)
    assert low < 0.5


def test_judge_agrees_with_human_labels():
    # the heuristic judge + threshold should reproduce the hand labels exactly
    cal = calibrate(load_cases(), HeuristicJudge())
    assert cal["n"] == 10
    assert cal["accuracy"] == 1.0


def test_empty_answer_is_trivially_grounded():
    assert HeuristicJudge().score("", ["anything"]) == 1.0
