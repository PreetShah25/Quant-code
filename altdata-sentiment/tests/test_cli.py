from altdata_sentiment.cli import score_text

def test_score_text_basic():
    assert score_text("record profit, guidance strong")>0.5
    assert score_text("miss plunges on weak guidance")<-1.0
