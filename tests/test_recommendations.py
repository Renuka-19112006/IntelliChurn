from src.recommendations import classify_risk, get_retention_recommendations


def test_low_risk():
    assert classify_risk(0.20) == "Low"


def test_medium_risk():
    assert classify_risk(0.45) == "Medium"


def test_high_risk():
    assert classify_risk(0.80) == "High"


def test_risk_boundaries():
    assert classify_risk(0.30) == "Medium"
    assert classify_risk(0.60) == "High"


def test_high_risk_recommendations():
    customer = {"Contract": "Month-to-month"}
    risk, recommendations = get_retention_recommendations(customer, 0.80)

    assert risk == "High"
    assert any("proactive outreach" in item for item in recommendations)
    assert any("longer-term contract" in item for item in recommendations)


def test_low_risk_recommendations():
    customer = {}
    risk, recommendations = get_retention_recommendations(customer, 0.20)

    assert risk == "Low"
    assert any("normal engagement" in item for item in recommendations)
