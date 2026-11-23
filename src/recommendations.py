def classify_risk(probability):
    """Business-friendly churn risk bands."""
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"

def get_retention_recommendations(customer, probability):
    """
    Rule-based recommendations. These are business rules, not ML predictions.
    They are intentionally transparent so a student can explain them in a viva.
    """
    risk = classify_risk(probability)
    recommendations = []

    if risk == "High":
        recommendations.append("Prioritize the customer for a retention call or proactive outreach.")
    elif risk == "Medium":
        recommendations.append("Send a personalized retention offer and monitor the account.")
    else:
        recommendations.append("Maintain normal engagement and loyalty communication.")

    if customer.get("Contract") == "Month-to-month":
        recommendations.append("Offer a suitable longer-term contract with a transparent incentive.")

    if float(customer.get("MonthlyCharges", 0)) >= 80:
        recommendations.append("Review the customer's plan and offer a lower-cost or value-added package.")

    if customer.get("PaymentMethod") == "Electronic check":
        recommendations.append("Encourage automatic payment options and explain their convenience.")

    if customer.get("InternetService") == "Fiber optic":
        recommendations.append("Check service quality and technical support satisfaction.")

    if customer.get("TechSupport") == "No":
        recommendations.append("Offer technical support or a support-inclusive plan.")

    if customer.get("OnlineSecurity") == "No":
        recommendations.append("Consider a bundled security add-on or introductory security offer.")

    if int(customer.get("tenure", 0)) <= 6:
        recommendations.append("Use an early-tenure onboarding program to strengthen engagement.")

    return risk, recommendations
