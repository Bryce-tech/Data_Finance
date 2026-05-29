"""Rule-based business summaries for credit risk reviews."""


def generate_risk_summary(customer_info, risk_probability, top_factors, is_anomaly):
    """Return a concise French risk summary for one credit file.

    Parameters
    ----------
    customer_info:
        Mapping or pandas Series containing customer-level information.
    risk_probability:
        Estimated probability of a risky credit file.
    top_factors:
        Iterable of business-readable risk drivers.
    is_anomaly:
        Boolean flag from the anomaly detection module.
    """
    probability = float(risk_probability)
    factors = [str(factor) for factor in top_factors if str(factor).strip()]

    if probability >= 0.65:
        risk_level = "élevé"
        recommendation = "Une revue manuelle prioritaire est recommandée."
    elif probability >= 0.35:
        risk_level = "modéré"
        recommendation = "Une analyse complémentaire peut être utile."
    else:
        risk_level = "faible"
        recommendation = "Le dossier ne présente pas de signal majeur selon ce modèle."

    if factors:
        factor_text = "Les principaux facteurs sont " + ", ".join(factors[:3]) + "."
    else:
        factor_text = "Aucun facteur dominant ne ressort clairement."

    anomaly_text = (
        "Le dossier est aussi détecté comme atypique par le module d'anomalie."
        if is_anomaly
        else "Le dossier n'est pas identifié comme atypique par le module d'anomalie."
    )

    customer_id = customer_info.get("customer_id", "client sélectionné")

    return (
        f"Pour le dossier {customer_id}, le risque estimé est {risk_level} "
        f"avec une probabilité de {probability:.1%}. {factor_text} "
        f"{anomaly_text} {recommendation}"
    )
