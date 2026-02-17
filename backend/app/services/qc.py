RED_FLAG_TERMS = [
    "无法判断",
    "不确定",
    "建议结合临床",
]

REQUIRED_KEYS = [
    "病灶",
    "变化",
]


def evaluate_findings(findings: str) -> tuple[str, list[str]]:
    issues: list[str] = []
    text = findings.strip()

    if len(text) < 20:
        issues.append("findings_too_short")

    for key in REQUIRED_KEYS:
        if key not in text:
            issues.append(f"missing_required_token:{key}")

    for term in RED_FLAG_TERMS:
        if term in text:
            issues.append(f"red_flag:{term}")

    if any(item.startswith("red_flag:") for item in issues):
        return "review_required", issues
    if issues:
        return "blocked", issues
    return "pass", []
