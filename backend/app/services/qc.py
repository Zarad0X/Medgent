from typing import TypedDict


class QCIssues(TypedDict):
    format: list[str]
    completeness: list[str]
    safety: list[str]


SOFT_SAFETY_TERMS = [
    "无法判断",
    "不确定",
    "建议结合临床",
]

# Only these terms can trigger blocked.
HARD_DANGER_TERMS = [
    "危及生命",
    "张力性气胸",
    "活动性出血",
    "主动脉夹层",
    "脑疝",
]

REQUIRED_SYNONYMS = {
    "lesion_mention": ["病灶", "结节", "病变", "肿块", "占位", "阴影"],
    "change_mention": ["变化", "进展", "稳定", "增大", "缩小", "较前", "无明显变化"],
}


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def flatten_qc_issues(issues: QCIssues) -> list[str]:
    flat: list[str] = []
    for category in ("format", "completeness", "safety"):
        for issue in issues[category]:
            flat.append(f"{category}:{issue}")
    return flat


def evaluate_findings(findings: str) -> tuple[str, QCIssues]:
    text = findings.strip()
    issues: QCIssues = {"format": [], "completeness": [], "safety": []}

    if len(text) < 20:
        issues["format"].append("findings_too_short")

    if not _contains_any(text, REQUIRED_SYNONYMS["lesion_mention"]):
        issues["completeness"].append("missing_lesion_mention")
    if not _contains_any(text, REQUIRED_SYNONYMS["change_mention"]):
        issues["completeness"].append("missing_change_mention")

    for term in SOFT_SAFETY_TERMS:
        if term in text:
            issues["safety"].append(f"soft_flag:{term}")
    for term in HARD_DANGER_TERMS:
        if term in text:
            issues["safety"].append(f"danger_flag:{term}")

    if any(item.startswith("danger_flag:") for item in issues["safety"]):
        return "blocked", issues
    if any(issues.values()):
        return "review_required", issues
    return "pass", issues
