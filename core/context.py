from typing import List, Dict

def build_context_digest(episodes: List[Dict], facts: List[Dict], trait_names: List[str], budgets):
    ep_lines = [f"- {e['summary']}" for e in episodes[:budgets['episodes']]]
    fact_lines = [f"- {f['claim']}" for f in facts[:budgets['facts']]]
    trait_lines = [f"- {t}" for t in trait_names[:budgets['traits']]]

    digest = []
    if trait_lines:
        digest.append("Active traits:")
        digest.extend(trait_lines)
        digest.append("")
    if ep_lines:
        digest.append("Recent relevant episodes:")
        digest.extend(ep_lines)
        digest.append("")
    if fact_lines:
        digest.append("Known facts:")
        digest.extend(fact_lines)
        digest.append("")
    return "\n".join(digest).strip()
