import json
import re
from typing import Any, Dict, List


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"No JSON object found in model output: {text[:500]}")


def clamp_age(value: Any) -> int:
    try:
        age = int(value)
    except (TypeError, ValueError):
        age = 7
    return min(10, max(5, age))


def fallback_brief(user_request: str) -> Dict[str, Any]:
    return {
        "premise": user_request,
        "main_characters": [],
        "target_age": 7,
        "vibe": "cozy and gentle",
        "category": "cozy",
        "theme_or_lesson": "kindness and curiosity",
        "required_details": [],
        "avoid": ["scary intensity", "violence", "unsafe behavior"],
    }


_ALLOWED_CATEGORIES = {"cozy", "funny", "adventurous", "spooky", "grief", "educational", "magical"}


def normalize_category(value: Any) -> str:
    """Pin the brief's category to one of the allowed values.

    The intake LLM occasionally returns a free-form word in the category field
    (e.g. "dreamy", "calming"); map anything unknown back to "cozy" so the
    generator always has matching guidance from CATEGORY_GUIDANCE.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ALLOWED_CATEGORIES:
            return normalized
    return "cozy"


def normalize_required_details(details: Any, user_request: str) -> List[str]:
    if isinstance(details, str):
        details = [details]
    if not isinstance(details, list):
        return []

    request_words = set(re.findall(r"[a-zA-Z0-9]+", user_request.lower()))
    filtered = []
    unsafe_detail_terms = {
        "eat",
        "eating",
        "drink",
        "drinking",
        "sip",
        "sipping",
        "medicine",
        "potion",
        "cure",
        "launch",
        "fly",
        "flying",
        "climb",
        "climbing",
    }
    for detail in details:
        if not isinstance(detail, str):
            continue
        detail_words = {word for word in re.findall(r"[a-zA-Z0-9]+", detail.lower()) if len(word) > 2}
        if detail_words & unsafe_detail_terms:
            continue
        if detail_words & request_words:
            filtered.append(detail)
    return filtered


BRANDED_TERM_REPLACEMENTS = {
    "Elsa from Frozen": "a kind snow queen",
    "Elsa": "a kind snow queen",
    "Frozen": "a snowy kingdom",
    "Spider-Man": "a friendly web-slinging hero",
    "Spiderman": "a friendly web-slinging hero",
    "Peter Parker": "the hero",
    "Tony Stark": "the inventor",
    "Hogwarts School of Witchcraft and Wizardry": "a cozy school of magic",
    "Hogwarts": "a cozy school of magic",
    "Harry Potter": "a young wizard",
    "Hermione": "the clever friend",
}

# Names that should never appear in the final story body. The brief already gets
# `replace_branded_terms`, but gpt-3.5-turbo associates tropes with the original
# brands ("snow queen" -> Elsa, "web-slinger" -> Peter Parker) and re-adds the
# real names. We catch that here.
BRANDED_NAMES_IN_STORY = (
    # Specific character names + franchise place-names that are unambiguous brand
    # references. We deliberately do NOT include common English words like
    # "frozen" (matches "frozen in time", "frozen by fear") — the unique tokens
    # below are sufficient because Elsa, Anna+Olaf, Hogwarts, Peter Parker etc.
    # are themselves unambiguous franchise markers.
    "elsa",
    "anna and elsa",
    "peter parker",
    "tony stark",
    "spider-man",
    "spiderman",
    "harry potter",
    "hermione",
    "ron weasley",
    "hogwarts",
)


def replace_branded_terms(value: Any) -> Any:
    if isinstance(value, str):
        updated = value
        for source, replacement in BRANDED_TERM_REPLACEMENTS.items():
            updated = updated.replace(source, replacement)
        return updated
    if isinstance(value, list):
        return [replace_branded_terms(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_branded_terms(item) for key, item in value.items()}
    return value


_AGE_HINT_PATTERNS = (
    re.compile(r"\bfor\s+a?\s*(\d{1,2})[\s-]*year[\s-]*old", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})[\s-]*year[\s-]*old", re.IGNORECASE),
    re.compile(r"\bage\s+(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})\s+years?\s+old\b", re.IGNORECASE),
)


def extract_age_hint(user_request: str) -> int | None:
    """Extract a numeric age from the user's free-text request.

    Returning the raw integer (not clamped) lets the caller decide whether the
    age needs clamping or whether to ignore it. The intake LLM sometimes treats
    out-of-range ages (e.g. "13-year-old") as 'no age specified' and falls back
    to the default 7, which is the wrong behavior — the user did supply an age,
    it just needs to be clamped to 5–10. This regex pre-pass makes that
    deterministic.
    """
    for pattern in _AGE_HINT_PATTERNS:
        match = pattern.search(user_request)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def normalize_brief(raw: Dict[str, Any], user_request: str) -> Dict[str, Any]:
    brief = fallback_brief(user_request)
    brief.update({key: value for key, value in raw.items() if value not in (None, "")})
    brief = replace_branded_terms(brief)

    # Prefer the user-stated age (clamped) over whatever the intake LLM returned.
    # This rescues out-of-range ages from a silent fall-through to the default 7.
    user_age = extract_age_hint(user_request)
    if user_age is not None:
        brief["target_age"] = clamp_age(user_age)
    else:
        brief["target_age"] = clamp_age(brief.get("target_age", 7))

    if isinstance(brief.get("main_characters"), str):
        brief["main_characters"] = [brief["main_characters"]]
    if isinstance(brief.get("avoid"), str):
        brief["avoid"] = [brief["avoid"]]
    brief["required_details"] = normalize_required_details(brief.get("required_details"), user_request)
    brief["category"] = normalize_category(brief.get("category"))
    return brief


def fallback_critique(reason: str) -> Dict[str, Any]:
    """When critic output cannot be parsed, fail closed.

    A parse failure means we have no LLM signal about safety. The conservative play is
    to mark every hard check False so the pipeline triggers a revision (or refuses)
    rather than silently shipping an unaudited story.
    """
    return {
        "passes": False,
        "hard_checks": {
            "safe_for_children": False,
            "age_appropriate": False,
            "follows_request": False,
        },
        "soft_scores": {
            "bedtime_tone": 0,
            "vocabulary_fit": 0,
            "story_arc": 0,
            "read_aloud_quality": 0,
        },
        "strengths": [],
        "revision_suggestions": [f"Critic output could not be parsed: {reason}. Treat the story as unverified and revise conservatively."],
    }


def critique_passes(critique: Dict[str, Any]) -> bool:
    hard_checks = critique.get("hard_checks", {})
    soft_scores = critique.get("soft_scores", {})
    return (
        all(bool(hard_checks.get(key)) for key in ("safe_for_children", "age_appropriate", "follows_request"))
        and all(float(soft_scores.get(key, 0)) >= 0.5 for key in ("bedtime_tone", "vocabulary_fit", "story_arc", "read_aloud_quality"))
    )


def normalize_critique(raw: Dict[str, Any]) -> Dict[str, Any]:
    critique = fallback_critique("no reason provided")
    critique.update({key: value for key, value in raw.items() if value is not None})

    hard = critique.get("hard_checks") or {}
    critique["hard_checks"] = {
        "safe_for_children": bool(hard.get("safe_for_children", False)),
        "age_appropriate": bool(hard.get("age_appropriate", False)),
        "follows_request": bool(hard.get("follows_request", False)),
    }

    soft = critique.get("soft_scores") or {}
    normalized_soft = {}
    for key in ("bedtime_tone", "vocabulary_fit", "story_arc", "read_aloud_quality"):
        try:
            score = float(soft.get(key, 0))
        except (TypeError, ValueError):
            score = 0
        normalized_soft[key] = min(1, max(0, score))
    critique["soft_scores"] = normalized_soft

    for key in ("strengths", "revision_suggestions"):
        value = critique.get(key)
        if isinstance(value, str):
            critique[key] = [value]
        elif not isinstance(value, list):
            critique[key] = []

    critique["passes"] = bool(critique.get("passes", False)) and critique_passes(critique)
    return critique


UNSAFE_PATTERNS = (
    # Ingestion of unfamiliar magical food / medicine / potions.
    "drank from the bottle",
    "took a sip",
    "sipped the potion",
    "drink the potion",
    "drank the potion",
    "ate the glowing berries",
    "eating the magical",
    # Sneaking out / leaving home alone.
    "tiptoed out of bed",
    "snuck out of the house",
    "without waking his parents",
    "without waking her parents",
    "into the dense trees",
    "wandered into the woods",
    # Climbing onto / riding large animals or risky surfaces.
    "climb aboard",
    "climbed onto the whale",
    "swam out into deeper water",
    "climbed onto the windowsill",
    "climbed out the window",
    # Child-built devices actually flying / launching.
    "lifted off the ground",
    "launched into the sky",
    "took off into the sky",
    # Sleep / nap as euphemism for death.
    "just taking a little nap",
    "fell asleep and never woke",
    "went to sleep forever",
)


def apply_deterministic_safety_checks(critique: Dict[str, Any], story: str) -> Dict[str, Any]:
    """Pin `deterministic_safety_flags` to what the code actually matched.

    The critic prompt does not include this field in its schema, but LLMs sometimes
    hallucinate it anyway. We always overwrite, so the audit log reflects code, not the LLM.
    """
    story_lower = story.lower()
    matched = [pattern for pattern in UNSAFE_PATTERNS if pattern in story_lower]
    critique["deterministic_safety_flags"] = matched
    if not matched:
        return critique

    critique["passes"] = False
    critique.setdefault("hard_checks", {})["safe_for_children"] = False
    suggestions = critique.setdefault("revision_suggestions", [])
    suggestions.append(
        "Remove physically imitable or unsafe content flagged by deterministic safety checks. "
        "Use pretend play, dreams, trusted grown-ups, symbolic magic, and emotional comfort instead."
    )
    return critique


def apply_deterministic_branded_check(critique: Dict[str, Any], story: str) -> Dict[str, Any]:
    """Force `follows_request` to False if a copyrighted name leaked into the story.

    `replace_branded_terms` runs on the brief at intake, so the generator is asked
    to write about "a kind snow queen" rather than "Elsa". The model still
    sometimes re-attaches the original name ("...a kind snow queen named Elsa...").
    This check catches that and triggers a revision with explicit feedback.
    Copyright is a `follows_request` failure, not a `safe_for_children` failure.
    """
    story_lower = story.lower()
    matched = [name for name in BRANDED_NAMES_IN_STORY if name in story_lower]
    critique["branded_names_in_story"] = matched
    if not matched:
        return critique

    critique["passes"] = False
    critique.setdefault("hard_checks", {})["follows_request"] = False
    suggestions = critique.setdefault("revision_suggestions", [])
    suggestions.append(
        f"Branded character / franchise names leaked into the story: {matched}. "
        "Replace with original generic analogues (e.g. 'Elsa' -> 'the snow queen', "
        "'Peter Parker' -> 'the hero', 'Hogwarts' -> 'a cozy school of magic'). "
        "Do not use real-world or copyrighted names."
    )
    return critique
