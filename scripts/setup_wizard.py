
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAITS_JSON = DATA_DIR / "traits.json"       # dynamic traits crafted at onboarding
PERSONA_JSON = DATA_DIR / "persona.json"     # broader persona + mode + names
IDENTITY_JSON = DATA_DIR / "identity.json"   # convenience snapshot of names/role

def _ask(prompt: str, default: str = "") -> str:
    prompt_txt = f"{prompt.strip()}"
    if default:
        prompt_txt += f" [{default}]"
    prompt_txt += ": "
    val = input(prompt_txt).strip()
    return val if val else default

def _ask_choice(prompt: str, choices, default=None, multi=False):
    label = f"{prompt.strip()} ({'/'.join(choices)})"
    if default:
        label += f" [{default}]"
    label += ": "
    raw = input(label).strip()
    if not raw and default is not None:
        return [default] if multi else default
    if multi:
        # comma separated, trim, validate
        picked = [x.strip() for x in raw.split(",") if x.strip()]
        return [x for x in picked if x in choices]
    return raw if raw in choices else (default if default else choices[0])

def _ask_float(prompt: str, default: float, minv=0.0, maxv=1.0):
    label = f"{prompt.strip()} [{default}] ({minv}-{maxv}): "
    while True:
        raw = input(label).strip()
        if not raw:
            return float(default)
        try:
            v = float(raw)
            if v < minv: v = minv
            if v > maxv: v = maxv
            return v
        except:
            print("Please enter a number.")

def _ask_bool(prompt: str, default=True):
    d = "Y/n" if default else "y/N"
    raw = input(f"{prompt.strip()} [{d}]: ").strip().lower()
    if not raw:
        return bool(default)
    return raw in ("y","yes","true","1")

def run():
    print("\n=== Trait Engine Initial Setup ===\n")

    # Names
    ai_name   = _ask("Choose a name for your AI", "Ava")
    user_name = _ask("What should the AI call you?", "User")

    # Role/Mode
    role = _ask_choice(
        "Primary role for this AI?",
        choices=[
            "chat_assistant",
            "coding_assistant",
            "creative_writer",
            "research_assistant",
            "tutor",
            "productivity_coach",
            "data_analyst",
            "custom"
        ],
        default="chat_assistant",
        multi=False
    )
    custom_role_desc = ""
    if role == "custom":
        custom_role_desc = _ask("Describe the AI's primary function in a sentence", "")

    # Output/length preferences
    no_token_limits = _ask_bool("Disable output token limits when possible?", True)
    allow_long_form = _ask_bool("Allow very long-form replies by default?", True)
    prefer_concise  = False if allow_long_form else _ask_bool("Prefer concise replies?", True)

    # Persona Skeleton
    print("\n--- Persona & Traits ---")
    backstory = _ask("Short backstory or descriptor (optional)", "")
    objectives = _ask("Primary objectives or goals (comma-separated)", "")
    objectives_list = [x.strip() for x in objectives.split(",") if x.strip()]

    # Core style knobs (0..1 sliders)
    tone_formality    = _ask_float("Formality (0=casual, 1=formal)", 0.35, 0.0, 1.0)
    tone_warmth       = _ask_float("Warmth/Empathy (0=cool, 1=warm)", 0.75, 0.0, 1.0)
    tone_humour       = _ask_float("Humour (0=serious, 1=playful)", 0.35, 0.0, 1.0)
    tone_directness   = _ask_float("Directness (0=indirect, 1=straight)", 0.7, 0.0, 1.0)
    tone_creativity   = _ask_float("Creativity (0=literal, 1=imaginative)", 0.6, 0.0, 1.0)
    tone_skepticism   = _ask_float("Skepticism (0=credulous, 1=skeptical)", 0.55, 0.0, 1.0)
    tone_analytical   = _ask_float("Analytical (0=intuitive, 1=analytical)", 0.65, 0.0, 1.0)
    tone_verbosity    = _ask_float("Verbosity (0=terse, 1=verbose)", 0.65, 0.0, 1.0)

    # Conversation preferences
    greet_style = _ask_choice("Greeting style", ["short","neutral","warm","none"], default="warm")
    slang       = _ask_choice("Use slang", ["never","rarely","sometimes","often"], default="rarely")
    emojis      = _ask_choice("Use emojis", ["never","sparingly","sometimes","often"], default="sparingly")

    # Boundaries / Safety knobs
    refuse_unsure   = _ask_bool("Politely refuse when unsure?", True)
    cite_sources    = _ask_bool("Cite sources when making nontrivial claims (when browsing is available)?", True)
    avoid_personas  = _ask_bool("Avoid pretending to be a human?", True)
    avoid_thirdperson = _ask_bool("Avoid talking about self in third person?", True)

    # Guardrails / topics
    taboo_raw = _ask("Topics to avoid entirely (comma-separated, optional)", "")
    taboo = [x.strip() for x in taboo_raw.split(",") if x.strip()]

    # Custom traits list
    print("\nEnter any number of custom traits as 'name=weight' on separate lines.")
    print("Examples: 'curious=0.8', 'methodical=0.7'. Leave blank to finish.")
    custom_traits = []
    while True:
        line = input("trait: ").strip()
        if not line: break
        if "=" in line:
            n, w = line.split("=", 1)
            n = n.strip()
            try:
                w = float(w.strip())
            except:
                print("  weight must be a number 0..1, defaulting to 0.5")
                w = 0.5
            w = max(0.0, min(1.0, w))
            custom_traits.append({"name": n, "weight": w})
        else:
            custom_traits.append({"name": line, "weight": 0.5})

    # Skills / tool preferences
    skills_raw = _ask("Key skills/competencies (comma-separated, optional)", "")
    skills = [x.strip() for x in skills_raw.split(",") if x.strip()]

    # Mode-specific options
    mode_prefs = {}
    if role == "coding_assistant":
        mode_prefs["explain_while_coding"] = _ask_bool("Explain reasoning inline with code examples?", True)
        mode_prefs["prefer_standards"] = _ask_bool("Prefer standards and official docs?", True)
        mode_prefs["tests_first"] = _ask_bool("Encourage tests first (TDD) where useful?", False)
    elif role == "creative_writer":
        mode_prefs["style_refs"] = _ask("Style references (comma-separated authors/genres, optional)", "")
        mode_prefs["avoid_purple_prose"] = _ask_bool("Avoid purple prose?", True)
    elif role == "research_assistant":
        mode_prefs["summarize_first"] = _ask_bool("Summarize before deep-diving?", True)
        mode_prefs["contrast_views"] = _ask_bool("Contrast multiple viewpoints?", True)
    elif role == "tutor":
        mode_prefs["socratic"] = _ask_bool("Use Socratic questioning?", True)
        mode_prefs["check_understanding"] = _ask_bool("Periodically check for understanding?", True)
    elif role == "productivity_coach":
        mode_prefs["timeboxing"] = _ask_bool("Promote timeboxing?", True)
        mode_prefs["prioritization"] = _ask_choice("Default prioritization method", ["Eisenhower","MoSCoW","RICE"], default="Eisenhower")
    elif role == "data_analyst":
        mode_prefs["prefer_charts"] = _ask_bool("Prefer charts/tables in explanations?", True)
        mode_prefs["assumptions_block"] = _ask_bool("List assumptions explicitly?", True)

    persona = {
        "ai_name": ai_name,
        "user_name": user_name,
        "role": role,
        "custom_role_desc": custom_role_desc,
        "no_token_limits": no_token_limits,
        "allow_long_form": allow_long_form,
        "prefer_concise": prefer_concise,
        "backstory": backstory,
        "objectives": objectives_list,
        "style": {
            "formality": tone_formality,
            "warmth": tone_warmth,
            "humour": tone_humour,
            "directness": tone_directness,
            "creativity": tone_creativity,
            "skepticism": tone_skepticism,
            "analytical": tone_analytical,
            "verbosity": tone_verbosity,
            "greeting": greet_style,
            "slang": slang,
            "emojis": emojis
        },
        "boundaries": {
            "refuse_when_unsure": refuse_unsure,
            "cite_sources": cite_sources,
            "avoid_pretending_human": avoid_personas,
            "avoid_third_person_self": avoid_thirdperson,
            "taboo_topics": taboo
        },
        "skills": skills,
        "mode_prefs": mode_prefs,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    # Save persona & identity
    PERSONA_JSON.write_text(json.dumps(persona, indent=2), encoding="utf-8")
    IDENTITY_JSON.write_text(json.dumps({
        "ai_name": ai_name,
        "user_name": user_name,
        "role": role
    }, indent=2), encoding="utf-8")

    # Overwrite traits.json from custom traits only (remove any existing defaults)
    traits_doc = {
        "traits": custom_traits,  # NOTE: Only user-specified traits
        "mood": {"valence": 0.0, "arousal": 0.0, "stability": 1.0},
        "decay": {"mood_half_life_minutes": 90}
    }
    TRAITS_JSON.write_text(json.dumps(traits_doc, indent=2), encoding="utf-8")

    print("\n✅ Setup complete. Saved:")
    print(f"  - {PERSONA_JSON}")
    print(f"  - {TRAITS_JSON}")
    print(f"  - {IDENTITY_JSON}")
    print("\nYou can re-run this wizard anytime: python -m scripts.setup_wizard\n")

if __name__ == "__main__":
    run()
