"""Environmental Adaptation KB (proposal Table 3.6.3): adjust for UV, humidity and
temperature. No rule applies when weather is unavailable (all flags False)."""

from __future__ import annotations

from skiniq.rules.base import Action, Rule, ingredient_list

KB = "environmental"
LIGHT_STEPS = ("toner", "treatment")


def _flag(u, name: str) -> bool:
    return u.env_flags.get(name, False)


RULES: list[Rule] = [
    Rule("E1_uv_sunscreen", KB, "High UV -> prefer broad-protection sunscreens",
         applies=lambda u: _flag(u, "high_uv") or "sun_protection" in u.goals,
         matches=lambda p, u: p.step == "sunscreen",
         action=Action.PREFER, weight=0.3,
         reason=lambda p, u: (f"UV index is {u.environment.uv_index:g} ({u.environment.uv_level.replace('_', ' ')}): "
                              "daily sunscreen is essential") if u.environment else "Protects against UV damage"),
    Rule("E1_uv_photosensitising", KB, "High UV -> caution for retinoids and AHAs (increase sun sensitivity)",
         applies=lambda u: _flag(u, "high_uv"),
         matches=lambda p, u: p.has("retinoid") or p.has("aha"),
         action=Action.CAUTION, weight=-0.1,
         reason=lambda p, u: f"Contains {ingredient_list(p, 'retinoid' if p.has('retinoid') else 'aha', 2)}, "
         "which increases sun sensitivity: use at night and wear sunscreen"),
    Rule("E2_humid_heavy", KB, "Hot or humid weather -> caution for heavy products",
         applies=lambda u: _flag(u, "humid") or _flag(u, "hot"),
         matches=lambda p, u: "May Worsen Oily Skin" in p.tags and p.step in ("moisturizer", "mask"),
         action=Action.CAUTION, weight=-0.1,
         reason="May feel heavy in hot, humid weather"),
    Rule("E2_humid_light", KB, "Hot or humid weather -> prefer lightweight, oil-balancing products",
         applies=lambda u: _flag(u, "humid") or _flag(u, "hot"),
         matches=lambda p, u: "Good For Oily Skin" in p.tags or (p.step in LIGHT_STEPS and "Hydrating" in p.tags),
         action=Action.PREFER, weight=0.1,
         reason="Lightweight texture suited to hot, humid weather"),
    Rule("E3_dry_air", KB, "Dry or cold air -> prefer hydrating, barrier-repairing products",
         applies=lambda u: _flag(u, "dry_air") or _flag(u, "cold"),
         matches=lambda p, u: p.has("hyaluronic_acid") or p.has("ceramide") or "Hydrating" in p.tags,
         action=Action.PREFER, weight=0.15,
         reason="Adds moisture to protect skin in dry or cold air"),
]
