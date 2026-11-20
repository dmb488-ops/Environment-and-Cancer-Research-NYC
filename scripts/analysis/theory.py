"""Initial theoretical justification for each candidate factor (draft for group review).

rating  : strength of the scientific case that the factor affects lung-cancer INCIDENCE  (strong / moderate / weak / none)
expected: expected sign of the association with lung-cancer incidence (+ higher factor -> more cancer, - fewer, ? ambiguous)
priority: used only to break ties between near-duplicate factors (lower = keep)
Texts are short on purpose; the methods document carries the longer discussion and the sources.
"""
import json
import os

THEORY = {
    "smoke": dict(rating="strong", expected="+", priority=1, role="Behaviour (confounder / main exposure)",
                  why="Cigarette smoking is the number-one risk factor for lung cancer and is linked to about 80% of US lung-cancer deaths (CDC). The survey measures current adult smoking, while lung cancer today reflects smoking decades ago, so it is a proxy for each area's smoking history."),
    "pov": dict(rating="moderate", expected="+", priority=3, role="Socioeconomic context",
                why="Lower-income areas have more smoking, more occupational and indoor exposures and worse air. The link is partly mediated by smoking, "
                    "and for all cancers combined the gradient can reverse because screening-detected cancers are commoner in richer areas."),
    "ins": dict(rating="weak", expected="?", priority=4, role="Access to care (detection)",
                why="Lack of insurance mainly changes whether a cancer is found (screening, diagnosis), not whether it starts, so the sign is ambiguous: "
                    "poorer access lowers recorded incidence, while the socioeconomic disadvantage behind it raises risk."),
    "ndvi": dict(rating="moderate", expected="-", priority=3, role="Environment (protective, hypothesis)",
                 why="Vegetation can filter and dilute air pollution, encourages physical activity and lowers heat and stress. Observational cohorts report lower lung-cancer risk with higher NDVI (hazard ratio 0.89 per 0.1 NDVI in a 2025 cohort), but NDVI is an imperfect exposure measure and greenness also tracks lower-density suburban areas."),
    "so2": dict(rating="weak", expected="+", priority=6, role="Air pollution (combustion)",
                why="Marker of sulfur-containing fuel combustion (heating oil). Limited evidence for lung cancer on its own, and the NYC series in this project ends in 2016."),
    "o3": dict(rating="weak", expected="?", priority=5, role="Air pollution (secondary)",
               why="Ozone is a respiratory irritant but is not an established lung carcinogen. In cities its summer mean is lower near traffic (nitric oxide destroys ozone), "
                   "so it behaves as an inverse marker of the urban core."),
    "no2": dict(rating="moderate", expected="+", priority=3, role="Air pollution (traffic)",
                why="Traffic-related pollutant used as a marker of the traffic mixture; several cohort studies link it to lung-cancer incidence, but it is hard to separate from PM₂.₅ and other traffic emissions."),
    "no": dict(rating="weak", expected="+", priority=5, role="Air pollution (fresh combustion)",
               why="Fresh combustion emission that converts to NO₂ within minutes. No separate carcinogenic mechanism is established; it is almost a duplicate of NO₂."),
    "pm25": dict(rating="strong", expected="+", priority=2, role="Air pollution (particles)",
                 why="IARC classified outdoor air pollution, and particulate matter within it, as carcinogenic to humans (Group 1) in 2013, with sufficient evidence for lung cancer and a rising risk with rising exposure."),
    "bc": dict(rating="moderate", expected="+", priority=4, role="Air pollution (diesel / combustion)",
               why="Black carbon tracks diesel and other combustion soot; IARC classified diesel engine exhaust as carcinogenic to humans (Group 1, lung cancer) in 2012. Few studies use black carbon itself, and in NYC it is almost identical to PM₂.₅."),
    "snap": dict(rating="weak", expected="?", priority=5, role="Food environment (proxy)",
                 why="Count of SNAP-authorised stores per km². Mostly reflects how dense and commercial an area is; diet effects on lung cancer are modest and not captured by store counts."),
    "super": dict(rating="weak", expected="-", priority=4, role="Food environment (diet quality)",
                  why="Better access to supermarkets could improve diet (fruit and vegetable intake is weakly protective for lung cancer). Per-km² counts mainly track urban density."),
    "conv": dict(rating="weak", expected="+", priority=3, role="Food environment / tobacco retail",
                 why="Convenience stores are a main tobacco retail channel, so store density may act as a smoking-access proxy (a hypothesis; the literature still needs checking by the group). As a density measure it also tracks the urban core."),
    "tmean": dict(rating="none", expected="?", priority=7, role="Climate",
                  why="No established mechanism for lung-cancer incidence. Annual mean temperature mostly separates the dense, hotter core from the outer areas (urban heat island)."),
    "tmax": dict(rating="none", expected="?", priority=7, role="Climate",
                 why="Summer heat is a health hazard for other outcomes (heat illness, ozone formation) but has no established link to lung-cancer incidence. Skin cancer depends on UV, not temperature."),
    "precip": dict(rating="none", expected="?", priority=8, role="Climate (negative control)",
                   why="No plausible mechanism. Almost all of its variation is year-to-year weather shared by the whole city, so any neighborhood-level pattern is a location fingerprint."),
}

# factors left out of the regression set by design (still shown to the machine-learning models as checks)
EXCLUDED_BY_DESIGN = {
    "so2": "Data end in 2016 (7 of 14 years), so it cannot be used in the annual panel; weak mechanism; collinear with the other air pollutants.",
    "tmean": "No established mechanism; only 9% of its variation is between neighborhoods (the rest is year-to-year weather shared by the whole city).",
    "tmax": "No established mechanism; only 16% of its variation is between neighborhoods.",
    "precip": "No plausible mechanism; only 1% of its variation is between neighborhoods. Kept in the machine-learning models as a negative control.",
}

if __name__ == "__main__":      # patch the text of an existing results file without re-running the models
    from common import OUT
    p = os.path.join(OUT, "analysis_results.json")
    R = json.load(open(p))
    for k, t in THEORY.items():
        R["theory"][k].update(why=t["why"], role=t["role"])
    json.dump(R, open(p, "w"), separators=(",", ":"))
    print("patched", p)
