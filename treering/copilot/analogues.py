"""
FRADSCR Climate Analogue & RAG Knowledge Engine
================================================
Provides domain-specific historical climate memory, solar teleconnection
analogue matching, and retrieval-augmented context for Ethiopian hydroclimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class HistoricalClimateAnalogue:
    """Historical Ethiopian drought or climate milestone."""
    id: str
    year_range: str
    start_year: int
    end_year: int
    name: str
    local_name: str
    drought_class: int  # 0=Normal, 1=Moderate, 2=Severe
    severity_label: str
    solar_cycle: int
    solar_phase_description: str
    sunspot_mean: float
    ocean_drivers: str
    spei_min: float
    rwi_mean: float
    key_regions: List[str]
    humanitarian_impact: str
    prescriptive_actions_taken: List[str]
    modern_operational_takeaways: List[str]


# Authoritative historical catalogue of Ethiopian hydroclimatic milestones
HISTORICAL_ANALOGUES: List[HistoricalClimateAnalogue] = [
    HistoricalClimateAnalogue(
        id="famine_1888_1892",
        year_range="1888–1892",
        start_year=1888,
        end_year=1892,
        name="The Great Ethiopian Famine",
        local_name="Kifu Qen (ክፉ ቀን — The Evil Days)",
        drought_class=2,
        severity_label="Severe Drought",
        solar_cycle=12,
        solar_phase_description="Transition between Solar Cycle 12 minimum and Cycle 13 ascent",
        sunspot_mean=18.5,
        ocean_drivers="Extreme canonical El Niño followed by prolonged multi-basin SST warming",
        spei_min=-1.85,
        rwi_mean=0.58,
        key_regions=["Tigray", "Gondar", "Wollo", "Shoa", "Borana Plateau"],
        humanitarian_impact="Loss of ~1/3 human population; 90% livestock mortality from drought and rinderpest.",
        prescriptive_actions_taken=[
            "Emergency grain distribution from imperial granaries",
            "Pastoralist seasonal herd dispersion to high-elevation montane refugia",
        ],
        modern_operational_takeaways=[
            "Declare national water emergency immediately when solar minimum aligns with canonical El Niño",
            "Mandate early commercial destocking before livestock body condition deteriorates below market value",
            "Lock in strategic water reserves in major reservoirs (GERD, Tekeze) 12 months in advance",
        ],
    ),
    HistoricalClimateAnalogue(
        id="drought_1913_1914",
        year_range="1913–1914",
        start_year=1913,
        end_year=1914,
        name="1913 Sahel-Sudan-Ethiopian Low Flow Crisis",
        local_name="Yeqen Qen (የቀን ቀን)",
        drought_class=2,
        severity_label="Severe Drought",
        solar_cycle=14,
        solar_phase_description="Solar Cycle 14/15 absolute minimum",
        sunspot_mean=5.4,
        ocean_drivers="Deep negative Indian Ocean Dipole (IOD) suppressing monsoon cloud convergence",
        spei_min=-1.45,
        rwi_mean=0.71,
        key_regions=["Upper Blue Nile Basin", "Gondar", "Sudan borderland", "Afar"],
        humanitarian_impact="Historic collapse of Blue Nile discharge; lowest Roda Nilometer reading in 1,300 years.",
        prescriptive_actions_taken=[
            "Rationed water extraction along Blue Nile tributaries",
            "Migration of pastoral herds toward permanent river valleys",
        ],
        modern_operational_takeaways=[
            "Enact strict transboundary reservoir water release coordination between GERD and downstream dams",
            "Restrict non-essential irrigation in the Abbay (Blue Nile) basin",
            "Prioritize baseflow maintenance for municipal water supplies",
        ],
    ),
    HistoricalClimateAnalogue(
        id="famine_1973_1974",
        year_range="1973–1974",
        start_year=1973,
        end_year=1974,
        name="Wollo & Northern Highlands Famine",
        local_name="Yewollo Dirq (የወሎ ድርቅ)",
        drought_class=2,
        severity_label="Severe Drought",
        solar_cycle=20,
        solar_phase_description="Descending phase of Solar Cycle 20 approaching 1976 minimum",
        sunspot_mean=38.0,
        ocean_drivers="Successive failure of both Belg (spring) and Kiremt (summer) rains; eastern African atmospheric subsidence",
        spei_min=-1.52,
        rwi_mean=0.69,
        key_regions=["Wollo", "Tigray", "Northern Shoa", "Afar lowlands"],
        humanitarian_impact="100,000–200,000 human casualties; trigger for major Ethiopian revolution and political transition.",
        prescriptive_actions_taken=[
            "Late international famine relief mobilization",
            "Emergency trucked water delivery to roadside centers",
        ],
        modern_operational_takeaways=[
            "Treat Belg rainfall failure during solar cycle descent as a Tier-1 trigger for anticipatory action",
            "Never wait for nutritional surveys; trigger anticipatory cash transfers 3 months prior to harvest",
            "Maintain solar-powered deep borehole pump readiness across highland margins",
        ],
    ),
    HistoricalClimateAnalogue(
        id="famine_1984_1985",
        year_range="1984–1985",
        start_year=1984,
        end_year=1985,
        name="1984 Great Ethiopian Famine",
        local_name="Yemedebeya Dirq (የመደበኛ ድርቅ / 1977 ዓ.ም)",
        drought_class=2,
        severity_label="Severe Drought",
        solar_cycle=21,
        solar_phase_description="Late descending limb toward Solar Cycle 21 minimum (1986)",
        sunspot_mean=31.2,
        ocean_drivers="Super El Niño of 1982–1983 coupled with prolonged tropical Indian Ocean warming",
        spei_min=-1.92,
        rwi_mean=0.55,
        key_regions=["Tigray", "Wollo", "Gondar", "Eritrea", "Hararghe", "Borana"],
        humanitarian_impact="Over 1 million deaths; devastating socioeconomic upheaval and global humanitarian awareness.",
        prescriptive_actions_taken=[
            "Massive global food airlifts and feeding camps",
            "Resettlement programs from degraded northern highlands to south-west",
        ],
        modern_operational_takeaways=[
            "Aggressive multi-year borehole drilling and solarization before drought peaks",
            "Decentralized grain warehousing to avoid logistical choke points",
            "Reinforcement learning dispatch for pastoral water points to prevent localized aquifer collapse",
        ],
    ),
    HistoricalClimateAnalogue(
        id="drought_2002_2003",
        year_range="2002–2003",
        start_year=2002,
        end_year=2003,
        name="2002–2003 Horn of Africa Food Crisis",
        local_name="Ye-2002 Dirq (የ2002 ድርቅ)",
        drought_class=1,
        severity_label="Moderate Drought",
        solar_cycle=23,
        solar_phase_description="Post-maximum declining phase of Solar Cycle 23",
        sunspot_mean=81.0,
        ocean_drivers="Moderate El Niño modulating ITCZ northward advance",
        spei_min=-1.05,
        rwi_mean=0.82,
        key_regions=["Afar", "Somali", "East Shewa", "Borana Lowlands"],
        humanitarian_impact="14 million people in need of emergency food assistance; averted mass mortality via safety nets.",
        prescriptive_actions_taken=[
            "Early safety-net deployments leading to founding of PSNP (Productive Safety Net Programme)",
            "Targeted livestock emergency feeding and supplementary water supplies",
        ],
        modern_operational_takeaways=[
            "Activate the Productive Safety Net Programme (PSNP) scale-up window early in Q1",
            "Subsidize solar pump fuel replacement to keep communal water points operational 12+ hrs/day",
        ],
    ),
    HistoricalClimateAnalogue(
        id="drought_2015_2016",
        year_range="2015–2016",
        start_year=2015,
        end_year=2016,
        name="2015–2016 Super El Niño Drought",
        local_name="Ye-El Niño Dirq (የኤል ኒኞ ድርቅ)",
        drought_class=2,
        severity_label="Severe Drought",
        solar_cycle=24,
        solar_phase_description="Descending phase of Solar Cycle 24",
        sunspot_mean=50.5,
        ocean_drivers="Record 2015–16 Super El Niño coupled with positive Indian Ocean Dipole",
        spei_min=-1.70,
        rwi_mean=0.74,
        key_regions=["Afar", "Siti Zone", "Borana", "East Hararghe", "Wollo"],
        humanitarian_impact="Worst drought in 30 years in eastern pastoral belts; severe borehole drying.",
        prescriptive_actions_taken=[
            "Government-financed rapid humanitarian response of over $700M",
            "Emergency drilling of deep regional boreholes in pastoral corridors",
        ],
        modern_operational_takeaways=[
            "Solar water pump duty cycles must be dynamically throttled to prevent aquifer exhaustion",
            "Maintain digital real-time water point telemetry across pastoral zones",
            "Cross-border grazing and water sharing agreements with Kenya/Somalia must be pre-activated",
        ],
    ),
]


class ClimateAnalogueMatcher:
    """Matches forecast conditions against historical Ethiopian climatology."""

    def __init__(self, analogues: Optional[List[HistoricalClimateAnalogue]] = None) -> None:
        self.analogues = analogues or HISTORICAL_ANALOGUES

    def find_nearest_analogues(
        self,
        predicted_class: int,
        year: int,
        sunspot_count: Optional[float] = None,
        continuous_spei: Optional[float] = None,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Score and return the top-k closest historical climate analogues.

        Similarity scoring factors:
        - Drought class match (weight: 40%)
        - SPEI proximity (weight: 30%)
        - Solar activity proximity (weight: 20%)
        - Decadal cycle timing similarity (weight: 10%)
        """
        scored: List[Tuple[float, HistoricalClimateAnalogue]] = []

        for item in self.analogues:
            # Class match score
            class_diff = abs(item.drought_class - predicted_class)
            class_score = max(0.0, 1.0 - (class_diff * 0.5))

            # SPEI proximity score
            if continuous_spei is not None:
                spei_diff = abs(item.spei_min - continuous_spei)
                spei_score = math.exp(-spei_diff / 0.8)
            else:
                spei_score = 1.0 if item.drought_class == predicted_class else 0.5

            # Solar sunspot count proximity
            if sunspot_count is not None:
                solar_diff = abs(item.sunspot_mean - sunspot_count)
                solar_score = math.exp(-solar_diff / 50.0)
            else:
                solar_score = 0.8

            # Decadal cycle timing similarity (11-year Schwabe cycle harmonic alignment)
            cycle_phase_diff = abs((year - item.start_year) % 11)
            decadal_score = math.cos(2 * math.pi * (cycle_phase_diff / 11.0)) * 0.5 + 0.5

            composite_similarity = (
                (class_score * 0.40)
                + (spei_score * 0.30)
                + (solar_score * 0.20)
                + (decadal_score * 0.10)
            )
            scored.append((composite_similarity, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: List[Dict[str, Any]] = []

        for sim, item in scored[:top_k]:
            data = asdict(item)
            data["similarity_score"] = round(float(sim), 4)
            data["similarity_percentage"] = round(float(sim) * 100.0, 1)
            results.append(data)

        return results


_GLOBAL_MATCHER: Optional[ClimateAnalogueMatcher] = None


def get_analogue_matcher() -> ClimateAnalogueMatcher:
    """Retrieve the global climate analogue matcher singleton."""
    global _GLOBAL_MATCHER
    if _GLOBAL_MATCHER is None:
        _GLOBAL_MATCHER = ClimateAnalogueMatcher()
    return _GLOBAL_MATCHER
