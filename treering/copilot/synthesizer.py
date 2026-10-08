"""
FRADSCR Deterministic Climatological Synthesizer
================================================
Provides robust, zero-dependency, offline-capable natural language synthesis
for hydroclimatic early warnings, policy briefs, and advisory reports across
three supported languages (English, Amharic, Afaan Oromoo) and target personas.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from treering.copilot.prompts import MULTILINGUAL_GLOSSARY


class ClimatologicalAdvisorySynthesizer:
    """Generates structured, authoritative climatological advisories offline."""

    @staticmethod
    def generate_advisory(
        prediction_result: Dict[str, Any],
        historical_analogues: List[Dict[str, Any]],
        persona: Literal["minister", "pastoralist", "scientist"] = "minister",
        language: Literal["en", "am", "om"] = "en",
    ) -> Dict[str, Any]:
        """Synthesize a structured domain-specific advisory report.

        Args:
            prediction_result: Dictionary returned by predict_drought.
            historical_analogues: List of matched historical climate events.
            persona: Target audience ('minister', 'pastoralist', 'scientist').
            language: Target language ('en', 'am', 'om').

        Returns:
            Dictionary containing structured markdown and metadata.
        """
        lang = language if language in MULTILINGUAL_GLOSSARY else "en"
        glossary = MULTILINGUAL_GLOSSARY[lang]

        year = prediction_result.get("year", 2026)
        pred_class = prediction_result.get("predicted_drought_class", 0)
        confidence = prediction_result.get("model_confidence", 0.85)
        severity_label = prediction_result.get("severity_label", "Normal")
        continuous_spei = prediction_result.get("continuous_spei", 0.0)
        spei_ci = prediction_result.get("spei_confidence_interval", {})
        hydro = prediction_result.get("hydrogeology", {})
        grid = prediction_result.get("grid_cell", {})
        req_lat = grid.get("requested_lat", 9.0)
        req_lon = grid.get("requested_lon", 38.75)
        prescriptive_action = prediction_result.get("prescriptive_action", "HOLD FUNDS (CONSERVE)")

        # Map severity to localized string
        if pred_class == 2:
            local_severity = glossary["severity_severe"]
            banner_emoji = "🚨"
        elif pred_class == 1:
            local_severity = glossary["severity_moderate"]
            banner_emoji = "⚠️"
        else:
            local_severity = glossary["severity_normal"]
            banner_emoji = "✅"

        analogues_text = ClimatologicalAdvisorySynthesizer._format_analogues(
            historical_analogues, lang
        )
        recommendations = ClimatologicalAdvisorySynthesizer._build_recommendations(
            pred_class=pred_class,
            persona=persona,
            language=lang,
            hydro=hydro,
            prescriptive_action=prescriptive_action,
        )

        if lang == "am":
            content_md = f"""# {banner_emoji} {glossary['title']} ({year})
**ዒላማ ቦታ:** ኬክሮስ {req_lat:.2f}°, ኬንትሮስ {req_lon:.2f}° | **ሁኔታ:** **{local_severity}**  
**የአምሳያ ደረጃ:** {persona.capitalize()} | **የሞዴል እርግጠኝነት:** {confidence * 100:.1f}%

---

### {glossary['section_prediction']}
* **የተተነበየው የድርቅ ደረጃ:** {severity_label} (ክፍል {pred_class})
* **የሞዴል የውሳኔ እርግጠኝነት:** {confidence * 100:.1f}%
* **የተሰላው የSPEI እርጥበት ማውጫ:** {continuous_spei:+.2f} (ከ 10% እስከ 90% የዕድል ወሰን: [{spei_ci.get('p10', '—')}, {spei_ci.get('p90', '—')}])
* **የከርሰ ምድር ውጥረት ማውጫ:** {hydro.get('aquifer_stress_index', 0.0):.1f}%
* **የውሃ ፓምፕ የስራ ሰዓት ምክረ ሀሳብ:** {hydro.get('recommended_solar_pumping_hours', 8.0)} ሰዓት/በቀን

### {glossary['section_solar']}
* የፀሐይ ~11 ዓመት የሽዋቤ (Schwabe) ዑደት እና የሕንድ ውቅያኖስ የዲፖል (IOD) መስተጋብር በዚህ ወቅት በኢትዮጵያ የበጋ እና የክረምት ዝናብ ስርጭት ላይ ከፍተኛ ተፅዕኖ አሳድረዋል።
* የዛፍ ግንዶች ዓመታዊ የእድገት ቀለበቶች (Juniperus procera) እንደሚያሳዩት ባለፉት ዓመታት የተከማቸ የካርቦሃይድሬት እና የአፈር እርጥበት ክምችት ለቀጣይ ዓመታት የውሃ ሚዛን ወሳኝ ነው።

### {glossary['section_analogues']}
{analogues_text}

### {glossary['section_prescriptive']}
{recommendations}

---
*{glossary['disclaimer']}*
"""
        elif lang == "om":
            content_md = f"""# {banner_emoji} {glossary['title']} ({year})
**Bakka:** Latitude {req_lat:.2f}°, Longitude {req_lon:.2f}° | **Sadarkaa:** **{local_severity}**  
**Qophii:** {persona.capitalize()} | **Amanamummaa Moodeelii:** {confidence * 100:.1f}%

---

### {glossary['section_prediction']}
* **Sadarkaa Hoongee Raagame:** {severity_label} (Class {pred_class})
* **Amanamummaa Moodeelii:** {confidence * 100:.1f}%
* **Kallattii SPEI Jiidha Qilleensaa:** {continuous_spei:+.2f} (Daangaa 10%-90%: [{spei_ci.get('p10', '—')}, {spei_ci.get('p90', '—')}])
* **Dhiibbaa Bishaanii Lafa Jalaa:** {hydro.get('aquifer_stress_index', 0.0):.1f}%
* **Sa'aatii Paampii Aduu:** {hydro.get('recommended_solar_pumping_hours', 8.0)} sa'aatii/guyyaatti

### {glossary['section_solar']}
* Dhiibbaan marsaa aduu (Schwabe cycle ~waggaa 11) fi dambalii galaana Hindii (IOD) rooba Gannaafi Arfaasaa Itoophiyaa irratti dhiibbaa qabaachuun mirkanaa'eera.
* Mukti Gaattiraa (Juniperus procera) jiidha lafa keessaa fi dandeettii bishaanii waggoota darban eeggatee kan agarsiisu ta'uu mirkaneesseera.

### {glossary['section_analogues']}
{analogues_text}

### {glossary['section_prescriptive']}
{recommendations}

---
*{glossary['disclaimer']}*
"""
        else:  # English
            content_md = f"""# {banner_emoji} {glossary['title']} ({year})
**Target Location:** Lat {req_lat:.2f}°, Lon {req_lon:.2f}° | **Status:** **{local_severity}**  
**Advisory Persona:** {persona.capitalize()} | **Model Confidence:** {confidence * 100:.1f}%

---

### {glossary['section_prediction']}
* **Predicted Hydroclimatic Category:** {severity_label} (Class {pred_class})
* **Calibrated Decisive Confidence:** {confidence * 100:.1f}%
* **Standardized Precipitation-Evapotranspiration Index (SPEI):** {continuous_spei:+.2f} (10th–90th percentile: [{spei_ci.get('p10', '—')}, {spei_ci.get('p90', '—')}])
* **Aquifer Stress Index:** {hydro.get('aquifer_stress_index', 0.0):.1f}%
* **Recommended Solar Pumping Duty Cycle:** {hydro.get('recommended_solar_pumping_hours', 8.0)} hrs/day
* **Operational Directive:** `{prescriptive_action}`

### {glossary['section_solar']}
* **Schwabe Solar Teleconnection:** The ~11-year solar magnetic cycle modulates high-altitude stratospheric ozone heating, influencing the latitudinal migration of the Intertropical Convergence Zone (ITCZ) and Ethiopia's Kiremt summer monsoon.
* **Dendrochronological Memory:** Cambial cell division in *Juniperus procera* exhibits biological memory ($\tau = 0..2$ yr lag), indicating that root-zone moisture reserves and antecedent carbohydrate storage buffer multi-year deficits.

### {glossary['section_analogues']}
{analogues_text}

### {glossary['section_prescriptive']}
{recommendations}

---
*{glossary['disclaimer']}*
"""

        return {
            "year": year,
            "persona": persona,
            "language": lang,
            "severity_label": severity_label,
            "predicted_drought_class": pred_class,
            "confidence": confidence,
            "advisory_markdown": content_md,
            "prescriptive_action": prescriptive_action,
        }

    @staticmethod
    def _format_analogues(analogues: List[Dict[str, Any]], lang: str) -> str:
        """Format historical climate analogues into markdown text."""
        if not analogues:
            return "No historical analogues within similarity threshold."

        lines = []
        for a in analogues:
            name = a.get("name", "Historical Event")
            local_name = a.get("local_name", "")
            yr = a.get("year_range", "")
            sim = a.get("similarity_percentage", 0.0)
            spei = a.get("spei_min", -1.0)
            takeaways = a.get("modern_operational_takeaways", [])

            if lang == "am":
                line = f"* **{name} ({yr} - {local_name})** [ተመሳሳይነት: {sim}%]\n"
                line += f"  - የSPEI እርጥበት ማውጫ: {spei:.2f} | የተጎዱ አካባቢዎች: {', '.join(a.get('key_regions', []))}\n"
                if takeaways:
                    line += f"  - **ዋና ትምህርት:** {takeaways[0]}"
            elif lang == "om":
                line = f"* **{name} ({yr})** [Walfakkeenya: {sim}%]\n"
                line += f"  - SPEI: {spei:.2f} | Naannolee miidhaman: {', '.join(a.get('key_regions', []))}\n"
                if takeaways:
                    line += f"  - **Barumsa Guddaa:** {takeaways[0]}"
            else:
                line = f"* **{name} ({yr}) — {local_name}** [Similarity: {sim}%]\n"
                line += f"  - Peak Deficit: SPEI {spei:.2f} | Critical Regions: {', '.join(a.get('key_regions', []))}\n"
                if takeaways:
                    line += f"  - **Core Operational Lesson:** {takeaways[0]}"
            lines.append(line)

        return "\n".join(lines)

    RECOMMENDATION_REGISTRY: Dict[Tuple[int, str, str], str] = {
        # Severe Drought (Class 2) - Minister
        (2, "minister", "am"): (
            "* **የውሃ ክምችት ጥበቃ:** በታላቁ የኢትዮጵያ ሕዳሴ ግድብ (GERD) እና ሌሎች ዋና ግድቦች ላይ የውሃ መልቀቂያ ኮታዎችን በአስቸኳይ ይከልሱ።\n"
            "* **የአስቸኳይ ጊዜ በጀት:** ቀደምት የድርቅ መከላከል ድጋፍ ፈንድ (Anticipatory Action Financing) ያግብሩ።\n"
            "* **የእህል ክምችት:** ስትራቴጂካዊ የእህል ክምችት ወደ ሰሜኑና ደቡቡ አርብቶ አደር ዞኖች እንዲጓጓዝ ትዕዛዝ ያስተላልፉ።"
        ),
        (2, "minister", "om"): (
            "* **Bishaan Kuusuu:** Kuusaa bishaanii hidha Abbayyaa (GERD) fi hidhoota biroo keessatti qisaasa'ina bishaanii dhowwaa.\n"
            "* **Baajata Hatattamaa:** Baajata qophii balaa hoongee dursee kennamu hojiirra oolchaa.\n"
            "* **Midhaan Nyaataa:** Kuusaa midhaanii naannoolee horsiisee bultoota Booranaaf akka qaqqabu godhaa."
        ),
        (2, "minister", "en"): (
            "* **Strategic Reservoir Retention:** Tighten dry-season water release curves at the Grand Ethiopian Renaissance Dam (GERD) and Tekeze to safeguard multi-year baseflow.\n"
            "* **Anticipatory Financing:** Trigger the Contingency Emergency Response Component (CERC) 4 months prior to projected monsoon cessation.\n"
            "* **Regional Logistics:** Pre-position strategic grain and animal fodder reserves along the Tigray, Wollo, and Borana pastoral axes."
        ),
        # Severe Drought (Class 2) - Pastoralist
        (2, "pastoralist", "am"): (
            "* **የፀሐይ ፓምፕ አጠቃቀም:** የጉድጓድ ውሃ እንዳይነጥፍ የፓምፕ ስራ ሰዓትን በቀን ወደ {pump_hrs} ሰዓት ይገድቡ።\n"
            "* **የከብቶች ሽያጭ:** የእንስሳት አካል ሳይከሳ እና ዋጋቸው ሳይቀንስ አስቀድሞ የገበያ ሽያጭ (Commercial Destocking) ያካሂዱ።\n"
            "* **የጋራ ግጦሽ አስተዳደር:** ከአጎራባች ዞኖች ጋር ባህላዊ የውሃና የግጦሽ መጋራት ስምምነትን ያጠናክሩ።"
        ),
        (2, "pastoralist", "om"): (
            "* **Paampii Aduu:** Eela bishaanii akka hin gogneef hojii paampii guyyaatti sa'aatii {pump_hrs} tti daangessaa.\n"
            "* **Daldala Horii:** Beeyladoota osoo hin hirin gabaatti dhiyeessuun maallaqa qusadhaa.\n"
            "* **Mootummaa Gadaa:** Sirna Gadaa fayyadamuun haala dhiheessii bishaanii fi lafa dheedichaa wal qooddadhaa."
        ),
        (2, "pastoralist", "en"): (
            "* **Aquifer Conservation:** Throttle solar deep-borehole pumping to {pump_hrs} hours/day to prevent drawdown cone collapse in vulnerable alluvial basins.\n"
            "* **Commercial Destocking:** Advise pastoralist cooperatives to enact early commercial destocking while livestock market valuations remain viable.\n"
            "* **Customary Water Sharing:** Convene customary pastoral elders (e.g. Borana Gadaa water councils) to establish rotation protocols for communal deep tula wells."
        ),
        # Severe Drought (Class 2) - Scientist
        (2, "scientist", "am"): (
            "* **የሳተላይት ክትትል:** የቴራ-ሞዲስ (MODIS NDVI) እና የGRACE የከርሰ ምድር ውሃ ሳተላይት መረጃዎችን በየሳምንቱ ያነፃፅሩ።\n"
            "* **የዛፍ ቀለበት ጥናት:** በአካባቢው የዛፍ ግንዶች ላይ ተጨማሪ ናሙናዎችን በመውሰድ የባዮሎጂካል ካርቦሃይድሬት ማገገሚያ ሁኔታን ይመዝግቡ።"
        ),
        (2, "scientist", "om"): (
            "* **Hordoffii Saatelayitii:** Daataa saatelayitii MODIS NDVI fi GRACE torban torbaniin wal-bira qabaa ilaalaa.\n"
            "* **Qorannoo Geengoo Muka:** Damee qorannoo geengoo mukaatiin haala deebisanii dandeettii carbo-hayidireetii qoradhaa."
        ),
        (2, "scientist", "en"): (
            "* **Gravimetric & Soil Telemetry:** Cross-validate FRADSCR SPEI projections against GRACE satellite terrestrial water storage (TWS) anomalies.\n"
            "* **Dendroclimatic Sampling:** Deploy high-resolution micro-coring on high-altitude *Juniperus procera* stands to monitor xylogenesis and cambial cessation.\n"
            "* **Ensemble Dispersion Monitoring:** Track ensemble spread across Random Forest and XGBoost components as solar transition inflection approaches."
        ),
        # Normal or Moderate (Class 0/1)
        (0, "minister", "am"): (
            "* **መደበኛ ስራዎችን ማስቀጠል:** የውሃ አስተዳደር ስራዎችን በመደበኛ አሰራር ያከናውኑ (የፓምፕ ምክር: {pump_hrs} ሰዓት/በቀን)።\n"
            "* **የውሃ ማቆሪያዎችን ማዘጋጀት:** በቀጣይ ዓመታት ሊከሰት ለሚችል የድርቅ ዑደት የዝናብ ውሃ ማሰባሰቢያ ኩሬዎችን እና የጎርፍ መከላከያዎችን ይገንቡ።"
        ),
        (0, "minister", "om"): (
            "* **Hojii Idilee:** Hojii bishaanii haala idileen itti fufaa (Paampii Aduu: {pump_hrs} sa'aatii/guyyaatti).\n"
            "* **Bishaan Roobaa Kuusuu:** Hoongee waggoota dhufaniif qophaa'uuf bishaan roobaa haroowwan keessatti kuusaa."
        ),
        (0, "minister", "en"): (
            "* **Standard Operational Cycling:** Maintain standard pumping schedules ({pump_hrs} hrs/day) with continuous water-table monitoring.\n"
            "* **Aquifer Recharge Investment:** Exploit favorable soil moisture conditions to construct artificial recharge structures and sand dams in ephemeral stream beds.\n"
            "* **Early Reserve Accumulation:** Replenish decentralized communal grain banks while regional crop yields remain elevated."
        ),
    }

    @classmethod
    def _build_recommendations(
        cls,
        pred_class: int,
        persona: str,
        language: str,
        hydro: Dict[str, Any],
        prescriptive_action: str,
    ) -> str:
        """Build persona- and language-tailored actionable recommendations using declarative dispatch."""
        class_key = 2 if pred_class == 2 else 0
        pump_default = 6.0 if pred_class == 2 else 8.0
        pump_hrs = hydro.get("recommended_solar_pumping_hours", pump_default)

        # Lookup by (class, persona, language) with graceful fallback to English or default
        template = (
            cls.RECOMMENDATION_REGISTRY.get((class_key, persona, language))
            or cls.RECOMMENDATION_REGISTRY.get((class_key, persona, "en"))
            or cls.RECOMMENDATION_REGISTRY.get((class_key, "minister", language))
            or cls.RECOMMENDATION_REGISTRY[(class_key, "minister", "en")]
        )
        return template.format(pump_hrs=pump_hrs)
