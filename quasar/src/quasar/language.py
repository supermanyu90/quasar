"""Multilingual assistance, and the two-tier policy that keeps it safe.

The hard question in a multilingual venue is not "can we translate?" -- it is
"what happens the one time the translation is wrong?" For a menu, nothing. For
"leave by Gate 4", someone dies. A single translation pipeline cannot be right
for both, so Quasar does not have one.

**Tier 1 -- safety critical** (evacuation, medical, crowd movement, weather hold).
No generative translation. Ever. The model's only authority is to choose a
template id from the controlled catalogue and fill *typed* slots -- a gate id
drawn from an enumeration, an integer, a zone name. The sentence itself is
human-written and human-validated per language, and the renderer refuses to emit
any language whose entry is not marked ``HUMAN_VALIDATED``, even if a
machine-drafted string for that language exists in the catalogue. There is no
free-text field anywhere on the Tier-1 path -- see the BROADCAST action in
:mod:`quasar.schemas`, which has no ``message`` property for a model to write into.

When a spectator's language has no validated entry, the system does not machine
translate and does not go silent. It falls back to the validated languages,
raises the corresponding pictogram, and dispatches a steward to the zone
(SOP-COMMS-07#3). That is a worse outcome than a perfect translation and a much
better one than a confident mistranslation of an evacuation order.

**Tier 2 -- informational** (wayfinding, catering, lost property, match info).
Generative translation is allowed, behind two automatic gates: a round-trip
back-translation similarity check, and a named-entity preservation check that
verifies every gate number, seat, block letter and time survives the translation
unaltered -- the failure mode that actually bites in practice is not
ungrammatical Marathi, it is "Gate 4" quietly becoming "Gate 7".

The shipped catalogue is human-validated in English, Hindi and Marathi. The
remaining languages are served on the Tier-2 path and are listed here as
machine drafts precisely so the Tier-1 gate has something real to refuse: a
draft that exists is more dangerous than one that does not, because it looks
ready. Validating them is a translation-vendor task, not an engineering one, and
the code is structured so that landing a validated entry is a data change with
no code change.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Sequence

from quasar.types import LangCode


class Tier(str, Enum):
    SAFETY_CRITICAL = "safety_critical"
    INFORMATIONAL = "informational"


class ReviewStatus(str, Enum):
    HUMAN_VALIDATED = "human_validated"  # certified by a qualified translator
    MACHINE_DRAFT = "machine_draft"  # generated, not certified -- Tier-1 blocked


@dataclass(frozen=True, slots=True)
class Language:
    code: LangCode
    name: str
    endonym: str


SUPPORTED_LANGUAGES: Mapping[LangCode, Language] = {
    lang.code: lang
    for lang in (
        Language("en", "English", "English"),
        Language("hi", "Hindi", "हिन्दी"),
        Language("mr", "Marathi", "मराठी"),
        Language("bn", "Bengali", "বাংলা"),
        Language("ta", "Tamil", "தமிழ்"),
        Language("te", "Telugu", "తెలుగు"),
        Language("kn", "Kannada", "ಕನ್ನಡ"),
        Language("gu", "Gujarati", "ગુજરાતી"),
        Language("pa", "Punjabi", "ਪੰਜਾਬੀ"),
        Language("ur", "Urdu", "اردو"),
        Language("fr", "French", "Français"),
        Language("es", "Spanish", "Español"),
        Language("ar", "Arabic", "العربية"),
        Language("ja", "Japanese", "日本語"),
    )
}

# Pictograms carry the Tier-1 fallback when a language has no validated entry.
PICTOGRAMS: Mapping[str, str] = {
    "MSG-MED-CORRIDOR": "PICTO-MEDICAL-CORRIDOR-CLOSED",
    "MSG-EVAC-GATE": "PICTO-EXIT-THIS-WAY",
    "MSG-GATE-DIVERT": "PICTO-GATE-CLOSED-USE-ALTERNATIVE",
    "MSG-WX-HOLD": "PICTO-SEEK-SHELTER",
}


class SlotKind(str, Enum):
    GATE = "gate"  # must name a real gate
    EDGE = "edge"  # must name a real corridor
    ZONE = "zone"  # must name a real zone
    INTEGER = "integer"


@dataclass(frozen=True, slots=True)
class Slot:
    name: str
    kind: SlotKind


@dataclass(frozen=True, slots=True)
class Rendering:
    text: str
    status: ReviewStatus


@dataclass(frozen=True, slots=True)
class Template:
    id: str
    tier: Tier
    slots: tuple[Slot, ...]
    renderings: Mapping[LangCode, Rendering]

    def validated_languages(self) -> tuple[LangCode, ...]:
        return tuple(
            code
            for code, r in self.renderings.items()
            if r.status is ReviewStatus.HUMAN_VALIDATED
        )


CATALOGUE: Mapping[str, Template] = {
    "MSG-MED-CORRIDOR": Template(
        id="MSG-MED-CORRIDOR",
        tier=Tier.SAFETY_CRITICAL,
        slots=(Slot("zone", SlotKind.ZONE), Slot("corridor", SlotKind.EDGE)),
        renderings={
            "en": Rendering(
                "Medical emergency in the {zone}. The {corridor} corridor is closed. "
                "Please keep it clear for medical staff and follow steward instructions.",
                ReviewStatus.HUMAN_VALIDATED,
            ),
            "hi": Rendering(
                "{zone} में चिकित्सा आपातकाल। {corridor} गलियारा बंद है। "
                "कृपया इसे चिकित्सा कर्मियों के लिए खाली रखें और सुरक्षा कर्मचारियों के निर्देशों का पालन करें।",
                ReviewStatus.HUMAN_VALIDATED,
            ),
            "mr": Rendering(
                "{zone} मध्ये वैद्यकीय आणीबाणी. {corridor} मार्गिका बंद आहे. "
                "कृपया ती वैद्यकीय कर्मचाऱ्यांसाठी मोकळी ठेवा आणि सुरक्षा कर्मचाऱ्यांच्या सूचनांचे पालन करा.",
                ReviewStatus.HUMAN_VALIDATED,
            ),
            # Machine drafts, deliberately shipped unvalidated. The Tier-1 gate
            # refuses them; test_language.py asserts that it does.
            "ta": Rendering(
                "{zone} பகுதியில் மருத்துவ அவசரநிலை. {corridor} நடைபாதை மூடப்பட்டுள்ளது. "
                "மருத்துவ ஊழியர்களுக்காக அதைக் காலியாக வைக்கவும்.",
                ReviewStatus.MACHINE_DRAFT,
            ),
            "bn": Rendering(
                "{zone} এলাকায় চিকিৎসা জরুরি অবস্থা। {corridor} করিডোর বন্ধ। "
                "অনুগ্রহ করে চিকিৎসা কর্মীদের জন্য পথ খালি রাখুন।",
                ReviewStatus.MACHINE_DRAFT,
            ),
        },
    ),
    "MSG-EVAC-GATE": Template(
        id="MSG-EVAC-GATE",
        tier=Tier.SAFETY_CRITICAL,
        slots=(Slot("gate", SlotKind.GATE),),
        renderings={
            "en": Rendering(
                "Please leave the stand calmly using Gate {gate}. Do not run. "
                "Follow the green exit signs and steward instructions.",
                ReviewStatus.HUMAN_VALIDATED,
            ),
            "hi": Rendering(
                "कृपया गेट {gate} से शांतिपूर्वक बाहर निकलें। दौड़ें नहीं। "
                "हरे निकास चिह्नों और सुरक्षा कर्मचारियों के निर्देशों का पालन करें।",
                ReviewStatus.HUMAN_VALIDATED,
            ),
            "mr": Rendering(
                "कृपया गेट {gate} मधून शांतपणे बाहेर पडा. धावू नका. "
                "हिरव्या निर्गम चिन्हांचे आणि सुरक्षा कर्मचाऱ्यांच्या सूचनांचे पालन करा.",
                ReviewStatus.HUMAN_VALIDATED,
            ),
        },
    ),
    "MSG-GATE-DIVERT": Template(
        id="MSG-GATE-DIVERT",
        tier=Tier.SAFETY_CRITICAL,  # crowd movement instruction: SOP-QUEUE-02#3
        slots=(Slot("from_gate", SlotKind.GATE), Slot("to_gate", SlotKind.GATE)),
        renderings={
            "en": Rendering(
                "Gate {from_gate} is very busy. Please use Gate {to_gate} instead. "
                "Stewards at Gate {to_gate} will help you.",
                ReviewStatus.HUMAN_VALIDATED,
            ),
            "hi": Rendering(
                "गेट {from_gate} पर बहुत भीड़ है। कृपया इसके बजाय गेट {to_gate} का उपयोग करें। "
                "गेट {to_gate} पर कर्मचारी आपकी मदद करेंगे।",
                ReviewStatus.HUMAN_VALIDATED,
            ),
            "mr": Rendering(
                "गेट {from_gate} वर खूप गर्दी आहे. कृपया त्याऐवजी गेट {to_gate} वापरा. "
                "गेट {to_gate} वरील कर्मचारी तुम्हाला मदत करतील.",
                ReviewStatus.HUMAN_VALIDATED,
            ),
        },
    ),
    "MSG-WX-HOLD": Template(
        id="MSG-WX-HOLD",
        tier=Tier.SAFETY_CRITICAL,
        slots=(),
        renderings={
            "en": Rendering(
                "Severe weather. Please move to the covered concourse and wait for the "
                "next announcement. Do not remain in open seating.",
                ReviewStatus.HUMAN_VALIDATED,
            ),
            "hi": Rendering(
                "खराब मौसम। कृपया ढके हुए कॉनकोर्स में जाएँ और अगली घोषणा की प्रतीक्षा करें। "
                "खुली सीटों पर न रहें।",
                ReviewStatus.HUMAN_VALIDATED,
            ),
            "mr": Rendering(
                "खराब हवामान. कृपया आच्छादित कॉनकोर्समध्ये जा आणि पुढील घोषणेची वाट पाहा. "
                "उघड्या आसनांवर थांबू नका.",
                ReviewStatus.HUMAN_VALIDATED,
            ),
        },
    ),
}


class CatalogueError(Exception):
    """A Tier-1 render was refused. Never swallowed: the caller must fall back."""


@dataclass(frozen=True, slots=True)
class Announcement:
    template_id: str
    language: LangCode
    text: str
    status: ReviewStatus


@dataclass(frozen=True, slots=True)
class Dispatch:
    """The result of asking for a safety-critical announcement in N languages."""

    template_id: str
    announcements: tuple[Announcement, ...]
    refused_languages: tuple[LangCode, ...]
    pictogram: str | None
    steward_required: bool

    @property
    def languages(self) -> tuple[LangCode, ...]:
        return tuple(a.language for a in self.announcements)


def _validate_slots(
    template: Template,
    slots: Mapping[str, object],
    *,
    known_gates: frozenset[str],
    known_edges: frozenset[str],
    known_zones: frozenset[str],
) -> dict[str, str]:
    """Type- and existence-check every slot before it enters a public announcement.

    This is the second half of the no-free-text guarantee. The schema stops the
    model writing a sentence; this stops it writing an *entity* -- announcing a
    gate that does not exist, or a corridor in another stadium.

    Returns the *internal ids*, which the caller then renders to public labels.
    """
    expected = {s.name for s in template.slots}
    provided = set(slots)
    if missing := expected - provided:
        raise CatalogueError(f"{template.id}: missing slots {sorted(missing)}")
    if extra := provided - expected:
        raise CatalogueError(f"{template.id}: unexpected slots {sorted(extra)}")

    resolved: dict[str, str] = {}
    for slot in template.slots:
        value = slots[slot.name]
        match slot.kind:
            case SlotKind.INTEGER:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise CatalogueError(f"{template.id}.{slot.name}: expected an integer")
                resolved[slot.name] = str(value)
            case SlotKind.GATE:
                if not isinstance(value, str) or value not in known_gates:
                    raise CatalogueError(f"{template.id}.{slot.name}: {value!r} is not a gate")
                resolved[slot.name] = value
            case SlotKind.EDGE:
                if not isinstance(value, str) or value not in known_edges:
                    raise CatalogueError(f"{template.id}.{slot.name}: {value!r} is not a corridor")
                resolved[slot.name] = value
            case SlotKind.ZONE:
                if not isinstance(value, str) or value not in known_zones:
                    raise CatalogueError(f"{template.id}.{slot.name}: {value!r} is not a zone")
                resolved[slot.name] = value
    return resolved


class MessageCatalogue:
    """Renders controlled messages. The only path to the PA system and push.

    ``labels`` maps an internal id to its public, per-language name. Announcements
    are rendered with the label; the audit log keeps the id. A spectator has never
    heard of "CORR-NE" -- they are looking at a wall that says "North-East
    Corridor", and that is what the announcement must say.
    """

    def __init__(
        self,
        *,
        known_gates: frozenset[str],
        known_edges: frozenset[str],
        known_zones: frozenset[str],
        labels: Mapping[str, Mapping[LangCode, str]] | None = None,
        templates: Mapping[str, Template] = CATALOGUE,
    ) -> None:
        self._templates = templates
        self._gates = known_gates
        self._edges = known_edges
        self._zones = known_zones
        self._labels = labels or {}

    def _label(self, value: str, language: LangCode) -> str:
        by_lang = self._labels.get(value)
        if not by_lang:
            return value
        return by_lang.get(language) or by_lang.get("en") or value

    def template(self, template_id: str) -> Template:
        t = self._templates.get(template_id)
        if t is None:
            raise CatalogueError(f"unknown template {template_id!r}")
        return t

    def render(
        self, template_id: str, language: LangCode, slots: Mapping[str, object]
    ) -> Announcement:
        """Render one language. Raises CatalogueError if Tier-1 policy forbids it."""
        template = self.template(template_id)
        resolved = _validate_slots(
            template,
            slots,
            known_gates=self._gates,
            known_edges=self._edges,
            known_zones=self._zones,
        )

        rendering = template.renderings.get(language)
        if rendering is None:
            raise CatalogueError(
                f"{template_id}: no catalogue entry for language {language!r}"
            )
        if (
            template.tier is Tier.SAFETY_CRITICAL
            and rendering.status is not ReviewStatus.HUMAN_VALIDATED
        ):
            raise CatalogueError(
                f"{template_id}: the {language!r} entry is {rendering.status.value}; "
                "a safety-critical announcement may not be made from an unvalidated "
                "translation (SOP-COMMS-07#1)"
            )

        public = {
            name: self._label(value, language) for name, value in resolved.items()
        }
        return Announcement(
            template_id=template_id,
            language=language,
            text=rendering.text.format(**public),
            status=rendering.status,
        )

    def dispatch(
        self, template_id: str, languages: Sequence[LangCode], slots: Mapping[str, object]
    ) -> Dispatch:
        """Render for every requested language, applying the SOP-COMMS-07#3 fallback.

        Refusing a language is not a failure of the dispatch: the announcement
        still goes out in the languages that *are* validated, with a pictogram
        and a steward, and the refusal is recorded so the gap is visible to
        operations rather than silently absorbed.
        """
        template = self.template(template_id)
        announcements: list[Announcement] = []
        refused: list[LangCode] = []

        for language in languages:
            try:
                announcements.append(self.render(template_id, language, slots))
            except CatalogueError:
                refused.append(language)

        if not announcements:
            # Not even English rendered: the template or slots are broken, and
            # broadcasting nothing is the correct, loud failure.
            raise CatalogueError(
                f"{template_id}: no requested language could be rendered ({list(languages)})"
            )

        return Dispatch(
            template_id=template_id,
            announcements=tuple(announcements),
            refused_languages=tuple(refused),
            pictogram=PICTOGRAMS.get(template_id) if refused else None,
            steward_required=bool(refused) and template.tier is Tier.SAFETY_CRITICAL,
        )


# ==========================================================================
# Tier-2: machine translation behind automatic quality gates.
# ==========================================================================

# What must survive a translation, and what must not be required to.
#
# The naive check -- "every entity string appears verbatim in the translation" --
# is wrong, and wrong in the direction that breaks the product: "Gate 4" rendered
# in Marathi is "गेट ४", in which the word "Gate" is *correctly* absent. Demanding
# it survive would reject every good Indic translation and pass an English one
# that said the wrong thing.
#
# What actually has to be invariant is narrower and sharper:
#
#   * ASCII codes  (CORR-NE, MED-2, G3) -- venue identifiers, never translated;
#   * numerals     (4, 19:40)           -- the gate number, the seat, the time;
#   * block/row/seat designators        -- the letter or number after the noun.
#
# The noun in front of the numeral is exactly the part a translator is *supposed*
# to change. So we check the numeral and the code, and let the words move.
_CODE = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b")  # CORR-NE, SOP-MED-03, MED-2
_GATE_ID = re.compile(r"\bG\d+\b")  # G3
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")  # 19:40
_DESIGNATOR = re.compile(
    r"\b(?:Gate|Block|Row|Seat|Stand|Turnstile)\s*([A-Z0-9]{1,4})\b", re.IGNORECASE
)
_BARE_NUMBER = re.compile(r"\b\d+\b")

# Indic and Arabic-Indic digits map back to ASCII before the comparison, so a
# translation that renders "4" as "४" still passes -- it is not an error.
_DIGIT_MAP = {
    ord(c): str(i % 10)
    for i, c in enumerate(
        "०१२३४५६७८९"  # Devanagari
        "০১২৩৪৫৬৭৮৯"  # Bengali
        "௦௧௨௩௪௫௬௭௮௯"  # Tamil
        "٠١٢٣٤٥٦٧٨٩"  # Arabic-Indic
    )
}


def normalise_digits(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_DIGIT_MAP)


@dataclass(frozen=True, slots=True)
class Entity:
    """A span of the source that must survive translation, and the token that
    carries its meaning."""

    display: str  # what a human sees in the log: "Gate 4"
    needle: str  # what must actually appear in the translation: "4"
    verbatim: bool  # True for codes (must appear as-is), False for numerals


def extract_entities(text: str) -> tuple[Entity, ...]:
    src = normalise_digits(text)
    entities: list[Entity] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(span: tuple[int, int]) -> bool:
        return any(span[0] < e and span[1] > s for s, e in consumed)

    for pattern in (_CODE, _GATE_ID, _TIME):
        for m in pattern.finditer(src):
            if overlaps(m.span()):
                continue
            consumed.append(m.span())
            entities.append(Entity(m.group(0), m.group(0), verbatim=True))

    for m in _DESIGNATOR.finditer(src):
        if overlaps(m.span()):
            continue
        consumed.append(m.span())
        entities.append(Entity(m.group(0), m.group(1), verbatim=False))

    for m in _BARE_NUMBER.finditer(src):
        if overlaps(m.span()):
            continue
        consumed.append(m.span())
        entities.append(Entity(m.group(0), m.group(0), verbatim=False))

    return tuple(entities)


def entities_preserved(source: str, translation: str) -> tuple[bool, tuple[str, ...]]:
    """True if every venue identifier and numeral in ``source`` survives translation.

    This is the check that catches "Gate 4" -> "Gate 7". No fluency metric will
    ever flag that: the mistranslation is perfectly fluent, perfectly grammatical,
    and sends the fan to the wrong side of the stadium.

    It deliberately does NOT require the surrounding words to survive -- "गेट ४"
    is a correct rendering of "Gate 4" and must pass.
    """
    tgt = normalise_digits(translation)
    tgt_alnum = re.sub(r"[^A-Za-z0-9]", "", tgt).upper()

    lost: list[str] = []
    for entity in extract_entities(source):
        if entity.verbatim:
            needle = re.sub(r"[^A-Za-z0-9]", "", entity.needle).upper()
            if needle and needle not in tgt_alnum:
                lost.append(entity.display)
        else:
            # The designator's token must appear as a standalone token, so that
            # "4" is not satisfied by the "4" inside "14".
            if not re.search(rf"(?<![A-Za-z0-9]){re.escape(entity.needle)}(?![A-Za-z0-9])", tgt, re.IGNORECASE):
                lost.append(entity.display)

    return (not lost), tuple(lost)


def char_ngram_similarity(a: str, b: str, *, n: int = 3) -> float:
    """chrF-style character n-gram F1. Script-agnostic, no model required."""
    def grams(text: str) -> set[str]:
        t = re.sub(r"\s+", " ", normalise_digits(text).strip().lower())
        return {t[i : i + n] for i in range(max(0, len(t) - n + 1))}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    overlap = len(ga & gb)
    if overlap == 0:
        return 0.0
    precision = overlap / len(gb)
    recall = overlap / len(ga)
    return 2.0 * precision * recall / (precision + recall)


class Translator(Protocol):
    """Any MT engine. In production this is the edge NMT model on the venue LAN."""

    def translate(self, text: str, *, source: LangCode, target: LangCode) -> str: ...


@dataclass(frozen=True, slots=True)
class TranslationResult:
    ok: bool
    text: str
    language: LangCode
    round_trip_similarity: float
    lost_entities: tuple[str, ...]
    reason: str = ""


# Round-trip similarity below this is treated as a failed translation. Tuned to
# be permissive about paraphrase and strict about meaning loss; the entity check
# does the precise work, this catches gross corruption.
ROUND_TRIP_THRESHOLD: float = 0.45


def translate_informational(
    translator: Translator,
    text: str,
    *,
    source: LangCode = "en",
    target: LangCode,
    threshold: float = ROUND_TRIP_THRESHOLD,
) -> TranslationResult:
    """Tier-2 translation with a round-trip and an entity-preservation gate.

    On failure the caller shows the source-language text rather than a suspect
    translation -- for informational content, an English answer a fan can
    partially read beats a fluent Marathi answer that names the wrong gate.
    """
    if target not in SUPPORTED_LANGUAGES:
        return TranslationResult(False, text, source, 0.0, (), f"unsupported language {target!r}")
    if target == source:
        return TranslationResult(True, text, target, 1.0, ())

    translated = translator.translate(text, source=source, target=target)
    back = translator.translate(translated, source=target, target=source)

    similarity = char_ngram_similarity(text, back)
    preserved, lost = entities_preserved(text, translated)

    if not preserved:
        return TranslationResult(
            False, text, source, similarity, lost,
            f"named entities lost in translation to {target!r}: {', '.join(lost)}",
        )
    if similarity < threshold:
        return TranslationResult(
            False, text, source, similarity, (),
            f"round-trip similarity {similarity:.2f} below threshold {threshold:.2f}",
        )
    return TranslationResult(True, translated, target, similarity, ())
