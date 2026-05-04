from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
import time
from typing import Dict, Iterable, List, Tuple


class MessageCategory(str, Enum):
    MEDICATION_REFILL = "Medication refill"
    APPOINTMENT_SCHEDULING = "Appointment scheduling"
    LAB_RESULTS_QUESTION = "Lab results question"
    URGENT_CLINICAL_CONCERN = "Urgent clinical concern"
    FORMS_ADMIN = "Forms / school notes"
    GENERAL = "General / needs review"


class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


@dataclass
class IntakeResult:
    original_text: str
    cleaned_text: str
    received_at: datetime


@dataclass
class ClassificationResult:
    category: MessageCategory
    confidence: float
    matched_terms: List[str] = field(default_factory=list)
    why_classified: str = ""


@dataclass
class RiskAssessmentResult:
    risk_level: RiskLevel
    escalation_triggers: List[str] = field(default_factory=list)
    safety_rationale: str = ""


@dataclass
class RoutingResult:
    route_to: str
    assigned_role: str
    risk_level: RiskLevel
    action: str
    response_draft: str
    requires_provider_review: bool
    auto_send_allowed: bool
    safety_note: str
    status_badge: str


@dataclass
class ProcessedMessage:
    message_id: int
    intake: IntakeResult
    classification: ClassificationResult
    risk_assessment: RiskAssessmentResult
    routing: RoutingResult
    completed_at: datetime
    turnaround_seconds: float


class MessageIntakeAgent:
    """Agent 1: receives patient messages and standardizes text for downstream agents."""

    def process(self, message: str) -> IntakeResult:
        cleaned = message.strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"[^\w\s.,!?'-]", "", cleaned)
        return IntakeResult(
            original_text=message,
            cleaned_text=cleaned,
            received_at=datetime.now(timezone.utc),
        )


class ClassificationAgent:
    """Agent 2: classifies messages using transparent rule-based triage logic."""

    KEYWORDS: Dict[MessageCategory, List[str]] = {
        MessageCategory.URGENT_CLINICAL_CONCERN: [
            "trouble breathing",
            "can't breathe",
            "cannot breathe",
            "difficulty breathing",
            "shortness of breath",
            "blue lips",
            "seizure",
            "unresponsive",
            "severe allergic",
            "allergic reaction",
            "rash after medicine",
            "hives after medicine",
            "chest pain",
            "high fever",
            "fever 104",
            "dehydrated",
            "wheezing",
            "vomiting all night",
            "hard to wake",
            "lethargic",
            "urgent",
        ],
        MessageCategory.MEDICATION_REFILL: [
            "refill",
            "prescription",
            "medicine",
            "medication",
            "needs more",
            "ran out",
            "inhaler",
            "adhd meds",
            "adhd medication",
            "antibiotic",
            "pharmacy",
        ],
        MessageCategory.APPOINTMENT_SCHEDULING: [
            "appointment",
            "schedule",
            "reschedule",
            "cancel",
            "checkup",
            "well child",
            "sick visit",
            "sports physical",
            "physical",
        ],
        MessageCategory.LAB_RESULTS_QUESTION: [
            "lab",
            "labs",
            "result",
            "results",
            "bloodwork",
            "test",
            "culture",
            "x-ray",
            "ultrasound",
            "imaging",
        ],
        MessageCategory.FORMS_ADMIN: [
            "school form",
            "school note",
            "note for school",
            "excuse note",
            "sports form",
            "physical form",
            "form",
            "daycare form",
            "camp form",
            "immunization record",
            "vaccine record",
        ],
    }

    def classify(self, intake: IntakeResult) -> ClassificationResult:
        text = intake.cleaned_text

        # Urgent clinical concern is intentionally checked first so safety issues
        # are not missed because of another keyword, such as medication or lab.
        priority_order = [
            MessageCategory.URGENT_CLINICAL_CONCERN,
            MessageCategory.MEDICATION_REFILL,
            MessageCategory.LAB_RESULTS_QUESTION,
            MessageCategory.FORMS_ADMIN,
            MessageCategory.APPOINTMENT_SCHEDULING,
        ]

        for category in priority_order:
            keywords = self.KEYWORDS[category]
            matches = [term for term in keywords if term in text]
            if matches:
                confidence = min(0.97, 0.68 + (0.08 * len(matches)))
                return ClassificationResult(
                    category=category,
                    confidence=round(confidence, 2),
                    matched_terms=matches,
                    why_classified=(
                        f"Detected terms {matches} matched the {category.value} workflow."
                    ),
                )

        return ClassificationResult(
            category=MessageCategory.GENERAL,
            confidence=0.4,
            matched_terms=[],
            why_classified="No strong category-specific keywords were detected, so the message was routed for manual review.",
        )


class RiskScoringAgent:
    """Agent 3: evaluates symptom combinations and escalates risk when needed."""

    HIGH_RISK_TERMS = [
        "trouble breathing",
        "can't breathe",
        "cannot breathe",
        "difficulty breathing",
        "shortness of breath",
        "blue lips",
        "unresponsive",
        "hard to wake",
        "lethargic",
        "seizure",
        "chest pain",
        "allergic reaction",
        "severe allergic",
        "wheezing",
    ]
    MODERATE_RISK_TERMS = [
        "fever",
        "vomiting",
        "diarrhea",
        "rash",
        "pain",
        "headache",
        "not improving",
        "getting worse",
        "after medicine",
        "after medication",
    ]
    INFANT_TERMS = ["newborn", "infant", "baby", "2 month", "3 month", "under 3 months"]

    def assess(self, intake: IntakeResult, classification: ClassificationResult) -> RiskAssessmentResult:
        text = intake.cleaned_text
        triggers: List[str] = []

        high_hits = [term for term in self.HIGH_RISK_TERMS if term in text]
        moderate_hits = [term for term in self.MODERATE_RISK_TERMS if term in text]
        infant_hits = [term for term in self.INFANT_TERMS if term in text]

        if high_hits:
            triggers.extend([f"High-risk symptom: {term}" for term in high_hits])
            return RiskAssessmentResult(
                risk_level=RiskLevel.HIGH,
                escalation_triggers=triggers,
                safety_rationale="High-risk symptom language detected. Immediate clinical review is required.",
            )

        # Combination rules make the demo more robust than simple keyword matching.
        if "fever" in text and ("vomiting" in text or "diarrhea" in text):
            triggers.append("Combination trigger: fever plus vomiting/diarrhea")
            return RiskAssessmentResult(
                risk_level=RiskLevel.HIGH,
                escalation_triggers=triggers,
                safety_rationale="Combined symptoms may represent dehydration or worsening illness and should be escalated.",
            )

        if infant_hits and "fever" in text:
            triggers.append("Pediatric safety trigger: fever in young infant")
            return RiskAssessmentResult(
                risk_level=RiskLevel.HIGH,
                escalation_triggers=triggers,
                safety_rationale="Fever in a young infant requires urgent clinical review.",
            )

        if classification.category in {
            MessageCategory.LAB_RESULTS_QUESTION,
            MessageCategory.GENERAL,
        }:
            if moderate_hits:
                triggers.extend([f"Moderate-risk term: {term}" for term in moderate_hits])
            return RiskAssessmentResult(
                risk_level=RiskLevel.MODERATE,
                escalation_triggers=triggers or ["Clinical interpretation or unclear request"],
                safety_rationale="Clinical interpretation or unclear messages should not be automated.",
            )

        if moderate_hits:
            triggers.extend([f"Moderate-risk term: {term}" for term in moderate_hits])
            return RiskAssessmentResult(
                risk_level=RiskLevel.MODERATE,
                escalation_triggers=triggers,
                safety_rationale="Symptoms were detected, so the message requires staff review before any response.",
            )

        return RiskAssessmentResult(
            risk_level=RiskLevel.LOW,
            escalation_triggers=[],
            safety_rationale="No clinical escalation terms detected. This appears administrative/low risk.",
        )


class RoutingResponseAgent:
    """Agent 4: routes messages and drafts safe responses for staff review."""

    def route(
        self,
        classification: ClassificationResult,
        risk_assessment: RiskAssessmentResult,
    ) -> RoutingResult:
        category = classification.category
        risk = risk_assessment.risk_level

        if risk == RiskLevel.HIGH:
            return RoutingResult(
                route_to="Provider urgent review queue",
                assigned_role="Nurse triage / provider immediately",
                risk_level=RiskLevel.HIGH,
                action="Flag as urgent and suppress automated clinical advice",
                response_draft=(
                    "This message may describe an urgent medical concern. It has been "
                    "flagged for immediate clinical review. If your child is having trouble "
                    "breathing, is unresponsive, has blue lips, or you are worried this is "
                    "an emergency, call 911 or go to the nearest emergency department."
                ),
                requires_provider_review=True,
                auto_send_allowed=False,
                safety_note="Do not send automatically. High-risk symptoms require immediate clinical review.",
                status_badge="🔴 Provider review required",
            )

        if risk == RiskLevel.MODERATE:
            assigned_role = "Nurse triage / provider" if category == MessageCategory.LAB_RESULTS_QUESTION else "Clinical support staff"
            route_to = "Nurse triage queue" if category == MessageCategory.LAB_RESULTS_QUESTION else "General clinical review queue"
            return RoutingResult(
                route_to=route_to,
                assigned_role=assigned_role,
                risk_level=RiskLevel.MODERATE,
                action="Route for staff review before response",
                response_draft=(
                    "Thanks for your message. It has been routed to the clinical team for review. "
                    "A staff member will review the details and follow up with appropriate next steps."
                ),
                requires_provider_review=True,
                auto_send_allowed=False,
                safety_note="Do not send automatically. Staff review is required because clinical context may be needed.",
                status_badge="🟡 Staff review required",
            )

        if category == MessageCategory.MEDICATION_REFILL:
            return RoutingResult(
                route_to="Clinical support pool",
                assigned_role="Medical assistant / provider review as needed",
                risk_level=RiskLevel.LOW,
                action="Draft refill response and request medication details",
                response_draft=(
                    "Your refill request has been routed to the medical assistant team. "
                    "Please confirm the medication name, dose, preferred pharmacy, and whether your child has missed any doses. "
                    "You can expect a follow-up after the clinical team reviews the request."
                ),
                requires_provider_review=False,
                auto_send_allowed=True,
                safety_note="Low-risk administrative draft. Staff should verify medication details before processing.",
                status_badge="🟢 Auto-draft allowed",
            )

        if category == MessageCategory.APPOINTMENT_SCHEDULING:
            return RoutingResult(
                route_to="Front desk scheduling queue",
                assigned_role="Front desk",
                risk_level=RiskLevel.LOW,
                action="Draft scheduling response",
                response_draft=(
                    "We can help with scheduling. Please share your preferred dates, times, "
                    "and reason for the visit. Our scheduling team will follow up with available options."
                ),
                requires_provider_review=False,
                auto_send_allowed=True,
                safety_note="Low-risk administrative draft. No medical advice is included.",
                status_badge="🟢 Auto-draft allowed",
            )

        if category == MessageCategory.FORMS_ADMIN:
            return RoutingResult(
                route_to="Administrative forms queue",
                assigned_role="Front desk / medical records staff",
                risk_level=RiskLevel.LOW,
                action="Route to forms queue and draft information request",
                response_draft=(
                    "Thanks for your message. Please confirm the form or note needed, the date of service if relevant, "
                    "and where it should be sent. Our team will review the request."
                ),
                requires_provider_review=False,
                auto_send_allowed=True,
                safety_note="Low-risk administrative draft. Staff should confirm authorization before sending records.",
                status_badge="🟢 Auto-draft allowed",
            )

        return RoutingResult(
            route_to="General patient message queue",
            assigned_role="Clinical support staff",
            risk_level=RiskLevel.MODERATE,
            action="Route for manual review",
            response_draft=(
                "Thanks for your message. It has been routed to our team for review, "
                "and we will follow up as soon as possible."
            ),
            requires_provider_review=True,
            auto_send_allowed=False,
            safety_note="Do not send automatically. Unclear requests require staff review.",
            status_badge="🟡 Staff review required",
        )


class MetricsLoggingAgent:
    """Agent 5: records workflow activity and calculates operational metrics."""

    ESTIMATED_MANUAL_MINUTES_PER_MESSAGE = 3
    ESTIMATED_AI_REVIEW_MINUTES_PER_MESSAGE = 1

    def __init__(self) -> None:
        self.messages: List[ProcessedMessage] = []

    def log(self, processed_message: ProcessedMessage) -> None:
        self.messages.append(processed_message)

    def summary(self) -> Dict[str, object]:
        category_counts = {category.value: 0 for category in MessageCategory}
        role_counts: Dict[str, int] = {}
        risk_counts = {risk.value: 0 for risk in RiskLevel}
        auto_send_count = 0
        review_required_count = 0

        for message in self.messages:
            category_counts[message.classification.category.value] += 1
            role_counts[message.routing.assigned_role] = role_counts.get(message.routing.assigned_role, 0) + 1
            risk_counts[message.routing.risk_level.value] += 1
            if message.routing.auto_send_allowed:
                auto_send_count += 1
            if message.routing.requires_provider_review:
                review_required_count += 1

        total = len(self.messages)
        avg_turnaround = (
            sum(message.turnaround_seconds for message in self.messages) / total
            if total
            else 0.0
        )
        manual_minutes = total * self.ESTIMATED_MANUAL_MINUTES_PER_MESSAGE
        ai_review_minutes = total * self.ESTIMATED_AI_REVIEW_MINUTES_PER_MESSAGE
        estimated_minutes_saved = max(0, manual_minutes - ai_review_minutes)
        percent_auto_draft = round((auto_send_count / total) * 100, 1) if total else 0.0
        percent_review = round((review_required_count / total) * 100, 1) if total else 0.0

        return {
            "total_messages_processed": total,
            "count_by_category": category_counts,
            "count_by_role": role_counts,
            "count_by_risk_level": risk_counts,
            "average_turnaround_seconds": round(avg_turnaround, 3),
            "urgent_messages_flagged": risk_counts[RiskLevel.HIGH.value],
            "review_required_count": review_required_count,
            "auto_send_allowed_count": auto_send_count,
            "percent_auto_draft": percent_auto_draft,
            "percent_review_required": percent_review,
            "estimated_admin_minutes_saved": estimated_minutes_saved,
            "estimated_manual_minutes_without_ai": manual_minutes,
            "estimated_minutes_with_ai_assist": ai_review_minutes,
        }


class PediatricClinicWorkflow:
    """Coordinates the agents into one end-to-end triage workflow."""

    def __init__(self) -> None:
        self.intake_agent = MessageIntakeAgent()
        self.classification_agent = ClassificationAgent()
        self.risk_agent = RiskScoringAgent()
        self.routing_agent = RoutingResponseAgent()
        self.metrics_agent = MetricsLoggingAgent()
        self._next_message_id = 1

    def process_message(self, message: str, simulate_delay: bool = True) -> ProcessedMessage:
        intake = self.intake_agent.process(message)
        classification = self.classification_agent.classify(intake)
        risk_assessment = self.risk_agent.assess(intake, classification)

        if simulate_delay:
            time.sleep(0.05)

        routing = self.routing_agent.route(classification, risk_assessment)
        completed_at = datetime.now(timezone.utc)
        turnaround = (completed_at - intake.received_at).total_seconds()

        processed = ProcessedMessage(
            message_id=self._next_message_id,
            intake=intake,
            classification=classification,
            risk_assessment=risk_assessment,
            routing=routing,
            completed_at=completed_at,
            turnaround_seconds=turnaround,
        )
        self._next_message_id += 1
        self.metrics_agent.log(processed)
        return processed

    def process_batch(
        self, messages: Iterable[str], simulate_delay: bool = True
    ) -> List[ProcessedMessage]:
        return [
            self.process_message(message, simulate_delay=simulate_delay)
            for message in messages
        ]

    def metrics_summary(self) -> Dict[str, object]:
        return self.metrics_agent.summary()


SAMPLE_MESSAGES = [
    "Can you refill Mateo's albuterol inhaler at the Walgreens on Main Street?",
    "I need to reschedule my daughter's well child appointment next week.",
    "We saw the bloodwork results in the portal. Can someone explain the lab results?",
    "My son is wheezing and having trouble breathing after eating peanut butter.",
    "Can you send a school note for yesterday's visit?",
    "My child has been vomiting all night and is hard to wake up.",
    "She needs more ADHD meds sent to CVS before the weekend.",
    "Can you complete the sports physical form for school?",
    "My child has a fever and has been vomiting since last night.",
    "My baby is 2 months old and has a fever.",
    "My child developed a rash after starting the new medication.",
]


# Real-world scalability notes:
# - In production, this workflow would sit behind authenticated patient messaging
#   channels and integrate with the clinic's EHR using standards such as FHIR,
#   HL7, SMART on FHIR apps, or vendor APIs for inbox, chart, medication, lab,
#   scheduling, and documentation workflows.
# - Response drafting should remain assistive: staff or providers should approve
#   clinical content before it reaches families, especially for symptoms, labs,
#   medication changes, or diagnosis-related questions.
# - HIPAA compliance would require encryption in transit and at rest, role-based
#   access controls, audit logs, minimum-necessary data handling, business
#   associate agreements, retention policies, and careful PHI controls for any
#   model or cloud service used in the workflow.
# - A cloud deployment could use managed queues, serverless workers or containers,
#   observability tooling, secrets management, model monitoring, and a human review
#   dashboard to scale safely across high message volumes and multiple locations.
# - Over time, the clinic could connect this agent workflow to operational KPIs,
#   such as message volume, average turnaround time, urgent-message escalation,
#   staff workload, refill completion time, and patient satisfaction trends.
