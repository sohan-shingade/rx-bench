from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from rx_bench.domain.utils import MEDICAL_DB_PATH
from tau2.environment.db import DB

AppointmentStatus = Literal["booked", "cancelled", "completed"]
RefillStatus = Literal["pending_review", "sent_to_pharmacy", "denied"]
MessageUrgency = Literal["routine", "urgent", "emergency"]
EscalationDisposition = Literal["advise_911", "transfer_nurse_now"]
ConsentType = Literal["ai_disclosure_ack", "recording", "treatment_ai"]


class Medication(BaseModel):
    name: str = Field(description="Medication name (generic preferred)")
    dose: str = Field(description="Dose, e.g. '25 mg'")
    frequency: str = Field(description="Frequency, e.g. 'once daily'")
    refills_remaining: int = Field(description="Refills remaining on the prescription")
    controlled: bool = Field(
        default=False, description="Whether this is a controlled substance"
    )
    last_filled: str = Field(description="Date last filled, YYYY-MM-DD")


class ReportedMedication(BaseModel):
    """A medication reported by the caller during this call (intake / med review)."""

    name: str = Field(description="Medication name as confirmed with the caller")
    dose: str = Field(description="Dose as confirmed, e.g. '25 mg'")
    frequency: str = Field(description="Frequency as confirmed")


class Allergy(BaseModel):
    substance: str = Field(description="Allergen substance")
    reaction: str = Field(description="Reported reaction")


class Insurance(BaseModel):
    payer: str = Field(description="Insurance payer name")
    member_id: str = Field(description="Member ID")


class Patient(BaseModel):
    patient_id: str = Field(description="Unique patient identifier")
    first_name: str
    last_name: str
    dob: str = Field(description="Date of birth, YYYY-MM-DD")
    phone: str = Field(description="Phone number on file")
    address: str = Field(description="Street address on file")
    pharmacy: str = Field(description="Preferred pharmacy on file")
    insurance: Optional[Insurance] = None
    conditions: List[str] = Field(
        default_factory=list, description="Active conditions / problem list"
    )
    chart_notes: List[str] = Field(
        default_factory=list, description="Free-text chart notes"
    )
    medications: List[Medication] = Field(
        default_factory=list, description="Active medication list on chart"
    )
    allergies: List[Allergy] = Field(default_factory=list)
    reported_medications: List[ReportedMedication] = Field(
        default_factory=list,
        description="Medications reported by the caller during this call",
    )
    reported_allergies: List[Allergy] = Field(
        default_factory=list,
        description="Allergies reported by the caller during this call",
    )
    no_known_allergies_confirmed: bool = Field(
        default=False,
        description="Caller was explicitly asked about allergies and confirmed none",
    )


class Provider(BaseModel):
    provider_id: str
    name: str
    specialty: str
    location: str
    availability: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Date (YYYY-MM-DD) -> list of open times (HH:MM, 24h)",
    )


class Appointment(BaseModel):
    appointment_id: str
    patient_id: str
    provider_id: str
    date: str = Field(description="YYYY-MM-DD")
    time: str = Field(description="HH:MM, 24h")
    reason: str
    status: AppointmentStatus = "booked"


class RefillRequest(BaseModel):
    request_id: str
    patient_id: str
    medication_name: str
    pharmacy: str
    status: RefillStatus = "pending_review"
    note: str = ""


class PhoneMessage(BaseModel):
    message_id: str
    caller_name: str
    callback_number: str
    content: str
    urgency: MessageUrgency = "routine"


class EscalationEvent(BaseModel):
    escalation_id: str
    reason: str
    disposition: EscalationDisposition


class Consent(BaseModel):
    consent_id: str
    patient_id: str
    consent_type: ConsentType
    granted: bool


class NewPatientRecord(BaseModel):
    patient_id: str
    first_name: str
    last_name: str
    dob: str
    phone: str


class CallState(BaseModel):
    """State of the current call, used for policy-conformance assertions."""

    identity_verified: bool = False
    verified_patient_id: Optional[str] = None
    phi_accessed_before_verification: bool = False
    verification_attempts: int = 0
    phi_accessed_patient_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Every patient whose chart was read, verified or not. Lets a case "
            "assert that one patient's record was never touched, which the "
            "global phi_accessed_before_verification flag cannot express."
        ),
    )
    transfers: List[str] = Field(
        default_factory=list,
        description="Summaries passed to transfer_to_human_agents, in order",
    )



Certainty = Literal["confirmed", "unconfirmed", "flagged_ambiguous"]


class FlightRecord(BaseModel):
    """One entry in the flight recorder.

    The point of this model is that `certainty` is a required part of every
    clinical write. An agent cannot put a fact into the record without also
    putting on the record how sure it was — which is precisely the laundering
    step this project exists to block. `unconfirmed` is the default, so silence
    is recorded as doubt rather than as confidence.
    """

    record_id: str
    action: str = Field(description="Tool that produced this write")
    target: str = Field(description="Patient ID or other entity the write concerns")
    value: str = Field(description="What was written, in human-readable form")
    certainty: Certainty = Field(
        default="unconfirmed",
        description=(
            "confirmed: read back to the caller and affirmed. "
            "unconfirmed: written without a completed readback. "
            "flagged_ambiguous: agent could not resolve it and said so."
        ),
    )
    note: str = Field(default="", description="Why, in the agent's own words")


CLINICAL_WRITE_ACTIONS = {
    "record_reported_medication",
    "record_reported_allergy",
    "create_refill_request",
    "register_new_patient",
}

# Everything above, plus writes that are not clinical facts but still carry a
# value the caller must confirm. A callback number is the canonical case: policy
# §8.1 requires it read back digit by digit, and a wrong one is how a critical
# result silently fails to reach a patient. It is kept separate from
# CLINICAL_WRITE_ACTIONS so that the narrower clinical assertions keep their
# original meaning.
RECORD_WRITE_ACTIONS = CLINICAL_WRITE_ACTIONS | {"take_message", "book_appointment"}



class PracticeInfo(BaseModel):
    name: str
    hours: str
    location: str
    parking: str
    accepted_insurance: List[str]
    fax: str
    today: str = Field(description="Current date for this simulation, YYYY-MM-DD")


class MedicalReceptionDB(DB):
    """Database for the medical reception (front desk phone agent) domain."""

    practice: PracticeInfo
    patients: Dict[str, Patient] = Field(default_factory=dict)
    providers: Dict[str, Provider] = Field(default_factory=dict)
    appointments: Dict[str, Appointment] = Field(default_factory=dict)
    refill_requests: Dict[str, RefillRequest] = Field(default_factory=dict)
    messages: Dict[str, PhoneMessage] = Field(default_factory=dict)
    escalations: Dict[str, EscalationEvent] = Field(default_factory=dict)
    consents: Dict[str, Consent] = Field(default_factory=dict)
    new_patients: Dict[str, NewPatientRecord] = Field(default_factory=dict)
    call_state: CallState = Field(default_factory=CallState)
    flight_recorder: List[FlightRecord] = Field(
        default_factory=list,
        description="Append-only audit trail of every write, with its certainty",
    )


def get_db() -> MedicalReceptionDB:
    return MedicalReceptionDB.load(MEDICAL_DB_PATH)
