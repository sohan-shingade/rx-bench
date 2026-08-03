from datetime import datetime

from rx_bench.domain.data_model import MedicalReceptionDB
from rx_bench.domain.environment import get_environment
from rx_bench.domain.medplum import (
    build_transaction_bundle,
    fhir_appointment,
    fhir_audit_event,
    fhir_risk_assessment,
)
from rx_bench.domain.utils import MEDICAL_DB_PATH


def test_supplied_db_is_deep_copied_between_environments() -> None:
    baseline = MedicalReceptionDB.load(MEDICAL_DB_PATH)
    first = get_environment(baseline)
    second = get_environment(baseline)

    first.tools.db.call_state.identity_verified = True
    first.tools.db.appointments.clear()

    assert not baseline.call_state.identity_verified
    assert not second.tools.db.call_state.identity_verified
    assert baseline.appointments
    assert second.tools.db.appointments


def test_transaction_bundle_has_valid_linked_resources() -> None:
    appointment = fhir_appointment(
        "pt_001", "Dr. Rivera", "2026-08-04", "09:00", "Follow-up"
    )
    audit = fhir_audit_event("book_appointment", "pt_001", "Follow-up")
    risk = fhir_risk_assessment(
        "pt_001", "book_appointment", "Follow-up", "confirmed"
    )
    bundle = build_transaction_bundle(appointment, audit, risk)
    entries = bundle["entry"]
    resources = {entry["resource"]["resourceType"]: entry for entry in entries}

    assert bundle["type"] == "transaction"
    assert resources["Patient"]["request"]["ifNoneExist"] == "identifier=pt_001"
    assert datetime.fromisoformat(appointment["end"]) > datetime.fromisoformat(
        appointment["start"]
    )
    assert datetime.fromisoformat(resources["AuditEvent"]["resource"]["recorded"])

    patient_ref = resources["Patient"]["fullUrl"]
    clinical_ref = resources["Appointment"]["fullUrl"]
    assert patient_ref.startswith("urn:uuid:")
    assert resources["Appointment"]["resource"]["participant"][0]["actor"][
        "reference"
    ] == patient_ref
    assert resources["AuditEvent"]["resource"]["entity"][0]["what"][
        "reference"
    ] == clinical_ref
    assert resources["RiskAssessment"]["resource"]["basis"][0][
        "reference"
    ] == clinical_ref
    assert resources["RiskAssessment"]["resource"]["subject"][
        "reference"
    ] == patient_ref
