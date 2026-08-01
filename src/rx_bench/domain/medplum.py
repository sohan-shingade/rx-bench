"""Best-effort dual-write of domain writes into a Medplum FHIR project.

The tau2 DB remains the eval source of truth; every write tool also posts the
equivalent FHIR resource here so the demo (and the flight-recorder AuditEvents)
can be shown live inside Medplum.

Configuration via env vars:
    MEDPLUM_BASE_URL   e.g. https://api.medplum.com  (default)
    MEDPLUM_CLIENT_ID / MEDPLUM_CLIENT_SECRET   client-credentials pair

If credentials are missing or any request fails, writes are appended to a local
JSONL spool (data/tau2/domains/medical_reception/medplum_spool.jsonl) so
nothing is lost and eval runs are never blocked or slowed by the network.
"""

import json
import os
import threading
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

from rx_bench.domain.utils import MEDICAL_DATA_DIR

SPOOL_PATH = MEDICAL_DATA_DIR / "medplum_spool.jsonl"
_TIMEOUT = 5


class MedplumClient:
    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "MEDPLUM_BASE_URL", "https://api.medplum.com"
        ).rstrip("/")
        self.client_id = os.environ.get("MEDPLUM_CLIENT_ID")
        self.client_secret = os.environ.get("MEDPLUM_CLIENT_SECRET")
        self._token: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.client_id and self.client_secret) and os.environ.get(
            "MEDPLUM_DISABLE"
        ) != "1"

    def _get_token(self) -> Optional[str]:
        if self._token:
            return self._token
        try:
            resp = requests.post(
                f"{self.base_url}/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            self._token = resp.json()["access_token"]
            return self._token
        except Exception as e:
            logger.warning(f"Medplum auth failed, spooling writes: {e}")
            return None

    def _spool(self, resource: dict) -> None:
        try:
            SPOOL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SPOOL_PATH, "a") as fp:
                fp.write(json.dumps(resource) + "\n")
        except Exception as e:
            logger.warning(f"Medplum spool write failed: {e}")

    def create(self, resource: dict) -> None:
        """Fire-and-forget create of a FHIR resource. Never raises."""
        self._spool(resource)
        if not self.enabled:
            return
        with self._lock:
            token = self._get_token()
        if token is None:
            return
        try:
            resp = requests.post(
                f"{self.base_url}/fhir/R4/{resource['resourceType']}",
                json=resource,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/fhir+json",
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code >= 400:
                logger.warning(
                    f"Medplum create {resource['resourceType']} -> {resp.status_code}: {resp.text[:200]}"
                )
        except Exception as e:
            logger.warning(f"Medplum create failed (spooled locally): {e}")


_client = MedplumClient()


def _patient_ref(patient_id: str) -> dict:
    return {"reference": f"Patient?identifier={patient_id}", "display": patient_id}


def fhir_medication_statement(patient_id: str, name: str, dose: str, frequency: str) -> dict:
    return {
        "resourceType": "MedicationStatement",
        "status": "active",
        "medicationCodeableConcept": {"text": f"{name} {dose}"},
        "subject": _patient_ref(patient_id),
        "dosage": [{"text": f"{dose} {frequency}"}],
        "note": [{"text": "Reported by patient via phone intake (voice agent)"}],
    }


def fhir_allergy(patient_id: str, substance: str, reaction: str) -> dict:
    return {
        "resourceType": "AllergyIntolerance",
        "clinicalStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                    "code": "active",
                }
            ]
        },
        "code": {"text": substance},
        "patient": _patient_ref(patient_id),
        "reaction": [{"manifestation": [{"text": reaction}]}],
    }


def fhir_appointment(patient_id: str, provider_name: str, date: str, time: str, reason: str) -> dict:
    return {
        "resourceType": "Appointment",
        "status": "booked",
        "description": reason,
        "start": f"{date}T{time}:00-07:00",
        "participant": [
            {"actor": _patient_ref(patient_id), "status": "accepted"},
            {"actor": {"display": provider_name}, "status": "accepted"},
        ],
    }


def fhir_refill_task(patient_id: str, medication_name: str, pharmacy: str, note: str) -> dict:
    return {
        "resourceType": "Task",
        "status": "requested",
        "intent": "order",
        "code": {"text": "Prescription refill request"},
        "description": f"Refill {medication_name} to {pharmacy}. {note}".strip(),
        "for": _patient_ref(patient_id),
    }


def fhir_communication(caller_name: str, callback_number: str, content: str, urgency: str) -> dict:
    return {
        "resourceType": "Communication",
        "status": "completed",
        "priority": {"routine": "routine", "urgent": "urgent", "emergency": "stat"}[urgency],
        "sender": {"display": f"{caller_name} ({callback_number})"},
        "payload": [{"contentString": content}],
    }


def fhir_escalation_flag(reason: str, disposition: str) -> dict:
    return {
        "resourceType": "Flag",
        "status": "active",
        "code": {"text": f"EMERGENCY ESCALATION [{disposition}]: {reason}"},
        "subject": {"display": "current caller"},
    }


def fhir_consent(patient_id: str, consent_type: str, granted: bool) -> dict:
    return {
        "resourceType": "Consent",
        "status": "active" if granted else "rejected",
        "scope": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/consentscope", "code": "patient-privacy"}]},
        "category": [{"text": consent_type}],
        "patient": _patient_ref(patient_id),
    }


def fhir_new_patient(patient_id: str, first: str, last: str, dob: str, phone: str) -> dict:
    return {
        "resourceType": "Patient",
        "identifier": [{"value": patient_id}],
        "name": [{"given": [first], "family": last}],
        "birthDate": dob,
        "telecom": [{"system": "phone", "value": phone}],
    }


# ---------------------------------------------------------------------------
# Flight recorder
#
# Two resources per write. The AuditEvent answers "what did the agent do"; the
# RiskAssessment answers "how sure was it, and who confirmed". Splitting them
# matters: an audit trail that records only the action lets an unverified guess
# and a read-back confirmation look identical downstream, which is the exact
# failure this project is about.
# ---------------------------------------------------------------------------

_CERTAINTY_RISK = {
    "confirmed": ("low", "Read back to the caller and affirmed"),
    "unconfirmed": ("high", "Written without a completed readback"),
    "flagged_ambiguous": (
        "moderate",
        "Agent could not resolve the value and flagged it rather than guessing",
    ),
}


def fhir_audit_event(action: str, target: str, value: str, note: str = "") -> dict:
    """An AuditEvent for any write the agent performed."""
    return {
        "resourceType": "AuditEvent",
        "type": {
            "system": "http://terminology.hl7.org/CodeSystem/audit-event-type",
            "code": "rest",
            "display": "RESTful Operation",
        },
        "subtype": [{"code": action, "display": action}],
        "action": "C",
        "outcome": "0",
        "agent": [
            {
                "type": {"text": "AI voice agent"},
                "who": {"display": "Bayview Family Medicine automated front desk"},
                "requestor": False,
            }
        ],
        "source": {"observer": {"display": "medical_reception voice agent"}},
        "entity": [
            {
                "what": {"display": target},
                "detail": [{"type": "value", "valueString": value}]
                + ([{"type": "note", "valueString": note}] if note else []),
            }
        ],
    }


def fhir_risk_assessment(
    patient_id: str, action: str, value: str, certainty: str, note: str = ""
) -> dict:
    """A RiskAssessment recording how certain the agent was about a clinical write.

    This is the resource that makes uncertainty auditable. A chart entry written
    from an unconfirmed transcription carries a `high` risk prediction next to
    it, so a downstream clinician can see that the value was never read back.
    """
    risk, rationale = _CERTAINTY_RISK.get(certainty, ("high", "Certainty not recorded"))
    return {
        "resourceType": "RiskAssessment",
        "status": "final",
        "subject": _patient_ref(patient_id),
        "method": {"text": "Voice agent readback verification"},
        "code": {"text": f"Transcription certainty for {action}"},
        "prediction": [
            {
                "outcome": {"text": f"Incorrect value recorded: {value}"},
                "qualitativeRisk": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                            "code": risk,
                        }
                    ],
                    "text": certainty,
                },
                "rationale": rationale,
            }
        ],
        "note": [{"text": note}] if note else [],
    }



def dual_write(resource: dict) -> None:
    """Called by write tools. Never raises, never blocks the eval."""
    _client.create(resource)
