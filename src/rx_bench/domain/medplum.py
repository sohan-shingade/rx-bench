"""Best-effort asynchronous mirror of domain writes into Medplum FHIR.

The tau2 DB remains the eval source of truth. Writes are queued in memory and a
single daemon thread submits transaction Bundles, so Medplum never blocks a tool
call. The mirror may be dropped when the process exits.

Configuration via env vars:
    MEDPLUM_BASE_URL   e.g. https://api.medplum.com  (default)
    MEDPLUM_CLIENT_ID / MEDPLUM_CLIENT_SECRET   client-credentials pair

Every Bundle is also appended to a local ``medplum_spool.jsonl`` in the
configured medical data directory. Network and spool failures are logged and
never affect the benchmark source of truth.
"""

import copy
import json
import os
import queue
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from loguru import logger

from rx_bench.domain.utils import MEDICAL_DATA_DIR

SPOOL_PATH = MEDICAL_DATA_DIR / "medplum_spool.jsonl"
_TIMEOUT = 5


class MedplumClient:
    """Best-effort asynchronous Medplum transaction submitter."""

    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "MEDPLUM_BASE_URL", "https://api.medplum.com"
        ).rstrip("/")
        self.client_id = os.environ.get("MEDPLUM_CLIENT_ID")
        self.client_secret = os.environ.get("MEDPLUM_CLIENT_SECRET")
        self._token: Optional[str] = None
        self._lock = threading.Lock()
        self._queue: queue.Queue[dict] = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

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

    def _spool(self, bundle: dict) -> None:
        try:
            SPOOL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SPOOL_PATH, "a") as fp:
                fp.write(json.dumps(bundle) + "\n")
        except Exception as e:
            logger.warning(f"Medplum spool write failed: {e}")

    def _worker(self) -> None:
        while True:
            bundle = self._queue.get()
            try:
                self._submit(bundle)
            finally:
                self._queue.task_done()

    def _submit(self, bundle: dict) -> None:
        self._spool(bundle)
        if not self.enabled:
            return
        with self._lock:
            token = self._get_token()
        if token is None:
            return
        try:
            resp = requests.post(
                f"{self.base_url}/fhir/R4",
                json=bundle,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/fhir+json",
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code >= 400:
                logger.warning(
                    f"Medplum transaction -> {resp.status_code}: {resp.text[:200]}"
                )
        except Exception as e:
            logger.warning(f"Medplum transaction failed (spooled locally): {e}")

    def create(self, bundle: dict) -> None:
        """Queue a transaction Bundle; returns immediately and never raises."""
        try:
            self._queue.put_nowait(bundle)
        except Exception as e:
            logger.warning(f"Medplum queue failed: {e}")


_client = MedplumClient()
_patient_reference_prefix = "urn:rx-bench:patient:"
_pending = threading.local()


def _patient_ref(patient_id: str) -> dict:
    return {
        "reference": f"{_patient_reference_prefix}{patient_id}",
        "display": patient_id,
    }


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


def fhir_appointment(
    patient_id: str,
    provider_name: str,
    date: str,
    time: str,
    reason: str,
    duration_minutes: int = 30,
) -> dict:
    start = datetime.strptime(f"{date}T{time}", "%Y-%m-%dT%H:%M").replace(
        tzinfo=ZoneInfo("America/Los_Angeles")
    )
    end = start + timedelta(minutes=duration_minutes)
    return {
        "resourceType": "Appointment",
        "status": "booked",
        "description": reason,
        "start": start.isoformat(),
        "end": end.isoformat(),
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


def fhir_audit_event(
    action: str,
    target: str,
    value: str,
    note: str = "",
    resource_ref: Optional[str] = None,
) -> dict:
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
        "recorded": datetime.now(timezone.utc).isoformat(),
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
                "what": {
                    **({"reference": resource_ref} if resource_ref else {}),
                    "display": target,
                },
                "detail": [{"type": "value", "valueString": value}]
                + ([{"type": "note", "valueString": note}] if note else []),
            }
        ],
    }


def fhir_risk_assessment(
    patient_id: str,
    action: str,
    value: str,
    certainty: str,
    note: str = "",
    resource_ref: Optional[str] = None,
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
        "basis": [{"reference": resource_ref}] if resource_ref else [],
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


def _replace_patient_references(value: object, patient_id: str, full_url: str) -> None:
    if isinstance(value, dict):
        if value.get("reference") == f"{_patient_reference_prefix}{patient_id}":
            value["reference"] = full_url
        for child in value.values():
            _replace_patient_references(child, patient_id, full_url)
    elif isinstance(value, list):
        for child in value:
            _replace_patient_references(child, patient_id, full_url)


def _patient_id(resource: dict) -> Optional[str]:
    if resource.get("resourceType") == "Patient":
        identifiers = resource.get("identifier", [])
        return identifiers[0].get("value") if identifiers else None

    prefix = _patient_reference_prefix
    stack: list[object] = [resource]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            reference = value.get("reference")
            if isinstance(reference, str) and reference.startswith(prefix):
                return reference.removeprefix(prefix)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return None


def build_transaction_bundle(
    resource: dict,
    audit_event: Optional[dict] = None,
    risk_assessment: Optional[dict] = None,
) -> dict:
    """Build one valid transaction Bundle without performing network I/O."""
    resource = copy.deepcopy(resource)
    patient_id = _patient_id(resource)
    patient_url = f"urn:uuid:{uuid.uuid4()}" if patient_id else None
    resource_url = (
        patient_url
        if resource.get("resourceType") == "Patient" and patient_url
        else f"urn:uuid:{uuid.uuid4()}"
    )

    entries = []
    if patient_id and resource.get("resourceType") != "Patient":
        entries.append(
            {
                "fullUrl": patient_url,
                "resource": {
                    "resourceType": "Patient",
                    "identifier": [{"value": patient_id}],
                },
                "request": {
                    "method": "POST",
                    "url": "Patient",
                    "ifNoneExist": f"identifier={patient_id}",
                },
            }
        )

    resources = [resource]
    if audit_event:
        audit_event = copy.deepcopy(audit_event)
        audit_event["entity"][0]["what"]["reference"] = resource_url
        resources.append(audit_event)
    if risk_assessment:
        risk_assessment = copy.deepcopy(risk_assessment)
        risk_assessment["basis"] = [{"reference": resource_url}]
        resources.append(risk_assessment)

    if patient_id and patient_url:
        for item in resources:
            _replace_patient_references(item, patient_id, patient_url)

    for item in resources:
        full_url = resource_url if item is resource else f"urn:uuid:{uuid.uuid4()}"
        request = {"method": "POST", "url": item["resourceType"]}
        if item["resourceType"] == "Patient" and patient_id:
            request["ifNoneExist"] = f"identifier={patient_id}"
        entries.append({"fullUrl": full_url, "resource": item, "request": request})

    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


def dual_write(resource: dict) -> None:
    """Queue a best-effort asynchronous mirror write; never raises or blocks."""
    try:
        resource_type = resource.get("resourceType")
        if resource_type == "AuditEvent":
            if resource.get("subtype", [{}])[0].get("code") in {
                "cancel_appointment",
                "record_no_known_allergies",
            }:
                _client.create(build_transaction_bundle(resource))
                return
            previous = getattr(_pending, "audit", None)
            if previous is not None:
                _client.create(build_transaction_bundle(previous))
            _pending.audit = resource
            _pending.risk = None
            return
        if resource_type == "RiskAssessment":
            _pending.risk = resource
            return

        bundle = build_transaction_bundle(
            resource,
            getattr(_pending, "audit", None),
            getattr(_pending, "risk", None),
        )
        _pending.audit = None
        _pending.risk = None
        _client.create(bundle)
    except Exception as e:
        logger.warning(f"Medplum mirror enqueue failed: {e}")
