import asyncio
import json
import os
import logging
from datetime import datetime, date
from typing import List, Optional, Dict

import streamlit as st
import nest_asyncio
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding Model
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

# Data Models (unchanged)
class CareTeamInfo(BaseModel):
    name: str
    role: Optional[str] = None
    practitioner: Optional[str] = None

class ExplanationOfBenefitInfo(BaseModel):
    claim_id: str
    service: str
    total_cost: float
    insurance_paid: float
    patient_paid: float
    date: str

class DocumentReferenceInfo(BaseModel):
    description: str
    date: str
    content: Optional[str] = None

class ConditionInfo(BaseModel):
    code: str
    display: str
    clinical_status: str
    onset_date: Optional[str] = None

class EncounterInfo(BaseModel):
    date: str
    reason_display: Optional[str] = None
    type_display: Optional[str] = None
    provider: Optional[str] = None
    practitioner: Optional[str] = None

class MedicationInfo(BaseModel):
    name: str
    start_date: Optional[str] = None
    instructions: Optional[str] = None
    prescriber: Optional[str] = None

class ObservationInfo(BaseModel):
    name: str
    value: str
    unit: Optional[str] = None
    date: Optional[str] = None

class ClaimInfo(BaseModel):
    service: str
    total_cost: float
    insurance_paid: float
    patient_paid: float
    date: str

class CarePlanInfo(BaseModel):
    description: str
    start_date: Optional[str] = None

class DiagnosticReportInfo(BaseModel):
    name: str
    date: str
    content: Optional[str] = None

class ProcedureInfo(BaseModel):
    name: str
    date: str

class ImmunizationInfo(BaseModel):
    name: str
    date: str

class AllergyIntoleranceInfo(BaseModel):
    substance: str
    reaction: Optional[str] = None
    onset_date: Optional[str] = None

class PatientInfo(BaseModel):
    given_name: str
    family_name: str
    birth_date: Optional[str] = None
    gender: Optional[str] = None
    identifier: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    care_team: List[CareTeamInfo] = Field(default_factory=list)
    explanations_of_benefit: List[ExplanationOfBenefitInfo] = Field(default_factory=list)
    document_references: List[DocumentReferenceInfo] = Field(default_factory=list)
    conditions: List[ConditionInfo] = Field(default_factory=list)
    encounters: List[EncounterInfo] = Field(default_factory=list)
    medications: List[MedicationInfo] = Field(default_factory=list)
    observations: List[ObservationInfo] = Field(default_factory=list)
    claims: List[ClaimInfo] = Field(default_factory=list)
    care_plans: List[CarePlanInfo] = Field(default_factory=list)
    diagnostic_reports: List[DiagnosticReportInfo] = Field(default_factory=list)
    procedures: List[ProcedureInfo] = Field(default_factory=list)
    immunizations: List[ImmunizationInfo] = Field(default_factory=list)
    allergies: List[AllergyIntoleranceInfo] = Field(default_factory=list)

class CaseSummary(BaseModel):
    patient_name: str
    age: int
    overall_assessment: str
    identifier: Optional[str] = None
    address: Optional[str] = None
    gender: Optional[str] = None
    birth_date: Optional[str] = None
    phone: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    care_team: List[CareTeamInfo]
    explanations_of_benefit: List[ExplanationOfBenefitInfo]
    document_references: List[DocumentReferenceInfo]
    conditions: List[ConditionInfo]
    encounters: List[EncounterInfo]
    medications: List[MedicationInfo]
    observations: List[ObservationInfo]
    claims: List[ClaimInfo]
    care_plans: List[CarePlanInfo]
    diagnostic_reports: List[DiagnosticReportInfo]
    procedures: List[ProcedureInfo]
    immunizations: List[ImmunizationInfo]
    allergies: List[AllergyIntoleranceInfo]

def calculate_age(birth_date_str: str) -> int:
    try:
        birth_dt = datetime.fromisoformat(birth_date_str).date()
        today = date.today()
        return today.year - birth_dt.year - ((today.month, today.day) < (birth_dt.month, birth_dt.day))
    except ValueError:
        logger.warning(f"Invalid birth date format: {birth_date_str}")
        return 0

def format_date(date_str: str) -> str:
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%d.%m.%Y (%A)")
    except ValueError:
        return date_str

def parse_synthea_patient(file_path: str) -> PatientInfo:
    logger.info(f"Reading file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            bundle = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise
    patient_resource = None
    care_team = []
    explanations_of_benefit = []
    document_references = []
    conditions = []
    encounters = []
    medications = []
    observations = []
    claims = []
    care_plans = []
    diagnostic_reports = []
    procedures = []
    immunizations = []
    allergies = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")
        if resource_type == "Patient":
            patient_resource = resource
        elif resource_type == "CareTeam":
            for participant in resource.get("participant", []):
                care_team.append(CareTeamInfo(
                    name=resource.get("name", "Unknown"),
                    role=participant.get("role", [{}])[0].get("coding", [{}])[0].get("display", "Unknown"),
                    practitioner=participant.get("member", {}).get("display", "Unknown")
                ))
        elif resource_type == "ExplanationOfBenefit":
            total_cost = float(resource.get("total", [{}])[0].get("amount", {}).get("value", 0))
            insurance_paid = float(resource.get("payment", {}).get("amount", {}).get("value", 0))
            explanations_of_benefit.append(ExplanationOfBenefitInfo(
                claim_id=resource.get("claim", {}).get("reference", "Unknown"),
                service=resource.get("item", [{}])[0].get("productOrService", {}).get("coding", [{}])[0].get("display", "Unknown"),
                total_cost=total_cost,
                insurance_paid=insurance_paid,
                patient_paid=total_cost - insurance_paid,
                date=resource.get("created", "")
            ))
        elif resource_type == "DocumentReference":
            document_references.append(DocumentReferenceInfo(
                description=resource.get("description", "Unknown"),
                date=resource.get("created", ""),
                content=resource.get("content", [{}])[0].get("attachment", {}).get("data")
            ))
        elif resource_type == "Condition":
            conditions.append(ConditionInfo(
                code=resource.get("code", {}).get("coding", [{}])[0].get("code", "Unknown"),
                display=resource.get("code", {}).get("coding", [{}])[0].get("display", "Unknown"),
                clinical_status=resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "unknown"),
                onset_date=resource.get("onsetDateTime")
            ))
        elif resource_type == "Encounter":
            encounters.append(EncounterInfo(
                date=resource.get("period", {}).get("start", ""),
                reason_display=resource.get("reasonCode", [{}])[0].get("coding", [{}])[0].get("display") if resource.get("reasonCode") else None,
                type_display=resource.get("type", [{}])[0].get("coding", [{}])[0].get("display") if resource.get("type") else None,
                provider=resource.get("serviceProvider", {}).get("display", "Unknown"),
                practitioner=resource.get("participant", [{}])[0].get("individual", {}).get("display")
            ))
        elif resource_type == "MedicationRequest":
            if resource.get("status") == "active":
                medications.append(MedicationInfo(
                    name=resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "Unknown"),
                    start_date=resource.get("authoredOn"),
                    instructions=resource.get("dosageInstruction", [{}])[0].get("text"),
                    prescriber=resource.get("requester", {}).get("display")
                ))
        elif resource_type == "Observation":
            value = resource.get("valueQuantity", {}).get("value", resource.get("valueString", "Unknown"))
            observations.append(ObservationInfo(
                name=resource.get("code", {}).get("coding", [{}])[0].get("display", "Unknown"),
                value=str(value),
                unit=resource.get("valueQuantity", {}).get("unit"),
                date=resource.get("effectiveDateTime")
            ))
        elif resource_type == "Claim":
            total_cost = float(resource.get("total", {}).get("value", 0))
            insurance_paid = sum(float(item.get("adjudicatedAmount", {}).get("value", 0)) for item in resource.get("insurance", []))
            claims.append(ClaimInfo(
                service=resource.get("item", [{}])[0].get("productOrService", {}).get("coding", [{}])[0].get("display", "Unknown"),
                total_cost=total_cost,
                insurance_paid=insurance_paid,
                patient_paid=total_cost - insurance_paid,
                date=resource.get("created")
            ))
        elif resource_type == "CarePlan":
            care_plans.append(CarePlanInfo(
                description=resource.get("description", resource.get("category", [{}])[0].get("coding", [{}])[0].get("display", "Unknown")),
                start_date=resource.get("period", {}).get("start")
            ))
        elif resource_type == "DiagnosticReport":
            diagnostic_reports.append(DiagnosticReportInfo(
                name=resource.get("code", {}).get("coding", [{}])[0].get("display", "Unknown"),
                date=resource.get("effectiveDateTime", ""),
                content=resource.get("presentedForm", [{}])[0].get("data") if resource.get("presentedForm") else None
            ))
        elif resource_type == "Procedure":
            procedures.append(ProcedureInfo(
                name=resource.get("code", {}).get("coding", [{}])[0].get("display", "Unknown"),
                date=resource.get("performedDateTime", "")
            ))
        elif resource_type == "Immunization":
            immunizations.append(ImmunizationInfo(
                name=resource.get("vaccineCode", {}).get("coding", [{}])[0].get("display", "Unknown"),
                date=resource.get("occurrenceDateTime", "")
            ))
        elif resource_type == "AllergyIntolerance":
            allergies.append(AllergyIntoleranceInfo(
                substance=resource.get("code", {}).get("coding", [{}])[0].get("display", "Unknown"),
                reaction=resource.get("reaction", [{}])[0].get("manifestation", [{}])[0].get("coding", [{}])[0].get("display") if resource.get("reaction") else None,
                onset_date=resource.get("onset")
            ))
    if not patient_resource:
        logger.error(f"No patient resource found in {file_path}.")
        raise ValueError("No patient resource found in file.")
    name_entry = patient_resource.get("name", [{}])[0]
    address_data = patient_resource.get("address", [{}])[0] if patient_resource.get("address") else {}
    address = ", ".join(address_data.get("line", [])) if address_data.get("line") else None
    phone = next((tel["value"] for tel in patient_resource.get("telecom", []) if tel.get("system") == "phone"), None)
    return PatientInfo(
        given_name=name_entry.get("given", [""])[0],
        family_name=name_entry.get("family", ""),
        birth_date=patient_resource.get("birthDate", None),
        gender=patient_resource.get("gender", None),
        identifier=patient_resource.get("identifier", [{}])[0].get("value") if patient_resource.get("identifier") else None,
        address=address,
        phone=phone,
        city=address_data.get("city", "Unknown"),
        state=address_data.get("state", "Unknown"),
        country=address_data.get("country", "Unknown"),
        care_team=care_team,
        explanations_of_benefit=explanations_of_benefit,
        document_references=document_references,
        conditions=conditions,
        encounters=encounters,
        medications=medications,
        observations=observations,
        claims=claims,
        care_plans=care_plans,
        diagnostic_reports=diagnostic_reports,
        procedures=procedures,
        immunizations=immunizations,
        allergies=allergies
    )

async def generate_case_summary(patient_data: PatientInfo) -> CaseSummary:
    overall_assessment = "Patient data processed successfully."
    return CaseSummary(
        patient_name=f"{patient_data.given_name} {patient_data.family_name}",
        age=calculate_age(patient_data.birth_date) if patient_data.birth_date else 0,
        overall_assessment=overall_assessment,
        identifier=patient_data.identifier,
        address=patient_data.address,
        gender=patient_data.gender,
        birth_date=patient_data.birth_date,
        phone=patient_data.phone,
        city=patient_data.city,
        state=patient_data.state,
        country=patient_data.country,
        care_team=patient_data.care_team,
        explanations_of_benefit=patient_data.explanations_of_benefit,
        document_references=patient_data.document_references,
        conditions=patient_data.conditions,
        encounters=patient_data.encounters,
        medications=patient_data.medications,
        observations=patient_data.observations,
        claims=patient_data.claims,
        care_plans=patient_data.care_plans,
        diagnostic_reports=patient_data.diagnostic_reports,
        procedures=patient_data.procedures,
        immunizations=patient_data.immunizations,
        allergies=patient_data.allergies
    )

def load_all_patients(file_path: str) -> List[CaseSummary]:
    summaries = []
    logger.info(f"Processing file: {file_path}")
    try:
        patient_info = parse_synthea_patient(file_path)
        case_summary = asyncio.run(generate_case_summary(patient_info))
        summaries.append(case_summary)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        st.error(f"Error loading patient data: {e}")
    return summaries

def json_to_text(case_summaries: List[CaseSummary]) -> str:
    lines = []
    for s in case_summaries:
        lines.append(f"Patient: {s.patient_name}, Age: {s.age}")
        if s.conditions:
            lines.append("Diagnoses:")
            for c in s.conditions:
                lines.append(f"- {c.display} ({c.clinical_status})")
        if s.encounters:
            lines.append("Encounters:")
            for e in s.encounters:
                lines.append(f"- {e.date}: {e.reason_display or 'No reason'}, Type: {e.type_display}, Provider: {e.provider}")
        if s.medications:
            lines.append("Medications:")
            for m in s.medications:
                lines.append(f"- {m.name}, Start: {m.start_date or 'Unknown'}")
        if s.allergies:
            lines.append("Allergies:")
            for a in s.allergies:
                lines.append(f"- {a.substance}, Reaction: {a.reaction or 'Unknown'}")
        if s.procedures:
            lines.append("Procedures:")
            for p in s.procedures:
                lines.append(f"- {p.name}, Date: {p.date}")
        lines.append("\n")
    return "\n".join(lines)

def setup_chatbot(case_summaries: List[CaseSummary]):
    text_data = json_to_text(case_summaries)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(text_data)
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def handle_query(query: str) -> str:
        query_lower = query.lower().strip()

        # Handle casual greetings with consistent responses
        if "nasılsın" in query_lower or "how are you" in query_lower or "comment ça va" in query_lower:
            return "I'm good, thank you! / Je vais bien, merci ! How can I assist you? / Comment puis-je vous aider ?"
        elif "günaydın" in query_lower or "good morning" in query_lower or "bonjour" in query_lower:
            return "Good morning! / Bonjour ! How can I help you? / Comment puis-je vous aider ?"
        elif "teşekkürler" in query_lower or "thank you" in query_lower or "merci" in query_lower:
            return "You're welcome! / De rien ! Always here to help. / Toujours là pour aider."

        # Patient selection logic
        if len(case_summaries) == 1:
            patient = case_summaries[0]
        else:
            return "Multiple patients detected. Please specify the patient name. / Plusieurs patients détectés. Veuillez préciser le nom du patient."

        # Common patient data
        encounters_sorted = sorted(patient.encounters, key=lambda x: x.date or "") if patient.encounters else []
        first_visit = encounters_sorted[0] if encounters_sorted else None
        last_visit = encounters_sorted[-1] if encounters_sorted else None

        # Predefined bilingual responses
        name_str = f"Name / Nom: {patient.patient_name}"
        age_str = f"Age / Âge: {patient.age if patient.age > 0 else 'Unknown / Inconnu'}"
        phone_str = f"Phone / Téléphone: {patient.phone or 'Unknown / Inconnu'}"
        doctors_str = f"Doctors / Médecins: {', '.join({ct.practitioner for ct in patient.care_team if ct.practitioner} | {e.practitioner for e in patient.encounters if e.practitioner}) or 'Unknown / Inconnu'}"
        info_str = (
            f"Name / Nom: {patient.patient_name}\n"
            f"Age / Âge: {patient.age if patient.age > 0 else 'Unknown / Inconnu'}\n"
            f"Birth Date / Date de naissance: {patient.birth_date or 'Unknown / Inconnu'}\n"
            f"Gender / Genre: {patient.gender or 'Unknown / Inconnu'}\n"
            f"Phone / Téléphone: {patient.phone or 'Unknown / Inconnu'}"
        )
        appointments_str = (
            "Appointments / Rendez-vous:\n" +
            "\n".join([f"- Date: {e.date}, Reason / Raison: {e.reason_display or 'Unspecified / Non spécifié'}, Type / Type: {e.type_display or 'Unspecified / Non spécifié'}, Provider / Fournisseur: {e.provider}" for e in encounters_sorted]) 
            if encounters_sorted else "No appointment information available. / Aucune information sur les rendez-vous disponible."
        )
        meds_str = (
            "Medications / Médicaments:\n" +
            "\n- ".join([f"{m.name}, Start / Début: {m.start_date or 'Unknown / Inconnu'}" for m in patient.medications]) 
            if patient.medications else "No medications recorded. / Aucun médicament enregistré."
        )
        allergies_str = (
            "Allergies / Allergies:\n" +
            "\n- ".join([f"{a.substance}, Reaction / Réaction: {a.reaction or 'Unknown / Inconnu'}" for a in patient.allergies]) 
            if patient.allergies else "No allergies recorded. / Aucune allergie enregistrée."
        )
        symptoms_str = (
            "Symptoms / Symptômes:\n" +
            "\n- ".join([f"{c.display} ({c.clinical_status})" for c in patient.conditions]) 
            if patient.conditions else "No symptoms recorded. / Aucun symptôme enregistré."
        )
        diagnoses_str = (
            "Diagnoses / Diagnostics:\n" +
            "\n- ".join([f"{c.display} ({c.clinical_status})" for c in patient.conditions]) 
            if patient.conditions else "No diagnoses recorded. / Aucun diagnostic enregistré."
        )
        treatments_str = (
            "Treatments / Traitements:\n" +
            "\n- ".join([f"{p.name}, Date: {p.date}" for p in patient.procedures]) 
            if patient.procedures else "No treatments recorded. / Aucun traitement enregistré."
        )
        insurance_str = (
            "Insurance History / Historique d'assurance:\n" +
            "\n- ".join([f"{eob.service}: Insurance / Assurance ${eob.insurance_paid}, Patient / Patient ${eob.patient_paid}, Total / Total ${eob.total_cost}, Date: {eob.date}" for eob in patient.explanations_of_benefit]) 
            if patient.explanations_of_benefit else "No insurance history available. / Aucun historique d'assurance disponible."
        )
        reports_str = (
            "Reports / Rapports:\n" +
            "\n- ".join([f"{dr.name}, Date: {dr.date}, Content / Contenu: {dr.content or 'None / Aucun'}" for dr in patient.diagnostic_reports]) 
            if patient.diagnostic_reports else "No reports available. / Aucun rapport disponible."
        )
        first_visit_str = f"First Hospital Visit / Première visite à l'hôpital: {format_date(first_visit.date) if first_visit else 'Unknown / Inconnu'}"
        last_visit_str = f"Last Hospital Visit / Dernière visite à l'hôpital: {format_date(last_visit.date) if last_visit else 'Unknown / Inconnu'}"
        # Dynamic examination information from JSON data
        examination_str = "Response based on data:\n"
        if patient.encounters:
            for e in sorted(patient.encounters, key=lambda x: x.date or ""):
                examination_str += f"{e.date}: {e.reason_display or 'No reason'}, Type: {e.type_display}, Provider: {e.provider}\n"
        if patient.conditions:
            for c in patient.conditions:
                examination_str += f"{c.display} ({c.clinical_status})\n"
        examination_str += "Encounters:\n"
        if patient.encounters:
            for e in sorted(patient.encounters, key=lambda x: x.date or ""):
                examination_str += f"{e.date}: {e.reason_display or 'No reason'}, Type: {e.type_display}, Provider: {e.provider}\n"
        if not patient.encounters and not patient.conditions:
            examination_str += "No examination information available."

        # Query matching with corrected mappings
        if "what is the patient's name" in query_lower or "quel est le nom du patient" in query_lower or "hastanın ismi nedir" in query_lower:
            return name_str
        elif "what is the patient's information" in query_lower or "quelles sont les informations du patient" in query_lower or "hastanın bilgisi nedir" in query_lower:
            return info_str
        elif ("what is the patient's appointment information" in query_lower or 
              "quelles sont les informations de rendez-vous du patient" in query_lower or 
              "appointment information" in query_lower or 
              "hastanın randevu bilgileri nelerdir" in query_lower):
            return appointments_str
        elif "who are the patient's doctors" in query_lower or "qui sont les médecins du patient" in query_lower or "hastanın doktorları kimdir" in query_lower or "hastanın doktoru kimdir" in query_lower:
            return doctors_str
        elif ("first hospital visit" in query_lower or 
              "quand a eu lieu la première visite à l'hôpital du patient" in query_lower or 
              "hastanın ilk hastane ziyareti ne zamandır" in query_lower or 
              "la première visite à l'hôpital du patient" in query_lower):
            return first_visit_str
        elif ("last hospital visit" in query_lower or 
              "quand a eu lieu la dernière visite à l'hôpital du patient" in query_lower or 
              "hastanın son hastane ziyareti ne zamandır" in query_lower or 
              "the patient's last hospital visit" in query_lower or 
              "la dernière visite à l'hôpital du patient" in query_lower):
            return last_visit_str
        elif ("what are the patient's medications" in query_lower or 
              "quels sont les médicaments du patient" in query_lower or 
              "medications" in query_lower or 
              "quels médicaments le patient utilise-t-il" in query_lower or 
              "hasta hangi ilaçları kullanıyor" in query_lower or 
              "hastanın ilaçları nelerdir" in query_lower):
            return meds_str
        elif ("insurance history" in query_lower or 
              "patient's insurance history" in query_lower or 
              "quel est l'historique d'assurance du patient" in query_lower or 
              "hastanın sigorta geçmişi nedir" in query_lower or 
              "historique d'assurance du patient" in query_lower):
            return insurance_str
        elif "allergies" in query_lower or "quelles sont les allergies du patient" in query_lower or "hastanın alerjileri nelerdir" in query_lower:
            return allergies_str
        elif "procedures" in query_lower or "quelles sont les procédures du patient" in query_lower or "hastanın işlemleri nelerdir" in query_lower:
            return treatments_str
        elif "symptoms" in query_lower or "quels sont les symptômes du patient" in query_lower or "hastanın semptomları nelerdir" in query_lower:
            return symptoms_str
        elif "diagnoses" in query_lower or "quels sont les diagnostics du patient" in query_lower or "hastanın tanıları nelerdir" in query_lower:
            return diagnoses_str
        elif "phone number" in query_lower or "quel est le numéro de téléphone du patient" in query_lower or "hastanın telefon numarası nedir" in query_lower:
            return phone_str
        elif "report information" in query_lower or "quelles sont les informations du rapport du patient" in query_lower or "hastanın rapor bilgisi nedir" in query_lower:
            return reports_str
        elif "age" in query_lower or "quel est l'âge du patient" in query_lower or "hastanın yaşı nedir" in query_lower:
            return age_str
        elif "treatments" in query_lower or "quels sont les traitements du patient" in query_lower or "hastanın tedavileri nelerdir" in query_lower:
            return treatments_str
        elif ("what is the patient's examination information" in query_lower or 
              "quelles sont les informations d'examen du patient" in query_lower or 
              "hastanın muayene bilgileri nelerdir" in query_lower):
            return examination_str

        # Fallback to vector search for unmatched queries
        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        return f"Response based on data: {context}"

    return handle_query

def main():
    st.title("Patient Chatbot")

    data_file = "C:/Users/hero_/Desktop/kanada chatbot/test3_102.json"
    if not os.path.exists(data_file):
        st.error(f"File not found: {data_file}. Please ensure test3_102.json is in the specified directory.")
        return

    if "case_summaries" not in st.session_state:
        with st.spinner("Loading patient data..."):
            case_summaries = load_all_patients(data_file)
            if not case_summaries:
                st.error("Failed to load patient data. Check the JSON file format.")
                return
            st.session_state["case_summaries"] = case_summaries
            st.session_state["chatbot"] = setup_chatbot(case_summaries)
        st.success(f"Loaded {len(case_summaries)} patient(s).")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Type your question here:")
    if query:
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Preparing response..."):
                response = st.session_state["chatbot"](query)
            st.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()