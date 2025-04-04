import asyncio
import json
import os
import logging
import base64
from datetime import datetime, date
from typing import List, Optional, Dict, Any

import streamlit as st
import nest_asyncio
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, pipeline
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

# Data Models
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

def decode_base64_content(encoded: Optional[str]) -> Optional[str]:
    if encoded:
        try:
            return base64.b64decode(encoded).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to decode base64 content: {e}")
            return encoded
    return None

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
                content=decode_base64_content(resource.get("content", [{}])[0].get("attachment", {}).get("data"))
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
                content=decode_base64_content(resource.get("presentedForm", [{}])[0].get("data") if resource.get("presentedForm") else None)
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
        city=address_data.get("city", "Bilinmiyor"),
        state=address_data.get("state", "Bilinmiyor"),
        country=address_data.get("country", "Bilinmiyor"),
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

class LightweightSummarizer:
    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        logger.info("Loading DistilBERT summarizer: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Summarizer loaded on device: %s", self.device)
        self.max_input_length = 512

    def split_text(self, text: str, max_length: int = 500) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= max_length:
            return [text]
        chunks = []
        current_chunk = []
        current_length = 0
        for token in tokens:
            if current_length + 1 > max_length:
                chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))
                current_chunk = [token]
                current_length = 1
            else:
                current_chunk.append(token)
                current_length += 1
        if current_chunk:
            chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))
        return chunks

    def summarize_text(self, text: str, max_length: int = 50, min_length: int = 25) -> str:
        if not text.strip():
            return ""
        try:
            chunks = self.split_text(text, self.max_input_length)
            summaries = []
            for chunk in chunks:
                input_length = len(self.tokenizer.tokenize(chunk))
                adjusted_max_length = min(max_length, max(input_length // 2, min_length))
                adjusted_min_length = min(min_length, input_length // 3)
                summary = self.summarizer(
                    chunk,
                    max_length=adjusted_max_length,
                    min_length=adjusted_min_length,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary.strip())
            return " ".join(summaries)
        except Exception as e:
            logger.warning(f"Error summarizing text: {e}. Returning raw text.")
            return text.strip()

    def unload_model(self):
        del self.summarizer
        torch.cuda.empty_cache()
        logger.info("Summarizer unloaded and memory cleared.")

async def generate_case_summary(patient_data: PatientInfo, summarizer: LightweightSummarizer) -> CaseSummary:
    sections = {
        "Demographic": f"Patient: {patient_data.given_name} {patient_data.family_name}, Birth: {patient_data.birth_date or 'Unknown'}, Gender: {patient_data.gender or 'Unknown'}, Identifier: {patient_data.identifier or 'Unknown'}, Address: {patient_data.address or 'Unknown'}",
        "CareTeam": "; ".join(f"{ct.name} (Role: {ct.role}, Practitioner: {ct.practitioner})" for ct in patient_data.care_team),
        "ExplanationsOfBenefit": "; ".join(f"{eob.service}: Total ${eob.total_cost}, Insurance ${eob.insurance_paid}" for eob in patient_data.explanations_of_benefit),
        "DocumentReferences": "; ".join(f"{dr.description}, Date: {dr.date}" for dr in patient_data.document_references),
        "Conditions": "; ".join(f"{c.display} ({c.clinical_status})" for c in patient_data.conditions),
        "Encounters": "; ".join(f"{e.date}: {e.reason_display or 'No reason'}, Type: {e.type_display}, Provider: {e.provider}" for e in patient_data.encounters),
        "Medications": "; ".join(f"{m.name}, Start: {m.start_date}" for m in patient_data.medications),
        "Observations": "; ".join(f"{o.name}: {o.value} {o.unit or ''}" for o in patient_data.observations),
        "Claims": "; ".join(f"{c.service}: Total ${c.total_cost}, Insurance ${c.insurance_paid}" for c in patient_data.claims),
        "CarePlans": "; ".join(f"{cp.description}, Start: {cp.start_date}" for cp in patient_data.care_plans),
        "DiagnosticReports": "; ".join(f"{dr.name}, Date: {dr.date}" for dr in patient_data.diagnostic_reports),
        "Procedures": "; ".join(f"{p.name}, Date: {p.date}" for p in patient_data.procedures),
        "Immunizations": "; ".join(f"{i.name}, Date: {i.date}" for i in patient_data.immunizations),
        "Allergies": "; ".join(f"{a.substance}, Reaction: {a.reaction or 'Unknown'}" for a in patient_data.allergies)
    }
    section_summaries = await summarize_sections(sections, summarizer)
    overall_summary = ". ".join(val for val in section_summaries.values() if val)
    return CaseSummary(
        patient_name=f"{patient_data.given_name} {patient_data.family_name}",
        age=calculate_age(patient_data.birth_date) if patient_data.birth_date else 0,
        overall_assessment=overall_summary,
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

async def summarize_sections(sections: Dict[str, str], summarizer: LightweightSummarizer) -> Dict[str, str]:
    tasks = [asyncio.to_thread(summarizer.summarize_text, text) for text in sections.values() if text.strip()]
    summarized_texts = await asyncio.gather(*tasks)
    return dict(zip([k for k in sections if sections[k].strip()], summarized_texts))

def load_all_patients(file_path: str) -> List[CaseSummary]:
    summaries = []
    summarizer = LightweightSummarizer()
    logger.info(f"Processing file: {file_path}")
    try:
        patient_info = parse_synthea_patient(file_path)
        case_summary = asyncio.run(generate_case_summary(patient_info, summarizer))
        summaries.append(case_summary)
    except ValueError as e:
        logger.warning(f"{file_path} is an invalid patient file: {e}. Skipping.")
    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {e}")
    summarizer.unload_model()
    return summaries

def json_to_text(case_summaries: List[CaseSummary]) -> str:
    lines = []
    for s in case_summaries:
        lines.append(f"Patient: {s.patient_name}, Age: {s.age}, Identifier: {s.identifier or 'Unknown'}, Address: {s.address or 'Unknown'}")
        lines.append(f"Overall Health: {s.overall_assessment}")
        if s.care_team:
            lines.append("Care Team:")
            for ct in s.care_team:
                lines.append(f"- {ct.name} (Role: {ct.role}, Practitioner: {ct.practitioner})")
        if s.explanations_of_benefit:
            lines.append("Explanations of Benefit:")
            for eob in s.explanations_of_benefit:
                lines.append(f"- {eob.service}: Total ${eob.total_cost}, Insurance ${eob.insurance_paid}, Patient ${eob.patient_paid}, Date: {eob.date}")
        if s.document_references:
            lines.append("Document References:")
            for dr in s.document_references:
                lines.append(f"- {dr.description}, Date: {dr.date}, Content: {dr.content or 'None'}")
        if s.conditions:
            lines.append("Diagnoses:")
            for c in s.conditions:
                lines.append(f"- {c.display} ({c.clinical_status})")
        if s.encounters:
            lines.append("Encounters:")
            for e in s.encounters:
                lines.append(f"- {e.date}: {e.reason_display or 'No reason'}, Type: {e.type_display}, Provider: {e.provider}, Doctor: {e.practitioner or 'Unknown'}")
        if s.medications:
            lines.append("Medications:")
            for m in s.medications:
                lines.append(f"- {m.name}, Start: {m.start_date or 'Unknown'}, Prescriber: {m.prescriber or 'Unknown'}")
        if s.observations:
            lines.append("Observations:")
            for o in s.observations:
                lines.append(f"- {o.name}: {o.value} {o.unit or ''}, Date: {o.date or 'Unknown'}")
        if s.claims:
            lines.append("Claims:")
            for c in s.claims:
                lines.append(f"- {c.service}: Total ${c.total_cost}, Insurance ${c.insurance_paid}, Patient ${c.patient_paid}, Date: {c.date}")
        if s.care_plans:
            lines.append("Care Plans:")
            for cp in s.care_plans:
                lines.append(f"- {cp.description}, Start: {cp.start_date}")
        if s.diagnostic_reports:
            lines.append("Diagnostic Reports:")
            for dr in s.diagnostic_reports:
                lines.append(f"- {dr.name}, Date: {dr.date}, Content: {dr.content or 'None'}")
        if s.procedures:
            lines.append("Procedures:")
            for p in s.procedures:
                lines.append(f"- {p.name}, Date: {p.date}")
        if s.immunizations:
            lines.append("Immunizations:")
            for i in s.immunizations:
                lines.append(f"- {i.name}, Date: {i.date}")
        if s.allergies:
            lines.append("Allergies:")
            for a in s.allergies:
                lines.append(f"- {a.substance}, Reaction: {a.reaction or 'Unknown'}, Onset: {a.onset_date or 'Unknown'}")
        lines.append("\n")
    return "\n".join(lines)

def is_turkish(query: str) -> bool:
    turkish_chars = "çğıöşüÇĞİÖŞÜ"
    return any(char in turkish_chars for char in query)

def setup_chatbot(case_summaries: List[CaseSummary]):
    text_data = json_to_text(case_summaries)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(text_data)
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    embeddings = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer=tokenizer)

    def generate_response(query: str, context: str) -> str:
        try:
            result = qa_pipeline(question=query, context=context)
            return result['answer'] if result['score'] > 0.1 else "Bu soruya net bir cevap veremedim." if is_turkish(query) else "I couldn't provide a clear answer to this question."
        except Exception:
            return "Bu soruya modelle yanıt veremedim." if is_turkish(query) else "I couldn't respond to this question with the model."

    def format_report_content(content: str) -> str:
        lines = content.split('\n')
        formatted_lines = []
        seen = set()
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                formatted_lines.append(line)
                seen.add(line)
        return "\n".join(formatted_lines)

    def handle_query(query: str) -> str:
        query_lower = query.lower().strip()
        turkish = is_turkish(query)

        # Handle casual greetings
        if "nasılsın" in query_lower or "how are you" in query_lower:
            return "İyiyim, teşekkür ederim! Sana nasıl yardımcı olabilirim?" if turkish else "I'm good, thank you! How can I assist you today?"
        elif "günaydın" in query_lower or "good morning" in query_lower:
            return "Günaydın! Sana nasıl yardımcı olabilirim?" if turkish else "Good morning! How can I help you?"
        elif "teşekkür" in query_lower or "thank you" in query_lower:
            return "Rica ederim, her zaman buradayım!" if turkish else "You're welcome, always here to help!"

        # Patient selection logic
        if len(case_summaries) == 1:
            patient = case_summaries[0]
        else:
            patient_name = next((q.split("patient" if not turkish else "hasta")[1].strip() for q in query_lower.split() if ("patient" if not turkish else "hasta") in q and len(q.split("patient" if not turkish else "hasta")) > 1), None)
            patient = next((s for s in case_summaries if patient_name and patient_name in s.patient_name.lower()), None)
            if not patient:
                return "Hasta bulunamadı veya birden fazla hasta var, lütfen hasta adını tam olarak belirtin (örneğin: 'Hasta Almeta56 Buckridge80')." if turkish else "Patient not found or multiple patients exist. Please specify the full patient name (e.g., 'Patient Almeta56 Buckridge80')."

        # Common patient data
        encounters_sorted = sorted(patient.encounters, key=lambda x: x.date or "") if patient.encounters else []
        first_visit = encounters_sorted[0] if encounters_sorted else None
        last_visit = encounters_sorted[-1] if encounters_sorted else None
        birth_place = f"{patient.city}, {patient.state}, {patient.country}" if patient.city != "Bilinmiyor" else "Bilinmiyor/Unknown"
        residence = f"{patient.city}, {patient.state}, {patient.country}" if patient.city != "Bilinmiyor" else "Bilinmiyor/Unknown"

        # Patient info response
        patient_info_response = [
            f"Adı/Name: {patient.patient_name}",
            f"Yaş/Age: {patient.age if patient.age > 0 else 'Bilinmiyor/Unknown'}",
            f"Doğum Tarihi/Birth Date: {patient.birth_date or 'Bilinmiyor/Unknown'}",
            f"Doğum Yeri/Birth Place: {birth_place}",
            f"Yaşadığı Yer/Residence: {residence}",
            f"Adres/Address: {patient.address or 'Bilinmiyor/Unknown'}",
            f"Telefon/Phone: {patient.phone or 'Bilinmiyor/Unknown'}",
            f"Şehir/City: {patient.city}",
            f"Eyalet/State: {patient.state}",
            f"Ülke/Country: {patient.country}",
            f"Cinsiyet/Gender: {patient.gender or 'Bilinmiyor/Unknown'}",
            f"İlk Hastane Ziyareti/First Hospital Visit: {f'{format_date(first_visit.date)}, Hastane/Hospital: {first_visit.provider}' if first_visit else 'Bilinmiyor/Unknown'}",
            f"Son Hastane Ziyareti/Last Hospital Visit: {f'{format_date(last_visit.date)}, Hastane/Hospital: {last_visit.provider}' if last_visit else 'Bilinmiyor/Unknown'}"
        ]
        patient_info_str = "\n".join(patient_info_response)

        # Simple responses
        name_str = f"Adı/Name: {patient.patient_name}"
        age_str = f"Yaş/Age: {patient.age if patient.age > 0 else 'Bilinmiyor/Unknown'}"
        phone_str = f"Telefon/Phone: {patient.phone or 'Bilinmiyor/Unknown'}"

        # Doctor response
        doctors = {ct.practitioner for ct in patient.care_team if ct.practitioner} | {e.practitioner for e in patient.encounters if e.practitioner} | {m.prescriber for m in patient.medications if m.prescriber}
        doctors_str = f"Doktor(lar)/Doctor(s): {', '.join(doctors) if doctors else 'Bilinmiyor/Unknown'}"

        # Medications response
        meds_response = "\n- ".join([f"{m.name}, Başlangıç/Start: {m.start_date or 'Bilinmiyor/Unknown'}" for m in patient.medications])
        meds_str = f"İlaçlar/Reçeteler/Medicines/Prescriptions:\n- {meds_response}" if meds_response else ("İlaç bilgisi yok/No medication info.")

        # Allergies response
        allergies_response = "\n- ".join([f"{a.substance}, Tepki/Reaction: {a.reaction or 'Bilinmiyor/Unknown'}" for a in patient.allergies])
        allergies_str = f"Alerjiler/Allergies:\n- {allergies_response}" if allergies_response else ("Alerji bilgisi yok/No allergy info.")

        # Diagnoses response
        diagnoses_response = "\n- ".join([f"{c.display} ({c.clinical_status})" for c in patient.conditions])
        diagnoses_str = f"Tanılar/Diagnoses:\n- {diagnoses_response}" if diagnoses_response else ("Tanı bilgisi yok/No diagnosis info.")

        # Symptoms response (same as diagnoses)
        symptoms_str = diagnoses_str

        # Procedures/Treatments response
        treatments_response = "\n- ".join([f"{p.name}, Tarih/Date: {p.date}" for p in patient.procedures] + [f"{cp.description}, Başlangıç/Start: {cp.start_date or 'Bilinmiyor/Unknown'}" for cp in patient.care_plans])
        treatments_str = f"Tedaviler/İşlemler/Treatments/Procedures:\n- {treatments_response}" if treatments_response else ("Tedavi/işlem bilgisi yok/No treatment/procedure info.")

        # Insurance response
        insurance_response = "\n- ".join([f"{eob.service}: Sigorta/Insurance ${eob.insurance_paid}, Hasta/Patient ${eob.patient_paid}, Toplam/Total ${eob.total_cost}, Tarih/Date: {eob.date}" for eob in patient.explanations_of_benefit])
        insurance_str = f"Sigorta Bilgisi/Insurance History:\n- {insurance_response}" if insurance_response else ("Sigorta bilgisi yok/No insurance info.")

        # Reports response (completely in Turkish, dynamically from test3_102.json)
        unique_reports = []
        seen_content = set()
        for dr in patient.diagnostic_reports + patient.document_references:
            content = format_report_content(dr.content or "Detay yok")
            report_key = (dr.date, content)
            if report_key not in seen_content:
                formatted_date = format_date(dr.date) if dr.date else "Tarih belirtilmemiş"
                report_name = dr.name if hasattr(dr, 'name') else dr.description
                unique_reports.append(f"- {report_name}, Tarih: {formatted_date}, İçerik:\n{content}")
                seen_content.add(report_key)
        reports_str_turkish = "Raporlar:\n" + "\n".join(unique_reports) if unique_reports else "Rapor bulunamadı."

        # Appointments/Examinations response
        appointments_response = "\n".join([
            f"Tarih/Date: {e.date}, Sebep/Reason: {e.reason_display or 'Belirtilmemiş/Unspecified'}, Tür/Type: {e.type_display or 'Belirtilmemiş/Unspecified'}, Sağlayıcı/Provider: {e.provider}"
            for e in sorted(patient.encounters, key=lambda x: x.date or "")
        ])
        appointments_str = f"Randevular/Muayeneler/Appointments/Examinations:\n{appointments_response}" if appointments_response else "Randevu/Muayene bilgisi yok/No appointment/examination info."

        # First/Last visit responses
        first_visit_str = f"İlk Hastane Ziyareti/First Hospital Visit: {f'{format_date(first_visit.date)}, Hastane/Hospital: {first_visit.provider}' if first_visit else 'Bilinmiyor/Unknown'}"
        last_visit_str = f"Son Hastane Ziyareti/Last Hospital Visit: {f'{format_date(last_visit.date)}, Hastane/Hospital: {last_visit.provider}' if last_visit else 'Bilinmiyor/Unknown'}"

        # Query handling with precise conditions
        if "randevu bilgisi" in query_lower or "randevular" in query_lower or "what is the patient's appointment information" in query_lower or "hastanın randevu bilgileri nelerdir" in query_lower or "hastanın randevuları nelerdir" in query_lower:
            return appointments_str
        elif "muayene bilgisi" in query_lower or "examination information" in query_lower or "what are the patient's examination information" in query_lower or "hastanın muayene bilgileri nelerdir" in query_lower or "hastanın muayeneleri nelerdir" in query_lower:
            return appointments_str
        elif "ismi nedir" in query_lower or "what is the patient's name" in query_lower:
            return name_str
        elif "yaşı nedir" in query_lower or "what is the patient's age" in query_lower:
            return age_str
        elif "telefon numarası nedir" in query_lower or "phone number" in query_lower or "what is the patient's phone number" in query_lower:
            return phone_str
        elif "doktoru kimdir" in query_lower or "doktorları kimdir" in query_lower or "doctor" in query_lower or "who is the patient's doctor" in query_lower:
            return doctors_str
        elif "hastanın bilgisi" in query_lower or "hasta bilgisi" in query_lower or "information" in query_lower or "what is the patient's information" in query_lower:
            return patient_info_str
        elif "alerjileri" in query_lower or "allergies" in query_lower or "what are the patient's allergies" in query_lower:
            return allergies_str
        elif "ilk hastane ziyareti" in query_lower or "first hospital visit" in query_lower or "when is the patient's first hospital visit" in query_lower:
            return first_visit_str
        elif "son hastane ziyareti" in query_lower or "last hospital visit" in query_lower or "when was the patient's last hospital visit" in query_lower:
            return last_visit_str
        elif "tanıları" in query_lower or "diagnoses" in query_lower or "what are the diagnoses of the patient" in query_lower or "diagnosis information" in query_lower or "tanı bilgisi" in query_lower:
            return diagnoses_str
        elif "semptomları" in query_lower or "symptoms" in query_lower or "what are the patient's symptoms" in query_lower:
            return symptoms_str
        elif "işlemleri" in query_lower or "tedavileri" in query_lower or "procedures" in query_lower or "treatments" in query_lower or "what are the patient's procedures" in query_lower or "what are the patient's treatments" in query_lower:
            return treatments_str
        elif "sigorta bilgisi" in query_lower or "sigorta geçmişi" in query_lower or "insurance" in query_lower or "insurance history" in query_lower or "what is the patient's insurance history" in query_lower:
            return insurance_str
        elif "reçete bilgisi" in query_lower or "kullandığı ilaçlar" in query_lower or "ilaçları" in query_lower or "medications" in query_lower or "prescription information" in query_lower or "medication information" in query_lower or "what medications does the patient use" in query_lower or "hasta hangi ilaçları kullanıyor" in query_lower:
            return meds_str
        elif "raporu" in query_lower or "rapor bilgisi" in query_lower or "reports" in query_lower or "report information" in query_lower or "what is the patient's report" in query_lower:
            # Türkçe sorgu için tamamen Türkçe cevap
            if turkish:
                return reports_str_turkish
            # İngilizce sorgu için mevcut reports_str (çift dilli) kullanılabilir
            return reports_str_turkish  # Senin isteğinle İngilizce sorguya da Türkçe cevap verecek şekilde ayarladım
        elif "bilgisi nedir" in query_lower or "knowledge" in query_lower or "what is the patient's knowledge" in query_lower:
            return patient_info_str

        # Fallback to retriever if no specific match
        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        return generate_response(query, context)

    return handle_query

def main():
    st.title("Hasta Chatbot / Patient Chatbot")
    st.write("Hastalar hakkında soru sorabilirsiniz. / You can ask about patients.")

    data_file = "C:/Users/hero_/Desktop/kanada chatbot/test3_102.json"
    if not os.path.exists(data_file):
        st.error(f"Dosya bulunamadı: {data_file}. Lütfen test3_102.json dosyasını belirtilen dizine ekleyin.")
        return

    if "case_summaries" not in st.session_state:
        with st.spinner("Hasta verileri yükleniyor... / Loading patient data..."):
            case_summaries = load_all_patients(data_file)
            if not case_summaries:
                st.error("Hasta verisi yüklenemedi. Dosyanın geçerli bir JSON formatında olduğundan emin olun.")
                return
            st.session_state["case_summaries"] = case_summaries
            st.session_state["chatbot"] = setup_chatbot(case_summaries)
        st.success(f"Toplam {len(case_summaries)} hasta yüklendi. / {len(case_summaries)} patients loaded.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Sorunuzu buraya yazın / Type your question here:")
    if query:
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Yanıt hazırlanıyor... / Preparing response..."):
                response = st.session_state["chatbot"](query)
            st.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()