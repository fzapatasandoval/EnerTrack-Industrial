"""API local del análisis integrado.

Escucha solo en el equipo. POST /analyze exige la cabecera X-API-Key con el
valor de la variable de entorno ENERTRACK_API_TOKEN.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from analyze_integrated import analyze
from forecast_series import parse_fecha, project_root

TOKEN_ENV = "ENERTRACK_API_TOKEN"
MAX_REGISTROS = 400
MAX_CONSULTA = 500
MAX_CONSUMO = 1_000_000.0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

app = FastAPI(title="EnerTrack análisis", docs_url=None, redoc_url=None, openapi_url=None)


class RegistroDiario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fecha: str = Field(min_length=8, max_length=32)
    consumo_kwh: float = Field(ge=0, le=MAX_CONSUMO)

    @field_validator("fecha")
    @classmethod
    def fecha_valida(cls, value: str) -> str:
        if parse_fecha(value) is None:
            raise ValueError("La fecha debe ser AAAA-MM-DD.")
        return value


class SolicitudAnalisis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registros: list[RegistroDiario] = Field(min_length=35, max_length=MAX_REGISTROS)
    imagen: str = Field(min_length=1, max_length=180)
    consulta: str = Field(min_length=1, max_length=MAX_CONSULTA)


def token_configurado() -> str | None:
    value = os.environ.get(TOKEN_ENV, "").strip()
    return value or None


def exigir_clave(clave: str | None) -> None:
    expected = token_configurado()
    if expected is None:
        raise HTTPException(status_code=503, detail="El servicio no está configurado.")
    provided = clave or ""
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="No autorizado.")


def imagen_local(nombre: str) -> Path:
    if nombre.startswith(("/", "\\")) or "\\" in nombre or ":" in nombre:
        raise HTTPException(status_code=422, detail="La imagen debe ser una ruta relativa del dataset local.")
    relative = Path(nombre)
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(status_code=422, detail="La imagen debe ser una ruta relativa del dataset local.")
    if relative.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=422, detail="La imagen debe ser jpg, jpeg, png o bmp.")
    dataset = (project_root() / "data" / "neu_surface_defects").resolve()
    candidate = (dataset / relative).resolve()
    if not candidate.is_relative_to(dataset) or not candidate.is_file():
        raise HTTPException(status_code=422, detail="La imagen no está en el dataset local.")
    return candidate


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "servicio": "enertrack-analisis",
        "autenticacion_configurada": token_configurado() is not None,
    }


@app.post("/analyze")
async def analyze_endpoint(request: Request, x_api_key: str | None = Header(default=None)) -> dict:
    exigir_clave(x_api_key)
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="El cuerpo no es JSON.")
    try:
        body = SolicitudAnalisis.model_validate(payload)
    except ValidationError as error:
        raise HTTPException(
            status_code=422,
            detail=[
                {"campo": ".".join(str(part) for part in item["loc"]), "mensaje": item["msg"]}
                for item in error.errors()
            ],
        )
    image_path = imagen_local(body.imagen)
    registros = [item.model_dump() for item in body.registros]
    return analyze(
        None,
        image_path,
        body.consulta,
        registros=registros,
        nombre_imagen=body.imagen,
    )
