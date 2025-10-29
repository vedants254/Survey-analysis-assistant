# backend/routers/files.py

"""
File Upload and Management Router - V2 Refactored
Uses SQLAlchemy ORM, async operations, and PostgreSQL.
"""

import uuid
import shutil
import hashlib
from typing import Dict, Any, List
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db import get_db
from models.v2_models import DataFile

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze DataFrame to categorize columns."""
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    categorical_columns = [col for col in df.columns if col not in numeric_columns and col not in date_columns]
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "total_columns": len(df.columns),
        "total_rows": len(df)
    }

def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Upload, process, and save a CSV or Excel file using SQLAlchemy ORM."""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed.")

    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            file_path.unlink()
            raise HTTPException(status_code=413, detail="File too large.")

        content_hash = calculate_file_hash(file_path)
        existing_file = await db.execute(select(DataFile).where(DataFile.content_hash == content_hash))
        if existing_file.scalars().first():
            file_path.unlink()
            raise HTTPException(status_code=409, detail="This exact file has already been uploaded.")

        df = pd.read_csv(file_path) if file_ext == ".csv" else pd.read_excel(file_path)
        analysis = analyze_dataframe(df)

        new_file = DataFile(
            file_uuid=str(uuid.uuid4()),
            filename=unique_filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size=file_size,
            content_hash=content_hash,
            columns=df.columns.tolist(),
            row_count=len(df),
            numeric_columns=analysis["numeric_columns"],
            categorical_columns=analysis["categorical_columns"],
            date_columns=analysis["date_columns"],
            status="READY"
        )
        db.add(new_file)
        await db.commit()
        await db.refresh(new_file)

        return {
            "file_id": new_file.id,
            "file_uuid": new_file.file_uuid,
            "filename": new_file.original_filename,
            "message": "File uploaded and processed successfully."
        }

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        error_msg = str(e)
        if "enum" in error_msg.lower():
            raise HTTPException(status_code=500, detail=f"Database enum error: {error_msg}")
        elif "permission" in error_msg.lower():
            raise HTTPException(status_code=500, detail=f"File permission error: {error_msg}")
        else:
            raise HTTPException(status_code=500, detail=f"Error processing file: {error_msg}")

@router.get("/")
async def list_files(db: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all available data files."""
    result = await db.execute(select(DataFile).order_by(DataFile.upload_time.desc()))
    files = result.scalars().all()
    return [
        {
            "file_id": file.id,
            "file_uuid": file.file_uuid,
            "filename": file.original_filename,
            "file_size": file.file_size,
            "row_count": file.row_count,
            "columns": file.columns,
            "upload_time": file.upload_time.isoformat() if file.upload_time else None,
            "status": file.status
        }
        for file in files
    ]

@router.get("/{file_id}/schema")
async def get_file_schema(file_id: int, db: AsyncSession = Depends(get_db)):
    """Get detailed schema information for a file."""
    result = await db.execute(select(DataFile).where(DataFile.id == file_id))
    file_info = result.scalars().first()
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "file_id": file_info.id,
        "filename": file_info.original_filename,
        "columns": file_info.columns,
        "numeric_columns": file_info.numeric_columns,
        "categorical_columns": file_info.categorical_columns,
        "date_columns": file_info.date_columns,
        "row_count": file_info.row_count,
    }