# backend/app/db.py

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from models.v2_models import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set!")

# The database URL for SQLAlchemy needs to be async.
# We are using psycopg2, so we need to use psycopg's async capabilities
ASYNC_DATABASE_URL = DATABASE_URL.replace("psycopg2", "psycopg")

engine = create_async_engine(ASYNC_DATABASE_URL, echo=True, future=True)

AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db() -> AsyncSession:
    """
    FastAPI dependency to get a database session.
    """
    async with AsyncSessionFactory() as session:
        yield session

async def init_db():
    """
    Creates all database tables.
    """
    async with engine.begin() as conn:
        # Use this to drop and re-create tables for development
        # await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
