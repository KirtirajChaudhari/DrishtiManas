"""User model."""
from enum import Enum as PyEnum
import uuid

from sqlalchemy import Boolean, Column, Enum, String
from sqlalchemy.dialects.postgresql import UUID

from .base import Base


class UserRole(str, PyEnum):
    TECHNICIAN = "technician"
    DOCTOR = "doctor"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.TECHNICIAN)
    is_active = Column(Boolean, default=True, nullable=False)
