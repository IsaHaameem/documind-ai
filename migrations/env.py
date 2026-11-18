from alembic import context

from documind.config import ConfigurationError, get_settings
from documind.database import Base, create_database

try:
    settings = get_settings()
    if context.is_offline_mode():
        context.configure(dialect_name="postgresql", target_metadata=Base.metadata, literal_binds=True)
        with context.begin_transaction():
            context.run_migrations()
    else:
        engine, _ = create_database(settings)
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=Base.metadata)
            with context.begin_transaction():
                context.run_migrations()
        engine.dispose()
except ConfigurationError as exc:
    raise SystemExit(str(exc)) from None
