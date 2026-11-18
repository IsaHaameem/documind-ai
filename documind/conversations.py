from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from .database import Conversation, Message
from .generation import Answer


class ConversationMissing(Exception):
    pass


class ConversationChanged(Exception):
    pass


def previous_questions(session: Session, identifier: UUID) -> tuple[list[str], int | None]:
    if session.get(Conversation, identifier) is None:
        raise ConversationMissing
    rows = session.scalars(select(Message).where(Message.conversation_id == identifier)
                           .order_by(Message.id.desc()).limit(8)).all()
    return [row.content for row in reversed(rows) if row.role == "user"], rows[0].id if rows else None


def save_turn(sessions: sessionmaker, identifier: UUID | None, prior_message: int | None,
              question: str, answer: Answer) -> tuple[UUID, Message]:
    with sessions.begin() as session:
        if identifier is None:
            conversation = Conversation(title=question[:150])
            session.add(conversation)
            session.flush()
        else:
            conversation = session.get(Conversation, identifier, with_for_update=True)
            if conversation is None:
                raise ConversationMissing
            latest = session.scalar(select(func.max(Message.id)).where(Message.conversation_id == identifier))
            if latest != prior_message:
                raise ConversationChanged
            conversation.updated_at = func.now()
        session.add(Message(conversation_id=conversation.id, role="user", content=question))
        session.flush()
        response = Message(
            conversation_id=conversation.id, role="assistant", content=answer.text,
            citations=[citation.model_dump(mode="json") for citation in answer.citations],
            insufficient_context=answer.insufficient_context,
        )
        session.add(response)
        session.flush()
        return conversation.id, response
