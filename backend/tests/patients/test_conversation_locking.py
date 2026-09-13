import uuid
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from app.patients.service import get_owned_conversation_for_update


def test_locked_conversation_query_has_no_outer_join():
    user_id = uuid.uuid4()
    conversation = SimpleNamespace(user_id=user_id)

    class Result:
        def first(self):
            return conversation

    class Database:
        def scalars(self, statement):
            sql = str(
                statement.compile(
                    dialect=postgresql.dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            )
            assert "FOR UPDATE" in sql
            assert "JOIN" not in sql
            return Result()

    result = get_owned_conversation_for_update(
        Database(), SimpleNamespace(id=user_id), uuid.uuid4()
    )

    assert result is conversation
