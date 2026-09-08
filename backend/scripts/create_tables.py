from app.db import Base, get_engine
import app.models  # noqa: F401


def main() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("tables ready")


if __name__ == "__main__":
    main()
