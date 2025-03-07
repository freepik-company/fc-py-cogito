from pydantic import BaseModel


class FastAPIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    access_log: bool = False

    @classmethod
    def default(cls):
        return cls()
