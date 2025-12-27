from dataclasses import field, is_dataclass, fields
from typing import Any, Generic, List, TypeVar
from .resource import ResourceRef
from .registry import ProducerBase

# =============================================================================
# Producers: typed IO dataclasses + one-phase wiring
# =============================================================================

def out_field() -> Any:
    return field(metadata={"io": "out"})


IO = TypeVar("IO")


class WiredProducer(ProducerBase, Generic[IO]):
    """
    A producer that is *wired* to concrete ResourceRef IO at construction time.

    Contract:
      - IO must be a dataclass.
      - Outputs are inferred from IO fields marked with out_field().
    """

    def __init__(self, io: IO) -> None:
        if not is_dataclass(io):
            raise TypeError(f"{self.__class__.__name__} expects a dataclass IO object")
        self.io: IO = io

        outs: List[str] = []
        for f in fields(io):
            if f.metadata.get("io") == "out":
                ref = getattr(io, f.name)
                if not isinstance(ref, ResourceRef):
                    raise TypeError(f"IO field '{f.name}' marked out but is not a ResourceRef")
                outs.append(ref.name)
        self.outputs = tuple(outs)
