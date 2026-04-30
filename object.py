from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ELIGIBLE_CLASSES = {"object3d", "platform"}


@dataclass
class Object3D:
    """In-memory representation of an Object3D scene entry."""

    object_id: str
    name: str
    class_name: str
    transform: List[float]
    model: str
    parent: str
    ignore_parent: bool
    raw: Dict[str, Any] = field(default_factory=dict)
    world_pos: Optional[Tuple[float, float, float]] = None
    biome_idx: Optional[int] = None
    is_eligible: bool = True
    should_alter: bool = False

    @classmethod
    def from_scene_entry(cls, object_id: str, obj_dict: Dict[str, Any]) -> "Object3D":
        return cls(
            object_id=str(object_id),
            name=str(obj_dict.get("name", object_id)),
            class_name=str(obj_dict.get("class", "")),
            transform=list(obj_dict.get("transform", [])),
            model=str(obj_dict.get("model", "")),
            parent=str(obj_dict.get("parent", "0")),
            ignore_parent=bool(obj_dict.get("ignoreParent", False)),
            raw=obj_dict,
            is_eligible=str(obj_dict.get("class", "")).lower() in ELIGIBLE_CLASSES and "transform" in obj_dict,
        )
