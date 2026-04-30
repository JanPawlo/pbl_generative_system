import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from object import Object3D


@dataclass
class SceneContext:
    scene_path: Path
    scene_text: str
    scene_data: Dict
    scene_format: Dict
    all_objects: Dict[str, Dict]
    object3d_items: List[Object3D]


def extract_world_position(obj_dict, all_objects):
    """Compute world position, respecting ignoreParent and simple parent hierarchy."""
    if obj_dict.get("ignoreParent", False):
        return tuple(obj_dict["transform"][0:3])
    px, py, pz = 0.0, 0.0, 0.0
    current = obj_dict
    while True:
        t = current.get("transform")
        if t is not None:
            px = t[0] + px * t[6]
            py = t[1] + py * t[7]
            pz = t[2] + pz * t[8]
        parent_id = str(current.get("parent", "0"))
        if parent_id == "0":
            break
        current = all_objects.get(parent_id)
        if current is None:
            break
    return (px, py, pz)


def detect_json_formatting(json_text):
    newline = "\r\n" if "\r\n" in json_text else "\n"
    indent = 4
    for line in json_text.splitlines():
        if not line.strip() or line.lstrip().startswith(("{", "}", "[", "]")):
            continue
        match = re.match(r"^(\s+)", line)
        if match:
            leading = match.group(1)
            if "\t" in leading:
                indent = "\t"
            else:
                indent = len(leading)
            break
    return {"indent": indent, "newline": newline}


def find_matching_brace(text, start_idx):
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
        return -1
    depth = 0
    in_string = False
    escaped = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def find_top_level_object_ranges(json_text):
    ranges = {}
    depth = 0
    in_string = False
    escaped = False
    i = 0
    n = len(json_text)

    while i < n:
        ch = json_text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            if depth == 1:
                key_start = i
                i += 1
                key_escaped = False
                while i < n:
                    if key_escaped:
                        key_escaped = False
                    elif json_text[i] == "\\":
                        key_escaped = True
                    elif json_text[i] == '"':
                        break
                    i += 1
                if i >= n:
                    break
                key_end = i
                key = json.loads(json_text[key_start:key_end + 1])

                j = key_end + 1
                while j < n and json_text[j].isspace():
                    j += 1
                if j < n and json_text[j] == ":":
                    j += 1
                    while j < n and json_text[j].isspace():
                        j += 1
                    if j < n and json_text[j] == "{":
                        close_idx = find_matching_brace(json_text, j)
                        if close_idx != -1:
                            ranges[str(key)] = (j, close_idx + 1)
                i = key_end + 1
                continue
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1

    return ranges


def update_scene_models_in_text(scene_text, model_updates):
    object_ranges = find_top_level_object_ranges(scene_text)
    replacements = []
    model_re = re.compile(r'("model"\s*:\s*)"((?:[^"\\]|\\.)*)"')
    for obj_id, new_model in model_updates.items():
        obj_key = str(obj_id)
        if obj_key not in object_ranges:
            raise ValueError(f"Could not find top-level object id '{obj_key}' in scene text.")
        obj_start, obj_end = object_ranges[obj_key]
        obj_text = scene_text[obj_start:obj_end]
        match = model_re.search(obj_text)
        if not match:
            raise ValueError(f"Object id '{obj_key}' has no 'model' field in scene text.")
        value_start = obj_start + match.start(2)
        value_end = obj_start + match.end(2)
        escaped_value = json.dumps(str(new_model), ensure_ascii=False)[1:-1]
        replacements.append((value_start, value_end, escaped_value))
    result = scene_text
    for start, end, new_value in sorted(replacements, key=lambda x: x[0], reverse=True):
        result = result[:start] + new_value + result[end:]
    return result


def parse_scene_text(scene_text: str, scene_path: Path = Path("")) -> SceneContext:
    scene_data = json.loads(scene_text)
    all_objects = {str(k): v for k, v in scene_data.items() if k != "name"}
    object3d_items: List[Object3D] = []

    for obj_id, obj_dict in all_objects.items():
        object3d = Object3D.from_scene_entry(obj_id, obj_dict)
        if object3d.is_eligible:
            object3d.world_pos = extract_world_position(obj_dict, all_objects)
            object3d_items.append(object3d)

    return SceneContext(
        scene_path=scene_path,
        scene_text=scene_text,
        scene_data=scene_data,
        scene_format=detect_json_formatting(scene_text),
        all_objects=all_objects,
        object3d_items=object3d_items,
    )


def parse_scene_file(scene_path: Path) -> SceneContext:
    scene_path = Path(scene_path)
    scene_text = scene_path.read_text(encoding="utf-8")
    return parse_scene_text(scene_text=scene_text, scene_path=scene_path)
