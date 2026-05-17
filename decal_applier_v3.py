import copy
import math
import random
import shutil
from pathlib import Path

from PIL import Image

from decal_projection import apply_decal_to_model
from json_parser import extract_world_position, update_scene_models_in_text


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def select_biome(world_pos, biomes, rng):
    if not biomes:
        return None
    dists = [(distance(world_pos, b["pos"]), i) for i, b in enumerate(biomes)]
    dists.sort(key=lambda x: x[0])
    if len(dists) == 1:
        return dists[0][1]
    first_dist, first_idx = dists[0]
    second_dist, second_idx = dists[1]
    biome = biomes[first_idx]
    if second_dist <= first_dist * biome.get("near_threshold", 1.0):
        if rng.random() < biome.get("second_chance", 0.0):
            return second_idx
    return first_idx


def weighted_sample_without_replacement(items, weights, k, rng):
    if k <= 0 or not items:
        return []
    n = len(items)
    available = list(range(n))
    w = list(weights)
    selected = []
    for _ in range(k):
        total = sum(w)
        if total == 0:
            break
        r = rng.random() * total
        cumulative = 0.0
        for i, idx in enumerate(available):
            cumulative += w[i]
            if cumulative >= r:
                selected.append(idx)
                w[i] = 0.0
                break
    return selected


def apply_decals(scene_data, scene_text, biomes, target_percent, assets_dir, output_dir, decal_dir, rng_state):
    rng = random.Random()
    rng.setstate(rng_state)
    modified_data = copy.deepcopy(scene_data)
    all_objects = {str(k): v for k, v in modified_data.items() if k != "name"}

    eligible = []
    assignment_map = {}
    for obj_id, obj_dict in all_objects.items():
        class_name = str(obj_dict.get("class", "")).lower()
        if class_name not in {"object3d", "platform"} or "transform" not in obj_dict:
            continue
        world_pos = extract_world_position(obj_dict, all_objects)
        biome_idx = select_biome(world_pos, biomes, rng)
        if biome_idx is not None:
            model_name = obj_dict["model"]
            eligible.append((obj_id, obj_dict, biome_idx))
            assignment_map[obj_id] = (biome_idx, model_name)

    if not eligible:
        raise RuntimeError("No eligible objects found in the scene.")

    target_count = round(len(eligible) * target_percent)
    target_count = min(target_count, len(eligible))
    if target_count > 0:
        weights = [biomes[idx]["intensity"] for _, _, idx in eligible]
        selected = weighted_sample_without_replacement(eligible, weights, target_count, rng)
    else:
        selected = []

    modified_set = set()
    for idx in selected:
        obj_id, _, _ = eligible[idx]
        modified_set.add(obj_id)

    global_counter = 1
    generated_textures = {}
    model_updates = {}

    assets_dir = Path(assets_dir)
    output_dir = Path(output_dir)
    decal_dir = Path(decal_dir)

    for idx in selected:
        obj_id, obj_dict, biome_idx = eligible[idx]
        biome = biomes[biome_idx]
        model_name = obj_dict["model"]
        src_model_dir = assets_dir / model_name
        if not src_model_dir.is_dir():
            raise FileNotFoundError(f"Model folder not found: {src_model_dir}")

        if not biome["decals"]:
            raise RuntimeError(f"Biome {biome_idx} has no decal selected.")
        decal_file = rng.choice(biome["decals"])
        decal_path = decal_dir / decal_file
        if not decal_path.exists():
            raise FileNotFoundError(f"Decal image not found: {decal_path}")

        decal_stem = decal_path.stem
        new_name = f"{model_name}_{global_counter}_{decal_stem}"
        dst_model_dir = output_dir / new_name
        dst_model_dir.mkdir(parents=True, exist_ok=False)

        for item in src_model_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dst_model_dir)

        for ext in [".mtl", ".png", ".obj"]:
            for orig_file in dst_model_dir.glob(f"*{ext}"):
                orig_file.rename(dst_model_dir / (new_name + ext))
                break

        mtl_path = dst_model_dir / (new_name + ".mtl")
        if mtl_path.exists():
            lines = mtl_path.read_text().splitlines()
            new_lines = []
            for line in lines:
                if line.strip().startswith("map_Kd "):
                    new_lines.append(f"map_Kd {new_name}.png")
                else:
                    new_lines.append(line)
            mtl_path.write_text("\n".join(new_lines))

        obj_path = dst_model_dir / (new_name + ".obj")
        if obj_path.exists():
            lines = obj_path.read_text().splitlines()
            new_lines = []
            for line in lines:
                if line.strip().startswith("mtllib "):
                    new_lines.append(f"mtllib {new_name}.mtl")
                else:
                    new_lines.append(line)
            obj_path.write_text("\n".join(new_lines))

        texture_path = dst_model_dir / (new_name + ".png")

        # --- 3D decal projection (replaces old flat alpha_composite) ---
        try:
            apply_decal_to_model(
                texture_path=texture_path,
                obj_path=obj_path,
                decal_path=decal_path,
                biome=biome,
            )
        except Exception as exc:
            raise ValueError(f"Failed to apply 3D decal to {new_name}: {exc}") from exc

        obj_dict["model"] = new_name
        model_updates[obj_id] = new_name
        generated_textures[obj_id] = texture_path
        global_counter += 1

    save_path = output_dir / "modified_scene.json"
    updated_scene_text = update_scene_models_in_text(scene_text, model_updates)
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        f.write(updated_scene_text)

    source_models = {}
    for obj_id, obj_dict in all_objects.items():
        if obj_id not in modified_set and obj_id in assignment_map:
            source_models[obj_id] = obj_dict.get("model", "")

    return {
        "target_count": target_count,
        "save_path": save_path,
        "generated_textures": generated_textures,
        "modified_set": modified_set,
        "source_models": source_models,
        "output_dir": output_dir,
    }
