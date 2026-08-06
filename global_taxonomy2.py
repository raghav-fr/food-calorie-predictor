import json
from pathlib import Path
from difflib import get_close_matches

# =====================================================
# Paths
# =====================================================

MERGED = Path("data_folder/merged")
LOGS = Path("data_folder/logs")

LOGS.mkdir(parents=True, exist_ok=True)

PHASE1_CLASSES = MERGED / "phase1_classes.txt"
PHASE2_CLASSES = MERGED / "phase2_classes.txt"

GLOBAL_CLASSES = MERGED / "global_classes.txt"
GLOBAL_MAP = MERGED / "global_class_map.json"
DEDUP_MAP = LOGS / "dedup_map.json"


# =====================================================
# Load Phase 1
# =====================================================

with open(PHASE1_CLASSES, encoding="utf-8") as f:
    phase1_classes = [
        line.strip().lower()
        for line in f
        if line.strip()
    ]

print(f"Phase1 classes : {len(phase1_classes)}")


# =====================================================
# Load Phase 2
# =====================================================

with open(PHASE2_CLASSES, encoding="utf-8") as f:
    phase2_classes = [
        line.strip().lower()
        for line in f
        if line.strip()
    ]

print(f"Raw Phase2 classes : {len(phase2_classes)}")


# =====================================================
# Normalization
# =====================================================

def normalize(name):

    return (
        name.lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace("&", "and")
            .strip()
    )


phase2_classes = [
    normalize(c)
    for c in phase2_classes
]


# =====================================================
# STEP 3
# Fuzzy Deduplicate Phase 2
# =====================================================

print("\nRunning fuzzy matching...")

unique_names = []

dedup_map = {}

manual_review = []

for name in sorted(set(phase2_classes)):

    matches = get_close_matches(
        name,
        unique_names,
        n=1,
        cutoff=0.85
    )

    if matches:

        canonical = matches[0]

        dedup_map[name] = canonical

        # Save possible ambiguous pairs
        score = __import__("difflib").SequenceMatcher(
            None,
            name,
            canonical
        ).ratio()

        if score < 0.92:
            manual_review.append(
                {
                    "name": name,
                    "canonical": canonical,
                    "similarity": round(score, 3)
                }
            )

    else:

        unique_names.append(name)
        dedup_map[name] = name


print(f"Unique Phase2 classes : {len(unique_names)}")


# Save dedup map

with open(
    DEDUP_MAP,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "mapping": dedup_map,
            "manual_review": manual_review
        },
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"Saved {DEDUP_MAP}")


# =====================================================
# STEP 4
# Merge Phase1 + Phase2
# =====================================================

phase1_set = set(phase1_classes)

new_classes = []

for cls in sorted(unique_names):

    if cls not in phase1_set:

        new_classes.append(cls)

global_classes = phase1_classes + new_classes

print(f"New Phase2 classes : {len(new_classes)}")
print(f"Global taxonomy : {len(global_classes)}")


# =====================================================
# Save class list
# =====================================================

with open(
    GLOBAL_CLASSES,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(global_classes))


# =====================================================
# Build maps
# =====================================================

class_to_id = {}

id_to_class = {}

for idx, cls in enumerate(global_classes):

    class_to_id[cls] = idx
    id_to_class[idx] = cls


with open(
    GLOBAL_MAP,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "names": global_classes,
            "map": class_to_id,
            "id_to_name": id_to_class
        },
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"Saved {GLOBAL_MAP}")

print("\nDone.")
print(f"Total global classes : {len(global_classes)}")