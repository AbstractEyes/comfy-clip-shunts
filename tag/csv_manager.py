# booru tag lookup
import os
import csv
import time
from typing import Optional, Set
from rapidfuzz import process
from langdetect import detect
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ─── Source Constants ──────────────────────────────────────────────────────────
DANBOORU_CSV = "../csv/danbooru.csv"
GELBOORU_CSV = "../csv/gelbooru.csv"
E621_CSV = "../csv/e621.csv"
R34X_CSV = "../csv/rule34_xxx.csv"

DANBOORU_CATEGORIES = {0: "general", 1: "artist", 2: "deprecated", 3: "copyright", 4: "character", 5: "metadata"}
E621_CATEGORIES = {0: "general", 1: "artist", 2: "deprecated", 3: "copyright", 4: "character", 5: "species",
                   6: "invalid", 7: "metadata", 8: "lore"}
GELBOORU_CATEGORIES = R34X_CATEGORIES = DANBOORU_CATEGORIES.copy()

# ─── Global Indexes ────────────────────────────────────────────────────────────
full_tag_dict = {}        # tag_name -> {post_count, category, aliases, sources}
full_alias_dict = {}      # alias_name -> set(tag_names)
full_category_dict = {}   # category -> set(tag_names)
source_tags = {}          # source_name -> set(tag_names)

# ─── Tag Processing Core ───────────────────────────────────────────────────────
def process_tag(tag_name, post_count, category, aliases, source_name):
    tag_data = full_tag_dict.setdefault(tag_name, {
        "post_count": 0,
        "category": category,
        "aliases": set(),
        "sources": set()
    })
    tag_data["post_count"] += post_count
    tag_data["aliases"].update(aliases)
    tag_data["sources"].add(source_name)
    full_category_dict.setdefault(category, set()).add(tag_name)
    source_tags.setdefault(source_name, set()).add(tag_name)

# ─── Loader Functions ──────────────────────────────────────────────────────────
def prepare_danbooru_tags(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            tag_name = row[0].strip()
            category = DANBOORU_CATEGORIES.get(int(row[1]), "unknown")
            count = int(row[2])
            aliases = [a.strip() for a in row[3].split(",")] if row[3] else []
            process_tag(tag_name, count, category, aliases, "danbooru")

def prepare_gelbooru_tags(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] == "name": continue
            tag_name = row[0].strip()
            count = int(row[1])
            category = row[2].strip()
            process_tag(tag_name, count, category, [], "gelbooru")

def prepare_e621_tags(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            tag_name = row[0].strip()
            category = E621_CATEGORIES.get(int(row[1]), "unknown")
            count = int(row[2])
            aliases = [a.strip() for a in row[3].split(",")] if row[3] else []
            process_tag(tag_name, count, category, aliases, "e621")

def prepare_rule34x_tags(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "type": continue
            category = R34X_CATEGORIES.get(int(row[0]), "unknown")
            count = int(row[1])
            tag_name = row[2].strip()
            aliases = [a.strip() for a in row[3].split(",")] if row[3] else []
            process_tag(tag_name, count, category, aliases, "rule34x")

# ─── Alias Conflict Resolution ─────────────────────────────────────────────────
def remove_overlapping_aliases():
    all_tags = set(full_tag_dict)
    for tag, data in full_tag_dict.items():
        overlapping = data["aliases"].intersection(all_tags)
        data["aliases"] -= overlapping
    rebuild_full_alias_dict()

def rebuild_full_alias_dict():
    full_alias_dict.clear()
    for tag, data in full_tag_dict.items():
        for alias in data["aliases"]:
            full_alias_dict.setdefault(alias, set()).add(tag)

# ─── Primary Entry Point ───────────────────────────────────────────────────────
def prepare_tag_interfaces():
    paths = {
        "danbooru": Path(DANBOORU_CSV),
        "gelbooru": Path(GELBOORU_CSV),
        "e621": Path(E621_CSV),
        "rule34x": Path(R34X_CSV),
    }
    if os.path.exists(paths["danbooru"]): prepare_danbooru_tags(paths["danbooru"])
    if os.path.exists(paths["gelbooru"]): prepare_gelbooru_tags(paths["gelbooru"])
    if os.path.exists(paths["e621"]): prepare_e621_tags(paths["e621"])
    if os.path.exists(paths["rule34x"]): prepare_rule34x_tags(paths["rule34x"])

    remove_overlapping_aliases()


# ─── Language Fuzzy Matcher ─────────────────────────────────────────────────────────────
class TagFuzzyMatcher:
    def __init__(self, known_tags: Set[str], min_score: int = 90):
        self.known_tags = list(known_tags)
        self.min_score = min_score

    def correct_tag(self, query: str) -> Optional[str]:
        result = process.extractOne(query, self.known_tags, score_cutoff=self.min_score)
        return result[0] if result else None

    def batch_correct(self, tag_list: list[str]) -> list[str]:
        return [self.correct_tag(tag.strip()) or tag for tag in tag_list]

# ─── Language Detection (No CSVs Required) ─────────────────────────────────────
LANGUAGE_TAG_CODES = {
    "en": "english", "ja": "japanese", "fr": "french", "la": "latin",
    "ru": "russian", "uk": "ukrainian", "zh-cn": "chinese", "zh": "chinese", "ar": "arabic"
}

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

def is_language(tag: str, target: str) -> bool:
    return LANGUAGE_TAG_CODES.get(detect_language(tag), "unknown") == target

# ─── Tag Cleaner ───────────────────────────────────────────────────────────────
def clean_tags(tags_in: list[str], prune_languages=None, apply_fuzzy=False, fuzzy_threshold=90) -> list[str]:
    if prune_languages is None:
        prune_languages = ["japanese", "chinese", "russian", "arabic", "french", "ukrainian", "latin"]

    to_omit = []
    for tag in tags_in:
        if any(is_language(tag, lang) for lang in prune_languages):
            to_omit.append(tag)

    tags_outy = [tag for tag in tags_in if tag not in to_omit]

    if apply_fuzzy:
        known = set(full_tag_dict).union(full_alias_dict)
        matcher = TagFuzzyMatcher(known, fuzzy_threshold)
        tags_outy = matcher.batch_correct(tags_outy)

    return tags_outy

# ─── Caption + Tag Line Cleaner ────────────────────────────────────────────────
def clean_english_caption_line(line: str, apply_fuzzy=True, prune_languages=None, fuzzy_threshold=90) -> str:
    if prune_languages is None:
        prune_languages = ["japanese", "chinese", "russian", "arabic", "french", "ukrainian", "latin"]

    if ".," in line:
        caption_part, tag_str = line.split(".,", 1)
        caption = caption_part.strip() + "."
        tag_list = [t.strip() for t in tag_str.strip().split(",") if t.strip()]
    else:
        return line.strip()

    cleaned_tags = clean_tags(tag_list, prune_languages, apply_fuzzy, fuzzy_threshold)
    return f"{caption}, " + ", ".join(cleaned_tags)

# ─── Init ──────────────────────────────────────────────────────────────────────
logger.info("Loading tag dictionaries...")
start = time.time()
prepare_tag_interfaces()
logger.info(f"Loaded {len(full_tag_dict)} tags in {time.time() - start:.2f}s")

