import random
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from transformers import AutoTokenizer
import numpy as np

from .symbolic_bulk_captions import *

def extract_offset(token: str) -> str:
    entry = random.choice(FULL_ASSOCIATIVE)
    parts = entry.split()
    if len(parts) >= 3:
        return parts[1] if token == "left" else parts[-1]
    return token

FULL_TOKEN_MAP = {
    # Gender and human attributes
    "subject": lambda: random.choice(SUBJECT_TYPES),
    "object": lambda: random.choice(OBJECT_TYPES),
    "clothing": lambda: random.choice(CLOTHING_TYPES),
    "humanoid": lambda: random.choice(HUMANOID_TYPES),
    "gender": lambda: resolve_gender_token(random.choice(GENDER_TYPES)),
    "male": lambda: random.choice(MALE_TAGS),
    "female": lambda: random.choice(FEMALE_TAGS),
    "ambiguous": lambda: random.choice(AMBIG_TAGS),
    "pose": lambda: random.choice(HUMAN_POSES),
    "animal": lambda: random.choice(ANIMAL_TYPES),
    "produce": lambda: random.choice(FRUIT_AND_VEGETABLE),
    "offset": lambda: random.choice(OFFSET_TAGS),
    "zone": lambda: random.choice(ZONE_TAGS),
    "grid": lambda: random.choice(GRID_TAGS),
    "relation": lambda: random.choice(RELATION_TAGS),
    "on_surface": lambda: random.choice(CLOTHING_SURFACE_LINKERS),
    "subject_angle": lambda: random.choice(SUBJECT_PHOTOGRAPH_ANGLE),
    "human_surface": lambda: random.choice(HUMAN_SURFACES),
    "chair": lambda: random.choice(CHAIR_TYPES),
    "human_pose": lambda: random.choice(HUMAN_POSES),
    "human_interaction": lambda: random.choice(HUMAN_INTERACTIONS),
    "human_expression": lambda: random.choice(HUMAN_EXPRESSIONS),
    "human_action": lambda: random.choice(HUMAN_ACTIONS),
    "clothes": lambda: random.choice(CLOTHING_TYPES),
    "upper_clothing": lambda: random.choice(UPPER_BODY_CLOTHES_TYPES),
    "lower_clothing": lambda: random.choice(LOWER_BODY_CLOTHES_TYPES),
    "socks": lambda: random.choice(SOCK_TYPES),
    "footwear": lambda: random.choice(FOOTWEAR_TYPES),
    "accessory": lambda: random.choice(ACCESSORY_TYPES),
    "jewelry": lambda: random.choice(JEWELRY_TYPES),
    "headwear": lambda: random.choice(HEADWEAR_TYPES),
    "hair_style": lambda: random.choice(HAIRSTYLES_TYPES),
    "hair_length": lambda: random.choice(HAIR_LENGTH_TYPES),
    "material": lambda: random.choice(MATERIAL_TYPES),
    "fabric": lambda: random.choice(FABRIC_TYPES),
    "texture": lambda: random.choice(TEXTURE_TAGS),
    "pattern": lambda: random.choice(PATTERN_TYPES),
    "surface": lambda: random.choice(HUMAN_SURFACES),
    "lighting": lambda: random.choice(LIGHTING_TYPES),
    "liquid": lambda: random.choice(LIQUID_TYPES),
    "background": lambda: random.choice(BACKGROUND_TYPES),
    "decoration": lambda: random.choice(DECORATION_TYPES),
    "object_left": lambda: extract_offset("left"),
    "object_right": lambda: extract_offset("right"),
    "shape": lambda: random.choice(SHAPE_TYPES),
    "style": lambda: random.choice(STYLE_TYPES),
    "emotion": lambda: random.choice(EMOTION_TYPES),
    "intent": lambda: random.choice(INTENT_TYPES),
    "quality": lambda: random.choice(QUALITY_IMPROVERS),
    "bad_quality": lambda: random.choice(QUALITY_REDUCERS),
    "prefix": lambda: random.choice(PREFIXES),
    "logic": lambda: random.choice(SYMBOLIC_LOGIC_TAGS),
    "associative": lambda: random.choice(ASSOCIATIVE_LOGICAL_TAGS),
    "verb": lambda: random.choice(VERBS),
    "adjective": lambda: random.choice(ADJECTIVES),
    "adverb": lambda: random.choice(ADVERBS),
    "color": lambda: random.choice(COLORS),
    "size": lambda: random.choice(SIZE),
    "scope": lambda: random.choice(SCOPE),
}


@dataclass
class SegmentedCaption:
    """Represents a caption with its segments and masked versions"""
    full_caption: str
    segments: List[Dict]  # List of {tokens, token_ids, category, masked_ids}
    total_tokens: int
    categories_used: Set[str]

class SymbolicCaptionGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("AbstractPhil/bert-beatrix-2048", use_fast=True)
        self.segment_length = 77
        self.max_tokens = 2048

        # Initialize token mappings
        self.symbolic_tokens = BEATRIX_SPECIAL_TOKENS_AND_SHUNTS
        self.category_templates = CATEGORICAL_TEMPLATES


        # Masking parameters
        self.mask_prob = 0.30
        self.mask_token_id = self.tokenizer.mask_token_id

    def resolve_token(self, token: str, category: str) -> str:
        """Resolve a token placeholder with actual content from your lists"""
        token_map = FULL_TOKEN_MAP

        resolver = token_map.get(token)
        if resolver:
            return resolver()
        return token

    def _extract_object_side(self, side: str = "left") -> str:
        """Extract object from FULL_ASSOCIATIVE"""
        return extract_offset(side)





    def generate_category_focused_caption(self, primary_category: str,
                                        secondary_categories: Optional[List[str]] = None,
                                        ensure_shunt_mapping: bool = True) -> str:
        """Generate a caption focused on a specific category with shunt mapping"""
        if primary_category not in self.category_templates:
            raise ValueError(f"Unknown category: {primary_category}")

        # Get shunt ID for primary category
        shunt_id = self.symbolic_tokens[primary_category]

        # Select template for primary category
        template = random.choice(self.category_templates[primary_category])

        # Add secondary category elements if specified
        if secondary_categories:
            for sec_cat in secondary_categories[:2]:  # Limit to 2 secondary
                if sec_cat in self.category_templates and random.random() > 0.5:
                    sec_template = random.choice(self.category_templates[sec_cat])
                    template += f", {sec_template}"

        # Resolve all placeholders
        def replace_placeholder(match):
            placeholder = match.group(1)
            return self.resolve_token(placeholder, primary_category)

        caption = re.sub(r'\{(\w+)\}', replace_placeholder, template)

        # Add symbolic token and shunt ID at the beginning if ensuring mapping
        if ensure_shunt_mapping:
            caption = f"{primary_category} [SHUNT_{shunt_id}] {caption}"
        else:
            caption = f"{primary_category} {caption}"

        return caption

    def tokenize_and_segment(self, caption: str, primary_category: str) -> List[Dict]:
        """Tokenize caption and split into 77-token segments"""
        # Tokenize the full caption
        tokens = self.tokenizer.tokenize(caption)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        segments = []
        for i in range(0, len(token_ids), self.segment_length):
            segment_ids = token_ids[i:i + self.segment_length]
            segment_tokens = tokens[i:i + self.segment_length]

            # Pad if necessary
            if len(segment_ids) < self.segment_length:
                padding_length = self.segment_length - len(segment_ids)
                segment_ids.extend([self.tokenizer.pad_token_id] * padding_length)
                segment_tokens.extend([self.tokenizer.pad_token] * padding_length)

            segments.append({
                'tokens': segment_tokens,
                'token_ids': segment_ids,
                'start_idx': i,
                'end_idx': min(i + self.segment_length, len(token_ids)),
                'category': primary_category,
                'shunt_id': primary_category, #todo make this function when we have shunt mapping
            })

        return segments

    def create_masked_version(self, token_ids: List[int], mask_prob: float = None) -> Tuple[List[int], List[int]]:
        """Create masked version of token IDs for MLM training"""
        mask_prob = mask_prob or self.mask_prob
        masked_ids = token_ids.copy()
        labels = [-100] * len(token_ids)  # -100 = ignore in loss

        # Don't mask special tokens or padding
        special_tokens = {self.tokenizer.pad_token_id, self.tokenizer.cls_token_id,
                         self.tokenizer.sep_token_id}

        for i, token_id in enumerate(token_ids):
            if token_id not in special_tokens and random.random() < mask_prob:
                labels[i] = token_ids[i]

                # 80% mask token
                if random.random() < 0.3:
                    masked_ids[i] = self.mask_token_id
                # 10% random token
                elif random.random() < 0.1:
                    masked_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
                # 10% keep original

        return masked_ids, labels

    def generate_training_sample(
        self,
        primary_category: Optional[str] = None,
        secondary_categories: Optional[List[str]] = None,
        num_captions: int = 26,
    ) -> SegmentedCaption:
        """
        Build *exactly* `num_captions` (default = 5) short captions, each driven
        by a distinct category template. The first caption’s category becomes
        the shunt anchor.
        """
        # -------- category selection ------------------------------------------------
        if primary_category is None:
            primary_category = random.choice(list(self.category_templates.keys()))

        chosen: List[str] = [primary_category]

        # user-provided secondary categories (if any) take priority
        if secondary_categories:
            for cat in secondary_categories:
                if cat not in chosen and cat in self.category_templates:
                    chosen.append(cat)

        # randomly fill up to the requested count
        remaining = [c for c in self.category_templates if c not in chosen]
        while len(chosen) < num_captions and remaining:
            cat = random.choice(remaining)
            remaining.remove(cat)
            chosen.append(cat)

        # -------- build individual caption strings ----------------------------------
        caption_snippets: List[str] = []
        for cat in chosen:
            tmpl = random.choice(self.category_templates[cat])

            def _sub(m):                       # local resolver
                return self.resolve_token(m.group(1), cat)
            resolved = re.sub(r"\{(\w+)\}", _sub, tmpl)
            caption_snippets.append(resolved)

        # join with your canonical delimiter
       # shunt_id = self.symbolic_tokens[primary_category]
        full_caption = (
            f"{primary_category} [PAD]" +
            "., ".join(caption_snippets)      # << five discrete captions
        )

        # -------- tokenise & segment -----------------------------------------------
        segments = self.tokenize_and_segment(full_caption, primary_category)
        for seg in segments:
            m, l = self.create_masked_version(seg["token_ids"])
            seg["masked_ids"], seg["labels"] = m, l

        return SegmentedCaption(
            full_caption=full_caption,
            segments=segments,
            total_tokens=sum(len(s["token_ids"]) for s in segments),
            categories_used=set(chosen),
        )


    def generate_batch(self, batch_size: int,
                      category_distribution: Optional[Dict[str, float]] = None) -> List[SegmentedCaption]:
        """Generate a batch of training samples with specified category distribution"""
        if not category_distribution:
            # Equal distribution across all 26 categories
            categories = list(self.symbolic_tokens.keys())
            category_distribution = {cat: 1.0 / len(categories) for cat in categories}

        batch = []
        for _ in range(batch_size):
            # Sample primary category based on distribution
            primary_category = np.random.choice(
                list(category_distribution.keys()),
                p=list(category_distribution.values())
            )

            # Randomly select 0-2 secondary categories
            other_categories = [c for c in category_distribution.keys() if c != primary_category]
            num_secondary = random.randint(0, min(0, len(other_categories)))
            secondary_categories = random.sample(other_categories, num_secondary) if num_secondary > 0 else None

            sample = self.generate_training_sample(primary_category, secondary_categories)
            batch.append(sample)

        return batch

    def prepare_for_training(self, samples: List[SegmentedCaption]) -> Dict:
        """Prepare samples for training in the format expected by the model"""
        all_segments = []
        all_masked_segments = []
        all_labels = []
        all_categories = []
        all_shunt_ids = []

        for sample in samples:
            for segment in sample.segments:
                all_segments.append(segment['token_ids'])
                all_masked_segments.append(segment['masked_ids'])
                all_labels.append(segment['labels'])
                all_categories.append(segment['category'])
                all_shunt_ids.append(segment['shunt_id'])

        return {
            'input_ids': np.array(all_segments),
            'masked_input_ids': np.array(all_masked_segments),
            'labels': np.array(all_labels),
            'category_names': all_categories,
            'shunt_ids': np.array(all_shunt_ids),
            'attention_mask': np.ones_like(all_segments)  # Adjust for padding
        }

# Example usage with all 26 categories
def generate_training_data_all_categories(num_samples: int = 1000000):
    """Generate training data for all 26 category shunts"""
    generator = SymbolicCaptionGenerator()

    # Define equal distribution across all 26 categories
    all_categories = list(generator.symbolic_tokens.keys())
    category_distribution = {cat: 1.0 / len(all_categories) for cat in all_categories}

    # Or use custom distribution favoring certain categories
    custom_distribution = {
        "<subject>": 0.02,
        "<subject1>": 0.01,
        "<subject2>": 0.01,
        "<pose>": 0.08,
        "<emotion>": 0.05,
        "<surface>": 0.05,
        "<lighting>": 0.05,
        "<material>": 0.05,
        "<accessory>": 0.05,
        "<footwear>": 0.05,
        "<upper_body_clothing>": 0.05,
        "<hair_style>": 0.05,
        "<hair_length>": 0.05,
        "<headwear>": 0.05,
        "<texture>": 0.05,
        "<pattern>": 0.05,
        "<grid>": 0.05,
        "<zone>": 0.05,
        "<offset>": 0.05,
        "<object_left>": 0.05,
        "<object_right>": 0.05,
        "<relation>": 0.05,
        "<intent>": 0.05,
        "<style>": 0.05,
        "<fabric>": 0.05,
        "<jewelry>": 0.05
    }

    # Verify distribution sums to 1.0
    assert abs(sum(custom_distribution.values()) - 1.0) < 0.001, "Distribution must sum to 1.0"

    # Generate in batches for efficiency
    batch_size = 5000
    num_batches = num_samples // batch_size

    all_training_data = []

    for i in range(num_batches):
        batch = generator.generate_batch(batch_size, custom_distribution)
        training_data = generator.prepare_for_training(batch)
        all_training_data.append(training_data)

        if i % 100 == 0:
            print(f"Generated {i * batch_size} samples...")
            # Show distribution of shunts used
            shunt_counts = {}
            for shunt_id in training_data['shunt_ids']:
                shunt_counts[shunt_id] = shunt_counts.get(shunt_id, 0) + 1
            print(f"Shunt distribution in batch: {shunt_counts}")

    return all_training_data

# Test the implementation with all categories
if __name__ == "__main__":
    generator = SymbolicCaptionGenerator()

    # Test each category
    print("Testing all 26 categories:\n")
    for category in generator.symbolic_tokens.keys():
        sample = generator.generate_training_sample()
        print(f"{category} (Shunt {generator.symbolic_tokens[category]}):")
        print(f"  Caption: {sample.full_caption}...")
        print(f"  Tokens: {sample.total_tokens}, Segments: {len(sample.segments)}\n")