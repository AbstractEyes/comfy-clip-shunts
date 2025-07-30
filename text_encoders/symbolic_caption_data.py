import random
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from transformers import AutoTokenizer
import numpy as np

from .symbolic_bulk_captions import *

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
        self.category_templates = self._initialize_category_templates()

        # Masking parameters
        self.mask_prob = 0.30
        self.mask_token_id = self.tokenizer.mask_token_id

    def _initialize_category_templates(self) -> Dict[str, List[str]]:
        """Initialize templates for all 26 categories with expanded diversity"""
        return {
            # 1. Subject tokens (3 variations)
            "<subject>": [
                "a {gender} {pose} {offset}",
                "a {gender} wearing {upper_clothing} while {pose}",
                "a {gender} with {hair_style} hair, {pose} {offset}",
                "{gender} in {zone} {pose}",
                "a {gender} holding {object_right} while {pose}",
                "detailed portrait of {gender} with {emotion} expression, {pose} near {surface}",
                "{gender} dressed in {material} {upper_clothing}, {pose} under {lighting}",
                "a {gender} with {hair_length} {hair_style} hair wearing {accessory}, {pose} {offset}",
                "figure of {gender} {pose} on {surface}, wearing {footwear} and {jewelry}",
                "{gender} displaying {emotion} while {pose}, {upper_clothing} made of {fabric}",
                "compositional study of {gender} {pose} in {zone}, illuminated by {lighting}",
                "a {gender} with {texture} {material} clothing, {pose} {relation} {object_right}",
                "{gender} featuring {pattern} {upper_clothing} and {headwear}, {pose} on {surface}",
                "artistic depiction of {gender} {pose}, wearing {jewelry} and {accessory}, {offset}",
                "stylized {gender} with {hair_length} hair styled in {hair_style}, {pose} near {object_left}"
            ],

            "<subject1>": [
                "first {gender} {pose} near {object_right}",
                "primary {gender} wearing {upper_clothing} {offset}",
                "main {gender} with {hair_style} hair on {surface}",
                "foreground {gender} {pose} with {accessory}",
                "leading {gender} in {material} clothing",
                "central {gender} displaying {emotion} expression, {pose} under {lighting}",
                "primary figure wearing {fabric} {upper_clothing} with {pattern}, {pose} in {zone}",
                "main subject with {hair_length} {hair_style} hair and {jewelry}, {pose} on {surface}",
                "first person dressed in {material} outfit with {footwear}, {pose} {relation} {object_right}",
                "foreground {gender} with {headwear} and {accessory}, {pose} {offset}",
                "primary {gender} showing {texture} clothing details, {pose} near {object_left}",
                "leading figure in {pattern} {upper_clothing}, {pose} illuminated by {lighting}",
                "main {gender} with {emotion} mood wearing {jewelry}, {pose} on {surface}",
                "central subject featuring {fabric} garments and {footwear}, {pose} in {zone}",
                "primary {gender} styled with {hair_style} and {accessory}, {pose} {offset}"
            ],

            "<subject2>": [
                "second {gender} {pose} {relation} first person",
                "another {gender} wearing {footwear} {offset}",
                "background {gender} with {emotion} expression",
                "accompanying {gender} holding {accessory}",
                "secondary {gender} on {surface}",
                "additional {gender} dressed in {material} {upper_clothing}, {pose} in {zone}",
                "second figure with {hair_length} {hair_style} hair, {pose} under {lighting}",
                "companion {gender} wearing {pattern} clothing and {jewelry}, {pose} {offset}",
                "background person showing {emotion} while {pose}, dressed in {fabric} garments",
                "secondary subject with {headwear} and {footwear}, {pose} near {object_right}",
                "another figure displaying {texture} {upper_clothing}, {pose} on {surface}",
                "accompanying {gender} with {accessory} and {jewelry}, {pose} {relation} main subject",
                "second person featuring {hair_style} hairstyle and {emotion} expression, {pose} in {zone}",
                "additional {gender} in {material} outfit with {pattern}, {pose} {offset}",
                "secondary figure wearing {fabric} clothing and {footwear}, {pose} under {lighting}"
            ],

            # 2. Pose
            "<pose>": [
                "{gender} {pose} near {object_right}",
                "person {pose} while wearing {upper_clothing}",
                "{gender} {pose} on {surface}",
                "figure {pose} under {lighting}",
                "{pose} position with {accessory}",
                "dynamic {pose} captured {offset}, wearing {material} clothing",
                "expressive {pose} on {surface}, illuminated by {lighting}",
                "{gender} demonstrating {pose} with {emotion} expression, near {object_left}",
                "graceful {pose} position wearing {fabric} {upper_clothing} and {footwear}",
                "athletic {pose} in {zone}, accessorized with {jewelry} and {accessory}",
                "contemplative {pose} {relation} {object_right}, dressed in {pattern} garments",
                "energetic {pose} under {lighting}, featuring {hair_style} hairstyle",
                "relaxed {pose} on {surface} with {texture} surroundings",
                "dramatic {pose} {offset}, wearing {headwear} and {upper_clothing}",
                "subtle {pose} gesture with {emotion} mood, adorned with {jewelry}"
            ],

            # 3. Emotion
            "<emotion>": [
                "{gender} looking {emotion} while {pose}",
                "a {emotion} {gender} {offset}",
                "{gender} with {emotion} expression",
                "{emotion} mood in {lighting}",
                "displaying {emotion} near {object_right}",
                "profound {emotion} expression captured on {gender}'s face while {pose}",
                "subtle {emotion} mood enhanced by {lighting} on {surface}",
                "{gender} conveying {emotion} through {pose}, wearing {upper_clothing}",
                "intense {emotion} displayed {offset}, with {hair_style} hair flowing",
                "nuanced {emotion} expression paired with {accessory} and {jewelry}",
                "{emotion} atmosphere created by {gender} {pose} near {object_left}",
                "complex {emotion} state shown through {pose} and {material} clothing",
                "genuine {emotion} moment captured in {zone} under {lighting}",
                "layered {emotion} expression with {pattern} {upper_clothing} and {footwear}",
                "evocative {emotion} presence {relation} {object_right}, {pose} position"
            ],

            # 4. Surface
            "<surface>": [
                "{gender} {pose} on {surface}",
                "{object_left} placed on {surface}",
                "a {surface} with {texture}",
                "{surface} supporting {object_right}",
                "{material} {surface} in {zone}",
                "weathered {surface} displaying {texture} beneath {gender} who {pose}",
                "polished {surface} reflecting {lighting}, supporting {object_left} and {object_right}",
                "textured {surface} with {pattern} details, {gender} {pose} upon it",
                "sturdy {surface} made of {material}, decorated with {accessory}",
                "elegant {surface} in {zone}, illuminated by {lighting} from above",
                "rustic {surface} showing {texture} patterns, {relation} {object_right}",
                "modern {surface} with {material} finish, supporting {gender} who {pose}",
                "ancient {surface} bearing {pattern} markings, {offset} in composition",
                "functional {surface} holding {object_left} and {accessory}, {lighting} enhanced",
                "decorative {surface} with {fabric} covering, positioned in {zone}"
            ],

            # 5. Lighting
            "<lighting>": [
                "scene illuminated by {lighting}",
                "{gender} under {lighting} while {pose}",
                "{lighting} casting shadows on {surface}",
                "{lighting} highlighting {texture}",
                "ambient {lighting} in {zone}",
                "dramatic {lighting} creating depth around {gender} who {pose} on {surface}",
                "soft {lighting} filtering through, highlighting {material} {upper_clothing}",
                "harsh {lighting} defining {texture} on {object_right} and {surface}",
                "natural {lighting} bathing {gender} with {hair_style} hair in warm glow",
                "artificial {lighting} emphasizing {pattern} on {fabric} clothing",
                "moody {lighting} setting atmosphere for {emotion} expression {offset}",
                "directional {lighting} sculpting {gender}'s {pose} near {object_left}",
                "diffused {lighting} softening {texture} details on {accessory} and {jewelry}",
                "contrasting {lighting} in {zone}, creating visual interest on {surface}",
                "cinematic {lighting} enhancing {material} properties of {upper_clothing}"
            ],

            # 6. Material
            "<material>": [
                "{upper_clothing} made of {material}",
                "a {material} {object_right}",
                "{gender} wearing {material} clothing",
                "{material} texture on {surface}",
                "luxurious {material} {accessory}",
                "refined {material} used in {upper_clothing} with {pattern} design",
                "raw {material} forming {object_left} {relation} {object_right}",
                "processed {material} creating {texture} on {surface} in {zone}",
                "synthetic {material} in {footwear} and {accessory} combination",
                "organic {material} draped as {upper_clothing}, {pose} enhancing flow",
                "composite {material} with {fabric} blend in {jewelry} and {headwear}",
                "traditional {material} worked into {pattern} for {gender}'s outfit",
                "modern {material} treatment on {surface} under {lighting}",
                "weathered {material} showing age on {object_right} {offset}",
                "polished {material} reflecting {lighting} on {accessory} details"
            ],

            # 7. Accessory
            "<accessory>": [
                "{gender} wearing {accessory}",
                "a {accessory} beside {object_right}",
                "{gender} holding {accessory}",
                "{accessory} made of {material}",
                "decorative {accessory} {offset}",
                "ornate {accessory} crafted from {material} with {pattern} details",
                "functional {accessory} complementing {upper_clothing} on {gender}",
                "vintage {accessory} paired with {jewelry} and {footwear}",
                "modern {accessory} featuring {texture} finish under {lighting}",
                "handcrafted {accessory} {relation} {object_left} in {zone}",
                "designer {accessory} with {fabric} elements worn by {gender}",
                "traditional {accessory} displaying {pattern} while {pose}",
                "statement {accessory} contrasting with {material} {upper_clothing}",
                "subtle {accessory} enhancing {emotion} expression {offset}",
                "layered {accessory} arrangement with {jewelry} on {surface}"
            ],

            # 8. Footwear
            "<footwear>": [
                "{gender} wearing {footwear}",
                "a pair of {footwear} {offset}",
                "{footwear} placed near {object_left}",
                "{material} {footwear} on {surface}",
                "stylish {footwear} in {zone}",
                "worn {footwear} made of {material} with {texture} details",
                "pristine {footwear} complementing {upper_clothing} and {accessory}",
                "custom {footwear} featuring {pattern} design under {lighting}",
                "practical {footwear} suited for {pose} on {surface}",
                "elegant {footwear} paired with {jewelry} and {fabric} garments",
                "weathered {footwear} telling stories, {relation} {object_right}",
                "designer {footwear} with {material} construction in {zone}",
                "comfortable {footwear} supporting {gender} while {pose}",
                "decorative {footwear} with {texture} embellishments {offset}",
                "functional {footwear} contrasting with formal {upper_clothing}"
            ],

            # 9. Upper body clothing
            "<upper_body_clothing>": [
                "{gender} wearing {upper_clothing}",
                "a {upper_clothing} {offset}",
                "{upper_clothing} draped over {surface}",
                "{fabric} {upper_clothing} with {pattern}",
                "{upper_clothing} in {zone}",
                "tailored {upper_clothing} made from {material} with {texture} finish",
                "flowing {upper_clothing} adorned with {pattern} and {accessory}",
                "structured {upper_clothing} paired with {jewelry} and {footwear}",
                "casual {upper_clothing} in {fabric} displaying {emotion} mood",
                "formal {upper_clothing} illuminated by {lighting} on {surface}",
                "vintage {upper_clothing} with {pattern} details {relation} {object_left}",
                "contemporary {upper_clothing} featuring {material} blend in {zone}",
                "layered {upper_clothing} creating {texture} visual interest {offset}",
                "embellished {upper_clothing} with {jewelry} accents under {lighting}",
                "minimalist {upper_clothing} contrasting with ornate {headwear}"
            ],

            # 10. Hair style
            "<hair_style>": [
                "{gender} with {hair_style} {hair_length} hair",
                "a {gender} sporting {hair_style}",
                "{hair_style} hair styled with {accessory}",
                "{hair_style} under {headwear}",
                "elegant {hair_style} {offset}",
                "intricate {hair_style} adorned with {jewelry} and {accessory}",
                "natural {hair_style} flowing in {hair_length} waves under {lighting}",
                "styled {hair_style} complementing {upper_clothing} and {emotion} expression",
                "textured {hair_style} with {pattern} elements near {object_right}",
                "classic {hair_style} updated with modern {accessory} in {zone}",
                "windswept {hair_style} creating movement while {pose} on {surface}",
                "polished {hair_style} contrasting with {texture} {material} clothing",
                "casual {hair_style} paired with {headwear} and {jewelry}",
                "dramatic {hair_style} enhanced by {lighting} effects {offset}",
                "traditional {hair_style} with {hair_length} styling {relation} {object_left}"
            ],

            # 11. Hair length
            "<hair_length>": [
                "{gender} with {hair_length} {hair_style} hair",
                "{hair_length} hair flowing {offset}",
                "displaying {hair_length} locks",
                "{hair_length} hair under {lighting}",
                "{hair_length} strands with {texture}",
                "luxurious {hair_length} hair styled in {hair_style} with {accessory}",
                "natural {hair_length} tresses cascading over {upper_clothing}",
                "precisely cut {hair_length} hair framing {emotion} expression",
                "flowing {hair_length} locks enhanced by {lighting} on {surface}",
                "textured {hair_length} hair adorned with {jewelry} and {headwear}",
                "voluminous {hair_length} style creating silhouette in {zone}",
                "sleek {hair_length} hair contrasting with {pattern} {fabric} clothing",
                "windblown {hair_length} strands during {pose} {offset}",
                "carefully maintained {hair_length} hair {relation} {object_right}",
                "dramatic {hair_length} styling complementing {material} {upper_clothing}"
            ],

            # 12. Headwear
            "<headwear>": [
                "{gender} wearing {headwear}",
                "a {headwear} {offset}",
                "{headwear} placed on {surface}",
                "{material} {headwear} with {pattern}",
                "stylish {headwear} complementing {hair_style}",
                "traditional {headwear} crafted from {material} with {texture} details",
                "modern {headwear} adorned with {accessory} and {jewelry}",
                "functional {headwear} protecting from {lighting} in {zone}",
                "decorative {headwear} featuring {pattern} design on {fabric}",
                "vintage {headwear} paired with {upper_clothing} and {footwear}",
                "statement {headwear} creating focal point while {pose}",
                "subtle {headwear} enhancing {hair_length} {hair_style} arrangement",
                "weather-appropriate {headwear} on {surface} near {object_left}",
                "ceremonial {headwear} with {material} construction {offset}",
                "casual {headwear} contrasting formal {upper_clothing} ensemble"
            ],

            # 13. Texture
            "<texture>": [
                "{surface} with {texture} finish",
                "{object_left} showing {texture}",
                "a {texture} pattern on {material}",
                "{texture} detail under {lighting}",
                "rich {texture} in {zone}",
                "complex {texture} created by {material} on {surface} under {lighting}",
                "subtle {texture} variations on {upper_clothing} and {accessory}",
                "pronounced {texture} contrasting smooth {object_right} in {zone}",
                "layered {texture} effects on {fabric} {pattern} design",
                "natural {texture} enhanced by weathering on {surface} {offset}",
                "artificial {texture} mimicking organic patterns on {footwear}",
                "varied {texture} creating visual interest {relation} {object_left}",
                "uniform {texture} across {material} {headwear} and {jewelry}",
                "rough {texture} juxtaposed with polished {accessory} details",
                "delicate {texture} revealed by {lighting} on {upper_clothing}"
            ],

            # 14. Pattern
            "<pattern>": [
                "{upper_clothing} with {pattern} design",
                "a {pattern} {material} {object_right}",
                "{pattern} covering {surface}",
                "intricate {pattern} on {fabric}",
                "{pattern} motif {offset}",
                "repeating {pattern} across {material} {upper_clothing} and {accessory}",
                "organic {pattern} inspired by nature on {surface} in {zone}",
                "geometric {pattern} creating rhythm on {fabric} {footwear}",
                "traditional {pattern} updated for modern {headwear} design",
                "abstract {pattern} enhanced by {lighting} on {texture} surface",
                "cultural {pattern} adorning {jewelry} and {upper_clothing}",
                "minimalist {pattern} contrasting busy {object_left} arrangement",
                "bold {pattern} making statement on {material} garment {offset}",
                "subtle {pattern} revealed under close inspection of {accessory}",
                "layered {pattern} combinations creating depth {relation} {object_right}"
            ],

            # 15. Grid
            "<grid>": [
                "composition following {grid} layout",
                "{gender} positioned on {grid}",
                "elements arranged in {grid}",
                "{grid} structure in {zone}",
                "visual {grid} with {object_left} and {object_right}",
                "precise {grid} alignment of {gender} {pose} with {surface} elements",
                "dynamic {grid} breaking traditional rules in {zone} placement",
                "harmonious {grid} balancing {object_left} and {object_right} {offset}",
                "mathematical {grid} underlying {pattern} on {material} surface",
                "intuitive {grid} guiding eye through {lighting} and shadow",
                "classical {grid} proportions for {gender} wearing {upper_clothing}",
                "modern {grid} interpretation with {accessory} as focal point",
                "organic {grid} suggested by natural {texture} arrangements",
                "rigid {grid} softened by {fabric} draping and {pose}",
                "conceptual {grid} relating {emotion} to spatial {relation}"
            ],

            # 16. Zone
            "<zone>": [
                "{gender} positioned in {zone}",
                "activity happening in {zone}",
                "{object_left} located in {zone}",
                "focus on {zone} area",
                "{lighting} illuminating {zone}",
                "primary action occurring in {zone} with {gender} {pose}",
                "secondary elements arranged in {zone} around {object_right}",
                "visual weight concentrated in {zone} through {lighting} placement",
                "negative space defining {zone} boundaries near {surface}",
                "compositional {zone} emphasized by {pattern} and {texture}",
                "foreground {zone} featuring {upper_clothing} and {accessory} details",
                "background {zone} providing context with {object_left} placement",
                "transitional {zone} linking elements through {material} continuity",
                "isolated {zone} creating focus on {emotion} expression {offset}",
                "interconnected {zone} relating {footwear} to {surface} interaction"
            ],

            # 17. Offset
            "<offset>": [
                "{gender} {offset}",
                "{object_right} placed {offset}",
                "scene captured {offset}",
                "composition {offset}",
                "elements arranged {offset}",
                "strategic placement {offset} creating visual tension with {object_left}",
                "balanced arrangement {offset} despite asymmetrical {pose}",
                "dramatic positioning {offset} enhanced by {lighting} direction",
                "subtle shift {offset} revealing {texture} on {surface}",
                "intentional framing {offset} emphasizing {pattern} details",
                "dynamic capture {offset} showing movement in {upper_clothing}",
                "classical placement {offset} following {grid} principles",
                "unexpected angle {offset} revealing hidden {accessory} details",
                "harmonious positioning {offset} uniting {zone} elements",
                "thoughtful arrangement {offset} guiding viewer through {emotion}"
            ],

            # 18. Object left
            "<object_left>": [
                "{object_left} on the left side",
                "{object_left} {relation} {object_right}",
                "{object_left} made of {material}",
                "prominent {object_left} in {zone}",
                "{object_left} under {lighting}",
                "carefully placed {object_left} with {texture} surface in {zone}",
                "weathered {object_left} showing {pattern} from use over time",
                "functional {object_left} serving purpose {relation} {gender}",
                "decorative {object_left} crafted from {material} with {fabric} accents",
                "symbolic {object_left} representing {emotion} in composition",
                "vintage {object_left} contrasting modern {object_right} {offset}",
                "organic {object_left} complementing structured {surface} geometry",
                "illuminated {object_left} catching {lighting} dramatically",
                "textured {object_left} providing tactile interest near {accessory}",
                "minimal {object_left} balancing ornate {upper_clothing} details"
            ],

            # 19. Object right
            "<object_right>": [
                "{object_right} on the right side",
                "{object_right} near {gender}",
                "{object_right} with {texture}",
                "decorative {object_right} {offset}",
                "{object_right} on {surface}",
                "significant {object_right} made from {material} in {zone}",
                "artistic {object_right} displaying {pattern} under {lighting}",
                "practical {object_right} used by {gender} while {pose}",
                "antique {object_right} with {texture} patina on {surface}",
                "contemporary {object_right} featuring {fabric} elements",
                "natural {object_right} {relation} manufactured {object_left}",
                "polished {object_right} reflecting surrounding {lighting} effects",
                "weathered {object_right} telling story through wear {offset}",
                "geometric {object_right} following {grid} placement rules",
                "organic {object_right} softening rigid {pattern} arrangements"
            ],

            # 20. Relation
            "<relation>": [
                "{object_left} {relation} {object_right}",
                "{gender} {relation} {surface}",
                "{accessory} {relation} {object_right}",
                "spatial {relation} between elements",
                "{relation} positioning in {zone}",
                "dynamic {relation} created between {gender} and {object_left} through {pose}",
                "harmonic {relation} linking {upper_clothing} to {surface} textures",
                "contrasting {relation} between {material} and {fabric} elements",
                "subtle {relation} suggested by {lighting} connecting distant objects",
                "physical {relation} demonstrated through {footwear} contact with {surface}",
                "visual {relation} established via {pattern} continuity across {zone}",
                "emotional {relation} between {emotion} expression and {object_right}",
                "compositional {relation} following {grid} to link {accessory} placement",
                "temporal {relation} implied between weathered {object_left} and new {jewelry}",
                "conceptual {relation} uniting {texture} variations {offset}"
            ],

            # 21. Intent
            "<intent>": [
                "creating {intent} mood",
                "{intent} purpose with {emotion}",
                "conveying {intent} through {pose}",
                "{intent} narrative in scene",
                "artistic {intent} {offset}",
                "deliberate {intent} expressed through {gender}'s {pose} and {emotion}",
                "subtle {intent} woven into {pattern} and {material} choices",
                "powerful {intent} communicated via {lighting} on {surface}",
                "layered {intent} revealed through {upper_clothing} and {accessory} symbolism",
                "cultural {intent} embedded in {jewelry} and {headwear} selection",
                "personal {intent} manifested in {hair_style} and {footwear} styling",
                "universal {intent} transcending specific {zone} placement",
                "complex {intent} requiring contemplation of {object_left} {relation} {object_right}",
                "immediate {intent} apparent in {texture} and {fabric} contrasts",
                "evolving {intent} suggested by transitional {lighting} {offset}"
            ],

            # 22. Style
            "<style>": [
                "rendered in {style} aesthetic",
                "{style} artistic approach",
                "{style} treatment of {lighting}",
                "distinctive {style} composition",
                "{style} interpretation {offset}",
                "refined {style} evident in {gender}'s {pose} and {upper_clothing} selection",
                "bold {style} expressed through {pattern} and {material} combinations",
                "subtle {style} nuances in {texture} treatment on {surface}",
                "period {style} accuracy in {headwear} and {footwear} details",
                "contemporary {style} merging with traditional {jewelry} elements",
                "experimental {style} pushing boundaries of {lighting} and {zone}",
                "classical {style} principles applied to modern {accessory} arrangement",
                "signature {style} recognizable in {emotion} portrayal and {hair_style}",
                "evolving {style} blending multiple influences in {fabric} choices",
                "cohesive {style} unifying disparate elements through {grid} structure"
            ],

            # 23. Fabric
            "<fabric>": [
                "{upper_clothing} made from {fabric}",
                "luxurious {fabric} {accessory}",
                "{fabric} draped over {surface}",
                "soft {fabric} with {pattern}",
                "{fabric} material in {zone}",
                "premium {fabric} woven with {material} threads creating {texture}",
                "delicate {fabric} flowing around {gender} during {pose}",
                "structured {fabric} maintaining form in {upper_clothing} design",
                "vintage {fabric} showing {pattern} popular in past eras",
                "innovative {fabric} blend combining natural and synthetic {material}",
                "handwoven {fabric} displaying artisanal {texture} under {lighting}",
                "sustainable {fabric} used in {footwear} and {accessory} construction",
                "traditional {fabric} treatment creating unique {pattern} {offset}",
                "modern {fabric} technology enabling {emotion} through drape",
                "layered {fabric} creating depth {relation} {surface} backdrop"
            ],

            # 24. Jewelry
            "<jewelry>": [
                "{gender} wearing {jewelry}",
                "elegant {jewelry} {offset}",
                "{jewelry} paired with {upper_clothing}",
                "sparkling {jewelry} under {lighting}",
                "{material} {jewelry} as accent",
                "heirloom {jewelry} crafted from {material} with {pattern} engravings",
                "contemporary {jewelry} complementing traditional {upper_clothing} style",
                "statement {jewelry} creating focal point against {fabric} backdrop",
                "delicate {jewelry} catching {lighting} with subtle sparkle",
                "layered {jewelry} arrangement enhancing {emotion} expression",
                "cultural {jewelry} representing heritage worn with {headwear}",
                "minimalist {jewelry} balancing ornate {accessory} details",
                "vintage {jewelry} showing {texture} from years of wear",
                "custom {jewelry} designed to match {footwear} embellishments",
                "symbolic {jewelry} placed strategically in {zone} for meaning"
            ],

            ## 25. Lower body clothing (adding this as it seems to be missing)
            #"<lower_body_clothing>": [
            #    "{gender} wearing {lower_clothing} with {pattern}",
            #    "flowing {lower_clothing} made of {fabric} {offset}",
            #    "{material} {lower_clothing} paired with {footwear}",
            #    "tailored {lower_clothing} in {zone} under {lighting}",
            #    "casual {lower_clothing} with {texture} details on {surface}",
            #    "formal {lower_clothing} complementing {upper_clothing} ensemble",
            #    "vintage {lower_clothing} featuring {pattern} from bygone era",
            #    "contemporary {lower_clothing} with innovative {material} blend",
            #    "functional {lower_clothing} designed for {pose} flexibility",
            #    "decorative {lower_clothing} adorned with {accessory} elements",
            #    "layered {lower_clothing} creating visual interest through {fabric}",
            #    "structured {lower_clothing} maintaining silhouette while {pose}",
            #    "flowing {lower_clothing} enhanced by movement and {lighting}",
            #    "traditional {lower_clothing} updated with modern {pattern}",
            #    "minimalist {lower_clothing} allowing focus on {jewelry} details"
            #],
#
            ## 26. Background (adding this for environmental context)
            #"<background>": [
            #    "atmospheric {background} setting enhancing {emotion}",
            #    "{background} environment complementing {style} aesthetic",
            #    "detailed {background} with {texture} elements in {zone}",
            #    "minimal {background} focusing attention on {gender} {pose}",
            #    "complex {background} incorporating {object_left} and {object_right}",
            #    "natural {background} with organic {pattern} under {lighting}",
            #    "architectural {background} providing {grid} structure",
            #    "abstract {background} created through {material} and light",
            #    "historical {background} context for period {upper_clothing}",
            #    "futuristic {background} contrasting vintage {accessory} elements",
            #    "textured {background} adding depth behind {surface} placement",
            #    "gradient {background} transitioning through {zone} areas",
            #    "patterned {background} echoing {fabric} design motifs",
            #    "atmospheric {background} enhanced by {lighting} effects {offset}",
            #    "contextual {background} supporting narrative {intent}"
            #]
        }

    def resolve_token(self, token: str, category: str) -> str:
        """Resolve a token placeholder with actual content from your lists"""
        token_map = {
            # Gender and human attributes
            "gender": lambda: resolve_gender_token(random.choice(GENDER_TYPES)),
            "pose": lambda: random.choice(HUMAN_POSES),
            "emotion": lambda: random.choice([
                # Core emotions
                "happy", "sad", "thoughtful", "confident", "mysterious", "playful", "serene",
                "intense", "melancholic", "joyful", "contemplative",

                # Positive spectrum
                "content", "grateful", "peaceful", "hopeful", "inspired", "excited", "ecstatic",
                "relieved", "tender", "affectionate", "cheerful", "uplifted", "amused", "carefree",

                # Negative spectrum
                "angry", "anxious", "fearful", "ashamed", "bitter", "jealous", "regretful", "resentful",
                "insecure", "lonely", "desperate", "guilty", "grieving", "disappointed", "frustrated",

                # Neutral / Ambiguous
                "neutral", "pensive", "stoic", "apathetic", "ambivalent", "indifferent", "tired",
                "detached", "blank", "uncertain",

                # Expressive or performative
                "flirtatious", "sly", "defiant", "proud", "sarcastic", "smug", "teasing", "teary",
                "bashful", "shy", "awkward", "curious", "startled", "embarrassed",

                # Elevated / rare
                "euphoric", "vindicated", "spiteful", "overwhelmed", "awe-struck", "tranquil",
                "reverent", "haunted", "devoted", "wistful", "mournful", "cathartic"
            ]),

            # Positioning and composition
            "offset": lambda: random.choice(OFFSET_TAGS).replace("*", random.choice(["viewed", "seen", "captured"])),
            "zone": lambda: random.choice(ZONE_TAGS) if 'ZONE_TAGS' in globals() else random.choice([
                "left side", "center", "right side", "foreground", "background",
                "upper third", "lower third", "middle ground", 'depicted-up', 'depicted-down', 'depicted-left', 'depicted-right',
                'left-up', 'left-down', 'left-left', 'left-right', 'right-up', 'right-down', 'right-left', 'right-right',
                'center-up', 'center-down', 'center-left', 'center-right', 'middle-up', 'middle-down', 'middle-left', 'middle-right',
                'top-left', 'top-right', 'bottom-left', 'bottom-right',
                'top-center', 'bottom-center', 'left-center', 'right-center'

            ]),
            "grid": lambda: random.choice(GRID_TAGS) if 'GRID_TAGS' in globals() else random.choice([
                "3x3 grid", "center point", "rule of thirds", "golden ratio",
                "diagonal composition", "symmetrical layout", "asymmetrical balance",
                "5x5 grid", "6x6 grid", "7x7 grid", "8x8 grid", "9x9 grid",
                "grid", "rule of 3", "rule of 5", "rule of 7", "rule of 9",
                "grid_a1", "grid_a2", "grid_a3", "grid_a4", "grid_a5",
                "grid_b1", "grid_b2", "grid_b3", "grid_b4", "grid_b5",
                "grid_c1", "grid_c2", "grid_c3", "grid_c4", "grid_c5",
                "grid_d1", "grid_d2", "grid_d3", "grid_d4", "grid_d5",
                "grid_e1", "grid_e2", "grid_e3", "grid_e4", "grid_e5",
            ]),
            "relation": lambda: random.choice([
                # Basic spatial relationships
                "next to", "beside", "on top of", "under", "to the right of", "to the left of",
                "above", "below", "in front of", "behind", "adjacent to",

                # Geometric / Directional
                "diagonally above", "diagonally below", "centered over", "off-center from",
                "between", "surrounding", "encircling", "aligned with", "opposite from", "mirrored by",

                # Touching / Contact
                "touching", "leaning against", "attached to", "stacked on", "resting against",
                "embedded in", "hooked onto", "hanging from", "sitting on",

                # Containment / Inclusion
                "inside", "outside of", "within", "encased in", "covered by", "enclosed within",
                "wrapped around", "nestled inside", "trapped under", "surrounded by",

                # Positional intent
                "leading", "following", "offset from", "hovering over", "drifting near",
                "partially covering", "peeking from behind", "projected onto",

                # Relational logic
                "subordinate to", "dominant over", "supporting", "obscuring", "revealed by",

                # Abstract / metaphorical (optional flair)
                "echoing", "reflecting", "shadowing", "contrasting with", "mimicking",
                "intertwined with", "linked to", "intersecting with"
            ]),


            # Clothing and accessories
            "upper_clothing": lambda: random.choice(UPPER_BODY_CLOTHES_TYPES),
            "footwear": lambda: random.choice(FOOTWEAR_TYPES),
            "accessory": lambda: random.choice(ACCESSORY_TYPES),
            "jewelry": lambda: random.choice(JEWELRY_TYPES),
            "headwear": lambda: random.choice(HEADWEAR_TYPES) if 'HEADWEAR_TYPES' in globals() else random.choice([
                # Common
                "hat", "cap", "beanie", "beret", "headband", "visor", "scarf", "bandana",
                "bucket hat", "snapback", "trucker hat", "fedora", "boater", "panama hat",
                "newsboy cap", "flat cap", "sun hat", "cloche", "bowler hat", "top hat",
                "bonnet", "balaclava", "hood", "earmuffs", "helmet", "turban",

                # Military/Tactical
                "combat helmet", "beret (military)", "shako", "bicorn", "tricorn", "pilot helmet",
                "kevlar helmet", "riot helmet", "garrison cap",

                # Cultural/Traditional
                "kufi", "tam", "keffiyeh", "ghutrah", "fez", "sombrero", "cowboy hat",
                "pith helmet", "sugegasa", "kasa", "pagri", "yarmulke", "shtreimel",
                "ushanka", "papakha",

                # Religious/Symbolic
                "bishop's mitre", "pope's tiara", "nun's coif", "monk hood", "hijab", "niqab",
                "veil", "priest biretta", "monastic hood",

                # Fantasy / Sci-fi / Style
                "tiara", "crown", "horned helmet", "wizard hat", "druid hood", "elven circlet",
                "steampunk goggles", "cyber visor", "space helmet", "dragon helm", "antler crown",
                "crystal crown", "halo", "digital interface helm", "energy visor"
            ]),

            # Hair attributes
            "hair_style": lambda: random.choice(HAIRSTYLES_TYPES),
            "hair_length": lambda: random.choice(HAIR_LENGTH_TYPES) if 'HAIR_LENGTH_TYPES' in globals() else random.choice([
                # Ultra-short
                "bald", "clean-shaven", "buzzed", "buzz cut", "stubble-length", "shaved sides", "crew cut", "fade cut",

                # Short styles
                "very short", "pixie-short", "pixie cut", "cropped", "ear-length", "temple-length", "sidecut short", "tapered",

                # Medium styles
                "chin-length", "jawline-length", "bob-length", "pageboy-length", "neck-length", "shoulder-length",
                "inverted bob", "lob-length", "curtain-length", "medium-layered",

                # Mid to long
                "collarbone-length", "upper-back-length", "bra-strap-length", "mid-back-length", "ribcage-length",
                "below-shoulder", "tied-back-length", "pulled-forward-length",

                # Long hair
                "waist-length", "belt-length", "hip-length", "tailbone-length", "thigh-length", "knee-length", "very long",

                # Extreme / Stylized
                "floor-length", "ankle-length", "calf-length", "dragging-length", "trailing-length",
                "gravity-defying", "floating-length", "looped-length", "sculpted-length",

                # Fantasy / Anime-inspired
                "ethereal-length", "supernatural-length", "twin-dragon-length", "wind-wrapped-length",
                "wing-length", "spellbound-length", "astral-length", "mythic-length",

                # Motion-based descriptors
                "swaying-length", "whipping-length", "flowing-length", "draped-length", "streaming-length",
                "twisting-length", "spiraling-length", "billowing-length", "coiled-length",

                # Cultural / ceremonial
                "samurai-length", "ritual-length", "ancestral-length", "battle-worn-length", "monastic-length"
            ]),


            # Materials and textures
            "material": lambda: random.choice(MATERIAL_TYPES),
            "fabric": lambda: random.choice(FABRIC_TYPES) if 'FABRIC_TYPES' in globals() else random.choice([
                "cotton", "linen", "silk", "wool", "denim", "leather", "canvas", "polyester",
                "nylon", "rayon", "spandex", "suede", "cashmere", "velvet", "satin", "tweed",
                "mesh", "lace", "organza", "chiffon", "tulle", "fleece", "terrycloth", "corduroy",
                "jacquard", "gabardine", "burlap", "neoprene", "lycra", "acrylic"
            ]),

            "texture": lambda: random.choice(TEXTURE_TAGS) if 'TEXTURE_TAGS' in globals() else random.choice([
                "smooth", "rough", "glossy", "matte", "metallic", "velvet", "satin", "leather",
                "wooden", "glass", "stone", "gritty", "bumpy", "pebbled", "cracked", "coarse",
                "silky", "sticky", "greasy", "fibrous", "crystalline", "slick", "powdery", "chipped",
                "ribbed", "brushed", "pitted", "etched", "polished", "weathered", "frosted"
            ]),

            "pattern": lambda: random.choice(PATTERN_TAGS) if 'PATTERN_TAGS' in globals() else random.choice([
                "striped", "checked", "floral", "geometric", "abstract", "paisley", "polka dot",
                "camouflage", "plaid", "herringbone", "chevron", "argyle", "animal print",
                "baroque", "tribal", "lacework", "zigzag", "marbled", "wave", "diamond", "leaf motif",
                "gradient", "burnout", "fractal", "chainlink", "scale pattern", "radial", "ink blot", "maze"
            ]),


            # Environment and lighting
            "surface": lambda: random.choice(HUMAN_SURFACES),
            "lighting": lambda: random.choice(LIGHTING_TYPES),

            # Objects
            "object_left": lambda: self._extract_object_side("left"),
            "object_right": lambda: self._extract_object_side("right"),

            # Style and intent
            "style": lambda: random.choice([
                # Core
                "photorealistic", "artistic", "minimalist", "dramatic", "cinematic", "vintage", "modern", "classical",
                "experimental", "documentary", "portrait",

                # Visual / Art Movement
                "surreal", "baroque", "rococo", "futurist", "brutalist", "art nouveau", "art deco",
                "postmodern", "constructivist", "expressionist", "impressionist", "cubist",
                "dadaist", "symbolist", "realist", "hyperrealistic", "pop art", "graffiti",

                # Medium-Based
                "oil painting", "digital illustration", "charcoal sketch", "ink drawing",
                "watercolor", "collage", "pastel", "pixel art", "low poly", "wireframe", "line art",

                # Photography / Film
                "noir", "monochrome", "sepia", "macro", "wide angle", "bokeh", "high contrast",
                "low light", "vintage film", "ultra HD", "soft focus", "long exposure",

                # Fashion / Editorial / Design
                "editorial", "runway", "street fashion", "industrial", "futuristic", "gothic",
                "cyberpunk", "steampunk", "biopunk", "dark academia", "light academia",
                "y2k", "vaporwave", "aesthetic core", "cozy", "boho", "urban", "eco-modern",

                # Genre-Fusion
                "mythological", "post-apocalyptic", "dreamlike", "otherworldly", "ritualistic", "religious iconography"
            ]),

            "intent": lambda: random.choice([
                # Core
                "emotional", "narrative", "aesthetic", "conceptual", "documentary", "expressive", "symbolic", "atmospheric",

                # Psychological / Emotional States
                "melancholic", "nostalgic", "euphoric", "haunting", "serene", "tense",
                "romantic", "tragic", "contemplative", "hopeful", "lonely", "reflective",
                "anxious", "exuberant", "playful", "sentimental", "stoic", "wistful",

                # Narrative Drivers
                "heroic", "mythic", "epic", "ritualistic", "transformational", "origin-focused",
                "coming of age", "rebirth", "sacrifice", "revelation", "mystery", "conflict-driven",
                "internal journey", "spiritual awakening", "moral tension",

                # Communication / Social
                "provocative", "political", "satirical", "educational", "cautionary", "persuasive",
                "journalistic", "testimonial", "allegorical", "activist",

                # Visual / Compositional
                "compositional study", "gesture-focused", "motion-driven", "light study", "texture-focused",
                "character centric", "environmental", "perspective-driven", "minimal narrative",

                # Conceptual / Meta
                "meta-narrative", "simulation", "deconstructed", "ritual subversion", "symbol-dense",
                "myth reimagined", "absurdist", "visual pun", "ontological reflection"
            ]),

        }

        resolver = token_map.get(token)
        if resolver:
            return resolver()
        return token

    def _extract_object_side(self, side: str = "left") -> str:
        """Extract object from FULL_ASSOCIATIVE"""
        if 'FULL_ASSOCIATIVE' in globals() and FULL_ASSOCIATIVE:
            entry = random.choice(FULL_ASSOCIATIVE)
            parts = entry.split()
            if len(parts) >= 3:
                return parts[1] if side == "left" else parts[-1]
        return "object"

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
                'shunt_id': primary_category
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