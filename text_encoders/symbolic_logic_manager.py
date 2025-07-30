
import re
import torch
import logging
from typing import List, Dict, Any, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer
from ..utils.rose_util import rose_score
from comfy.model_management import throw_exception_if_processing_interrupted

logger = logging.getLogger(__name__)

BEATRIX_SPECIAL_TOKENS_AND_SHUNTS = [
    "<subject>","<subject1>","<subject2>",
    "<pose>","<emotion>","<surface>",
    "<lighting>","<material>","<accessory>",
    "<footwear>", "<upper_body_clothing>","<hair_style>",
    "<hair_length>","<headwear>","<texture>",
    "<pattern>","<grid>","<zone>",
    "<offset>","<object_left>","<object_right>",
    "<relation>","<intent>","<style>",
    "<fabric>","<jewelry>",
    "[SHUNT_1000000]","[SHUNT_1000001]","[SHUNT_1000002]","[SHUNT_1000003]","[SHUNT_1000004]",
    "[SHUNT_1000005]","[SHUNT_1000006]","[SHUNT_1000007]","[SHUNT_1000008]","[SHUNT_1000009]","[SHUNT_1000010]",
    "[SHUNT_1000011]","[SHUNT_1000012]","[SHUNT_1000013]","[SHUNT_1000014]","[SHUNT_1000015]","[SHUNT_1000016]",
    "[SHUNT_1000017]","[SHUNT_1000018]","[SHUNT_1000019]","[SHUNT_1000020]","[SHUNT_1000021]","[SHUNT_1000022]",
    "[SHUNT_1000023]","[SHUNT_1000024]","[SHUNT_1000025]",
]

LOGIC_TOKENS = {"[SEP]", "[EACH]", "[OR]", "[AND]", "[MULT]", "[ADD]", "[SUB]", "[DIV]"}


REPLACE_WITH_SEP = {
    ".,|,.",
    ".,",
    "<w>",
    "</w>",
}


from transformers import PreTrainedModel, PreTrainedTokenizer


class SymbolicLogicManager:
    def __init__(
            self,
            base_prompt: str,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            pad_first: int = 2,
            min_length: int = 77,
            slice_length: int = 77,
            max_length: int = 2048,
            padding_token: str = "[PAD]",
            sep_token: str = "[SEP]",
            each_mode: str = "rigid",
            nudge_to_sentence: bool = True,
            special_tokens: List[str] = BEATRIX_SPECIAL_TOKENS_AND_SHUNTS,
            pbar= None
    ):
        self.raw_prompt = base_prompt
        self.model = model
        self.tokenizer = tokenizer

        self.pad_first = pad_first
        self.min_length = min_length
        self.slice_length = slice_length
        self.max_length = max_length
        self.padding_token = padding_token
        self.sep_token = sep_token
        self.each_mode = each_mode
        self.nudge = nudge_to_sentence

        self.special_tokens = set(special_tokens)
        self.logic_tokens = LOGIC_TOKENS
        self.replace_with_sep = REPLACE_WITH_SEP
        self.pbar = pbar

    def normalize_prompt(self, prompt: str) -> str:
        for token in self.replace_with_sep:
            prompt = prompt.replace(token, self.sep_token)
        return prompt

    def strip_unrecognized_specials(self, prompt: str) -> str:
        import re
        return prompt #re.sub(r"<[^>]+>|(\[[^\]]+\])", lambda m: m.group(0) if m.group(0) in self.special_tokens or m.group(
            #0) in self.logic_tokens else "", prompt)


    def replace_numeric_and_specials(self, prompt: str) -> str:
        words = prompt.split()
        clean = []
        for word in words:
            if word.isdigit():
                clean.append(self.padding_token)
            #@elif any(c for c in word if not c.isalnum() and c not in {".", ",", "|"}):
            #@    clean.append("[MASK]")
            else:
                clean.append(word)
        return " ".join(clean)

    def slice_prompt(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Converts the raw prompt into a list of (prompt_slice, metadata) tuples.
        This includes logic weighting and slice boundary control.
        """
        # Step 1: Normalize input
        prompt = self.raw_prompt
        logger.info(f"[DEBUG] Original prompt length: {len(prompt)}")
        logger.info(f"[DEBUG] Original prompt sample: {prompt[:100]}...")

        prompt = self.normalize_prompt(prompt)
        logger.info(f"[DEBUG] After normalize: {prompt[:100]}...")
        logger.info(f"[DEBUG] Count of [SEP] tokens: {prompt.count('[SEP]')}")

        #prompt = self.strip_unrecognized_specials(prompt)
        #prompt = self.replace_numeric_and_specials(prompt)
        logger.info(f"[DEBUG] After all preprocessing: {prompt[:100]}...")

        # Step 2: Split by logic-aware boundaries
        raw_segments = re.split(r"(\[SEP\]|\[EACH\]|\[AND\]|\[OR\]|\[MULT\]|\[ADD\]|\[SUB\]|\[DIV\])", prompt)
        logger.info(f"[DEBUG] Number of segments after split: {len(raw_segments)}")

        # First check for logic tokens
        has_logic_tokens = any(
            token in prompt for token in ["[SEP]", "[EACH]", "[AND]", "[OR]", "[MULT]", "[ADD]", "[SUB]", "[DIV]"])

        if has_logic_tokens:
            # Use existing logic for prompts with explicit logic tokens
            return self._slice_prompt_with_logic(prompt)
        else:
            # For prompts without logic tokens, chunk by token count
            return self._slice_prompt_by_tokens(prompt)

    def _slice_prompt_by_tokens(self, prompt: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Chunk prompt into slice_length-token segments.
        """
        # Tokenize the entire prompt first
        tokens = self.tokenizer.tokenize(prompt)

        slices = []

        # Process in chunks of slice_length tokens
        for i in range(0, len(tokens), self.slice_length):
            chunk_tokens = tokens[i:i + self.slice_length]

            # Convert back to text (approximately)
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)

            # Pad if needed (at token level)
            if len(chunk_tokens) < self.slice_length:
                # Pad with padding tokens
                padding_needed = self.slice_length - len(chunk_tokens)
                chunk_tokens.extend([self.padding_token] * padding_needed)
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)

            slices.append((chunk_text, {"weight": 1.0, "logic": []}))

        return slices

    def _slice_prompt_with_logic(self, prompt: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Original logic-based slicing, but fixed to respect token boundaries.
        """
        # Original splitting logic
        raw_segments = re.split(r"(\[SEP\]|\[EACH\]|\[AND\]|\[OR\]|\[MULT\]|\[ADD\]|\[SUB\]|\[DIV\])", prompt)

        segments = []
        logic_tokens = []

        for i, part in enumerate(raw_segments):
            if i % 2 == 0:  # Even indices are text segments
                if part.strip():
                    segments.append(part.strip())
            else:  # Odd indices are logic tokens
                logic_tokens.append(part)

        slices = []

        for i, segment in enumerate(segments):
            throw_exception_if_processing_interrupted()

            logic_token = logic_tokens[i] if i < len(logic_tokens) else None

            metadata = {
                "weight": 1.0,
                "logic": []
            }

            if logic_token:
                metadata["logic"].append(logic_token.strip("[]"))
                if logic_token in {"[MULT]", "[ADD]", "[SUB]", "[DIV]"}:
                    metadata["weight"] = self._compute_math(segment, logic_token.strip("[]"))

            # Tokenize to ensure we work with actual tokens
            segment_tokens = self.tokenizer.tokenize(segment)

            # Process in chunks of slice_length tokens
            segment_slices = []
            for j in range(0, len(segment_tokens), self.slice_length):
                chunk = segment_tokens[j:j + self.slice_length]

                # Pad if needed
                if len(chunk) < self.slice_length:
                    chunk.extend([self.padding_token] * (self.slice_length - len(chunk)))

                chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
                segment_slices.append((chunk_text, metadata.copy()))

            slices.extend(segment_slices)

            # Logic tokens like [AND]/[OR] imply changes across slices
            if logic_token == "[AND]":
                # Duplicate all slices from current segment
                for slice_text, slice_meta in segment_slices:
                    duplicate_meta = slice_meta.copy()
                    duplicate_meta["logic"] = ["AND_DUPLICATE"]
                    slices.append((slice_text, duplicate_meta))

            elif logic_token == "[OR]" and i + 1 < len(segments):
                # Add slices from next segment early
                next_segment = segments[i + 1]
                next_tokens = self.tokenizer.tokenize(next_segment)

                for j in range(0, len(next_tokens), self.slice_length):
                    chunk = next_tokens[j:j + self.slice_length]

                    if len(chunk) < self.slice_length:
                        chunk.extend([self.padding_token] * (self.slice_length - len(chunk)))

                    chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
                    or_meta = {"weight": 1.0, "logic": ["OR_CONTINUATION"]}
                    slices.append((chunk_text, or_meta))
            each_segments = []
            for i, segment in enumerate(segments):
                logic_token = logic_tokens[i] if i < len(logic_tokens) else None
                if logic_token == "[EACH]":
                    # Find [EACH] position in original prompt
                    each_content = self._extract_each_content(segment, len(segment))
                    each_segments.append((i, each_content))

            # Apply [EACH] mode to all slices
            if each_segments and self.each_mode != "omit":
                for seg_idx, each_content in each_segments:
                    # Apply to all slices after this segment
                    start_idx = sum(1 for j in range(seg_idx + 1))  # Calculate slice start index
                    remaining_slices = slices[start_idx:]
                    modified = self._handle_each_mode(remaining_slices, each_content, self.each_mode)
                    slices[start_idx:] = modified

        return slices

    def _compute_math(self, segment: str, mode: str = "MULT") -> float:
        import re
        nums = list(map(float, re.findall(r"\d+(?:\.\d+)?", segment)))
        if not nums:
            return 1.0

        if mode == "MULT":
            out = 1.0
            for n in nums: out *= n
            return out
        elif mode == "ADD":
            return sum(nums)
        elif mode == "SUB":
            return nums[0] - sum(nums[1:]) if len(nums) > 1 else nums[0]
        elif mode == "DIV":
            try:
                result = nums[0]
                for n in nums[1:]:
                    result /= n
                return result
            except ZeroDivisionError:
                return 1.0
        return 1.0

    def _get_tokenized_with_special_token_injection(self, slices) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare tokenized input with special token injection in training format:
        Every slice_length tokens: [SEP] [MASK] special_token [MASK] content...
        """
        special_token = self._testing_special_token
        mask_token = "[MASK]"

        all_tokens = []
        alpha_weights = []

        # Process each slice
        for text, meta in slices:
            # Clean up the text
            logic_free = text
            for tok in self.logic_tokens:
                logic_free = logic_free.replace(tok, "")

            # Tokenize this slice
            slice_tokens = self.tokenizer.tokenize(logic_free)

            # For every slice_length-token segment, inject pattern
            content_per_segment = self.slice_length - 4  # 4 tokens for [SEP] [MASK] special [MASK]

            for i in range(0, len(slice_tokens), content_per_segment):
                # Add [SEP] [MASK] special_token [MASK]
                all_tokens.extend([self.sep_token, mask_token, special_token, mask_token])
                alpha_weights.extend([1.0, 0.0, 1.0, 0.0])  # Weights for injected tokens

                # Add up to content_per_segment content tokens
                chunk_end = min(i + content_per_segment, len(slice_tokens))
                chunk = slice_tokens[i:chunk_end]
                all_tokens.extend(chunk)

                # Add weights for content tokens
                weight = float(meta.get("weight", 1.0))
                alpha_weights.extend([weight] * len(chunk))

                # Pad this slice_length-token segment if needed
                segment_length = 4 + len(chunk)
                if segment_length < self.slice_length:
                    pad_count = self.slice_length - segment_length
                    all_tokens.extend([self.padding_token] * pad_count)
                    alpha_weights.extend([0.0] * pad_count)

        # Convert to string and tokenize properly
        full_text = self.tokenizer.convert_tokens_to_string(all_tokens)
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
            padding=False,
            add_special_tokens=False
        )

        input_ids = encoded["input_ids"]
        input_len = input_ids.shape[-1]

        # Ensure alpha weights match token length
        if len(alpha_weights) < input_len:
            alpha_weights.extend([1.0] * (input_len - len(alpha_weights)))
        elif len(alpha_weights) > input_len:
            alpha_weights = alpha_weights[:input_len]

        # Pad to max_length
        pad_len = self.max_length - input_len

        if pad_len > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full((1, pad_len), self.tokenizer.pad_token_id, dtype=torch.long, device=input_ids.device)
            ], dim=-1)
            alpha_weights += [0.0] * pad_len
        else:
            input_ids = input_ids[:, :self.max_length]
            alpha_weights = alpha_weights[:self.max_length]

        alpha_mask = torch.tensor([alpha_weights], dtype=torch.float32, device=input_ids.device)

        return input_ids, alpha_mask

    def get_tokenized_tensor_with_alpha(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (input_ids, alpha_mask) as [1, max_length].
        Alpha mask is derived from slice weight metadata.
        """
        slices = self.slice_prompt()

        # For training format: every 77 tokens should start with [SEP] [MASK] special_token [MASK]
        if hasattr(self, '_testing_special_token'):
            return self._get_tokenized_with_special_token_injection(slices)

        prompt_chunks = []
        weight_chunks = []

        for text, meta in slices:
            throw_exception_if_processing_interrupted()
            logic_free = text
            for tok in self.logic_tokens:
                logic_free = logic_free.replace(tok, "")
            prompt_chunks.append(logic_free.strip())
            weight_chunks.append(float(meta.get("weight", 1.0)))

        # Join text
        full_prompt = f" {self.sep_token} ".join(prompt_chunks)
        # Add [CLS] at the beginning
        full_prompt = f"[CLS] {full_prompt}"

        # Tokenize prompt first to get the actual token count
        encoded = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=False,
            padding=False,
            add_special_tokens=False
        )

        input_ids = encoded["input_ids"]
        input_len = input_ids.shape[-1]

        # Create alpha mask that matches the tokenized length
        # Distribute weights proportionally across tokens for each chunk
        tokens_per_chunk = []
        for chunk in prompt_chunks:
            chunk_encoded = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=False,
                padding=False,
                add_special_tokens=False
            )
            tokens_per_chunk.append(chunk_encoded["input_ids"].shape[-1])

        # Account for separator tokens between chunks
        sep_encoded = self.tokenizer(
            f" {self.sep_token} ",
            return_tensors="pt",
            truncation=False,
            padding=False,
            add_special_tokens=False
        )
        sep_token_count = sep_encoded["input_ids"].shape[-1]

        # Build alpha vector with correct length
        alpha_vector = []
        for i, (token_count, weight) in enumerate(zip(tokens_per_chunk, weight_chunks)):
            alpha_vector.extend([weight] * token_count)
            if i < len(tokens_per_chunk) - 1:  # Add separator weight except for last chunk
                alpha_vector.extend([1.0] * sep_token_count)

        # Ensure alpha vector matches input length (handle any rounding errors)
        if len(alpha_vector) < input_len:
            alpha_vector.extend([1.0] * (input_len - len(alpha_vector)))
        elif len(alpha_vector) > input_len:
            alpha_vector = alpha_vector[:input_len]

        # Pad both token and alpha to max_length
        pad_len = self.max_length - input_len

        if pad_len > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full((1, pad_len), self.tokenizer.pad_token_id, dtype=torch.long, device=input_ids.device)
            ], dim=-1)
            alpha_vector += [0.0] * pad_len  # suppress attention on pad
        else:
            input_ids = input_ids[:, :self.max_length]
            alpha_vector = alpha_vector[:self.max_length]

        alpha_mask = torch.tensor([alpha_vector], dtype=torch.float32, device=input_ids.device)

        # Ensure both have the same length
        assert input_ids.shape[-1] == alpha_mask.shape[
            -1], f"Shape mismatch: input_ids {input_ids.shape} vs alpha_mask {alpha_mask.shape}"

        return input_ids, alpha_mask

    def _handle_each_mode(self, slices, each_content, each_mode):
        """
        Apply [EACH] content to all subsequent slices based on mode.
        each_content: list of tokens to repeat
        each_mode: "rigid", "push", "end", or "inject"
        """
        if not each_content:
            return slices

        modified_slices = []

        for i, (slice_text, metadata) in enumerate(slices):
            if i == 0:
                # First slice stays unchanged
                modified_slices.append((slice_text, metadata))
                continue

            # Tokenize current slice
            slice_tokens = self.tokenizer.tokenize(slice_text)

            if each_mode == "rigid":
                # Destructive: place at exact positions
                # Find original position of each_content in first slice
                first_slice_tokens = self.tokenizer.tokenize(slices[0][0])

                # Find where each_content starts in first slice
                start_pos = None
                for j in range(len(first_slice_tokens) - len(each_content) + 1):
                    if first_slice_tokens[j:j + len(each_content)] == each_content:
                        start_pos = j
                        break

                if start_pos is not None:
                    # Overwrite at same position
                    for j, token in enumerate(each_content):
                        if start_pos + j < len(slice_tokens):
                            slice_tokens[start_pos + j] = token

            elif each_mode == "push":
                # Non-destructive: insert after first 2 tokens
                insert_pos = min(2, len(slice_tokens))
                slice_tokens = slice_tokens[:insert_pos] + each_content + slice_tokens[insert_pos:]
                # Truncate if needed
                slice_tokens = slice_tokens[:self.slice_length]

            elif each_mode == "end":
                # Append to end if space
                remaining_space = self.slice_length - len(slice_tokens)
                if remaining_space > 0:
                    can_add = min(len(each_content), remaining_space)
                    slice_tokens.extend(each_content[:can_add])

            elif each_mode == "inject":
                # Insert at original position and shift
                first_slice_tokens = self.tokenizer.tokenize(slices[0][0])

                # Find where each_content starts
                start_pos = None
                for j in range(len(first_slice_tokens) - len(each_content) + 1):
                    if first_slice_tokens[j:j + len(each_content)] == each_content:
                        start_pos = j
                        break

                if start_pos is not None and start_pos < len(slice_tokens):
                    # Insert and shift
                    slice_tokens = slice_tokens[:start_pos] + each_content + slice_tokens[start_pos:]
                    # Truncate to slice_length
                    slice_tokens = slice_tokens[:self.slice_length]

            # Pad if needed
            if len(slice_tokens) < self.slice_length:
                slice_tokens.extend([self.padding_token] * (self.slice_length - len(slice_tokens)))

            # Convert back to text
            modified_text = self.tokenizer.convert_tokens_to_string(slice_tokens)
            modified_slices.append((modified_text, metadata))

        return modified_slices

    def _extract_each_content(self, segment, each_position):
        """
        Extract content between last comma and next comma around [EACH] position.
        """
        # Find comma positions
        last_comma = segment.rfind(',', 0, each_position)
        next_comma = segment.find(',', each_position)

        # Extract content
        if last_comma == -1:
            start = 0
        else:
            start = last_comma + 1

        if next_comma == -1:
            end = len(segment)
        else:
            end = next_comma

        each_text = segment[start:end].strip()
        return self.tokenizer.tokenize(each_text)

    @torch.no_grad()
    def extract_alpha_similarities(
            self,
            embedding_manager,
            top_k: int = 5,
            use_delta: bool = True,
            use_pooled: bool = True,
            projection_dim: int = 768
    ) -> List[Dict[str, Any]]:
        """
        Classify segments leveraging RoPE's relative position encoding.
        Tests multiple positions to find optimal token-content relationships.
        """
        # Get base prompt slices
        slices = self.slice_prompt()
        logger.info(f"[SymbolicLogicManager] Sliced prompt into {len(slices)} segments.")
        # Build the base prompt with [SEP] tokens
        prompt_chunks = []
        for text, meta in slices:
            logic_free = text
            for tok in self.logic_tokens:
                logic_free = logic_free.replace(tok, "")
            prompt_chunks.append(logic_free.strip())

        base_prompt = f" {self.sep_token} ".join(prompt_chunks)

        # Tokenize base prompt
        base_encoded = self.tokenizer(
            base_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            add_special_tokens=False
        )
        base_input_ids = base_encoded["input_ids"].to(self.model.device)
        base_attention_mask = base_encoded["attention_mask"].to(self.model.device)

        # Get base outputs
        base_outputs = self.model(base_input_ids, attention_mask=base_attention_mask)
        base_pooled = base_outputs.pooler_output
        base_last_hidden = base_outputs.last_hidden_state

        if base_pooled is None:
            base_pooled = base_last_hidden[:, 0]

        if base_pooled.dim() == 3:
            base_pooled = base_pooled.squeeze(1)

        logger.info(f"[SymbolicLogicManager] Base pooled shape: {base_pooled.shape}")

        candidates = []

        # Process each special token
        for special_token in BEATRIX_SPECIAL_TOKENS_AND_SHUNTS:
            throw_exception_if_processing_interrupted()

            is_shunt = special_token.startswith("[SHUNT_")

            # With RoPE, test multiple relative positions
            position_scores = []

            # Test different insertion strategies
            test_positions = [
                # (position_type, prompt_construction)
                ("prefix", f"{special_token} {base_prompt}"),  # Standard prefix
                ("sep_inject", base_prompt.replace(self.sep_token, f"{self.sep_token} {special_token}")),
                # After each SEP
                ("midpoint", self._insert_at_midpoint(special_token, base_prompt)),  # At prompt midpoint
            ]

            # For SHUNT tokens, also test with padding strategies
            if is_shunt:
                test_positions.extend([
                    ("shunt_spaced", f"{special_token} {self.padding_token} {base_prompt}"),  # With spacing
                    ("shunt_end", f"{base_prompt} {self.sep_token} {special_token}"),  # At end
                ])

            best_score = -float('inf')
            best_position = None
            best_outputs = None

            for pos_type, modified_prompt in test_positions:
                # Tokenize the modified prompt
                encoded = self.tokenizer(
                    modified_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    add_special_tokens=False
                )

                input_ids = encoded["input_ids"].to(self.model.device)
                attention_mask = encoded["attention_mask"].to(self.model.device)

                # Run through model
                outputs = self.model(input_ids, attention_mask=attention_mask)

                # Get representations
                modified_pooled = outputs.pooler_output
                hidden_states = outputs.last_hidden_state

                if modified_pooled is None:
                    modified_pooled = hidden_states[:, 0]

                if modified_pooled.dim() == 3:
                    modified_pooled = modified_pooled.squeeze(1)

                # Calculate position-specific score
                # RoPE makes tokens maintain relationships regardless of absolute position
                pos_score = self._calculate_rope_aware_score(
                    hidden_states,
                    attention_mask,
                    special_token,
                    base_pooled,
                    modified_pooled
                )

                position_scores.append({
                    'type': pos_type,
                    'score': pos_score,
                    'prompt': modified_prompt
                })

                if pos_score > best_score:
                    best_score = pos_score
                    best_position = pos_type
                    best_outputs = (outputs, modified_pooled, hidden_states, attention_mask)

            # Use best position for final scoring
            outputs, modified_pooled, hidden_states, attention_mask = best_outputs

            # Compute detailed scores
            scores = {}

            # 1. Best position score
            scores['best_position'] = best_position
            scores['position_score'] = best_score

            # 2. Pooled similarity
            sim = torch.cosine_similarity(modified_pooled, base_pooled, dim=-1)
            scores['pooled_similarity'] = sim.mean().item() if sim.numel() > 1 else sim.item()

            # 3. Attention pattern analysis (RoPE-aware)
            # Find where the special token appears most frequently
            token_ids = self.tokenizer.convert_tokens_to_ids(special_token)
            special_positions = (input_ids[0] == token_ids).nonzero(as_tuple=True)[0]

            if len(special_positions) > 0:
                # Analyze attention patterns around special tokens
                # With RoPE, tokens can attend effectively across distances
                avg_attention_spread = self._analyze_attention_spread(
                    hidden_states, attention_mask, special_positions
                )
                scores['attention_spread'] = avg_attention_spread
            else:
                scores['attention_spread'] = 0.0

            # 4. Contextual coherence across different distances
            scores['rope_coherence'] = self._measure_rope_coherence(
                hidden_states, attention_mask, special_token
            )

            # Combined score emphasizing RoPE benefits
            combined_score = (
                    scores['position_score'] * 0.3 +
                    scores['pooled_similarity'] * 0.3 +
                    scores['rope_coherence'] * 0.3 +
                    scores['attention_spread'] * 0.1
            )

            candidates.append({
                "md5": f"{special_token}_{best_position}",
                "score": combined_score,
                "trigger": special_token,
                "best_position": best_position,
                "position_scores": position_scores,
                "is_shunt": is_shunt,
                "detailed_scores": scores,
                "pooled": modified_pooled.squeeze(0).cpu().tolist() if use_pooled else None
            })

            if self.pbar:
                self.pbar.update(1)

        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    def _insert_at_midpoint(self, token, prompt):
        """Insert token at the midpoint of the prompt."""
        words = prompt.split()
        midpoint = len(words) // 2
        words.insert(midpoint, token)
        return " ".join(words)

    def _calculate_rope_aware_score(self, hidden_states, attention_mask, special_token, base_pooled, modified_pooled):
        """
        Calculate score that leverages RoPE's relative position properties.
        """
        # Basic similarity
        base_sim = torch.cosine_similarity(modified_pooled, base_pooled, dim=-1)
        base_score = base_sim.mean().item() if base_sim.numel() > 1 else base_sim.item()

        # Measure how well the special token integrates across different distances
        # With RoPE, a well-integrated token should maintain coherence regardless of distance
        token_id = self.tokenizer.convert_tokens_to_ids(special_token)
        token_positions = (attention_mask[0] == token_id).nonzero(as_tuple=True)[0]

        if len(token_positions) == 0:
            return base_score

        # Sample representations at different relative distances from special token
        distance_coherences = []
        for pos in token_positions:
            for distance in [1, 5, 10, 20, 50]:  # Various distances
                target_pos = pos + distance
                if target_pos < hidden_states.shape[1] and attention_mask[0, target_pos]:
                    token_repr = hidden_states[0, pos]
                    distant_repr = hidden_states[0, target_pos]
                    coherence = torch.cosine_similarity(
                        token_repr.unsqueeze(0),
                        distant_repr.unsqueeze(0),
                        dim=-1
                    ).item()
                    distance_coherences.append(coherence)

        if distance_coherences:
            # Higher variance means less consistent influence across distances
            coherence_std = torch.tensor(distance_coherences).std().item()
            distance_score = 1.0 / (1.0 + coherence_std)  # Lower std is better
        else:
            distance_score = 0.5

        return base_score * 0.7 + distance_score * 0.3

    def _analyze_attention_spread(self, hidden_states, attention_mask, special_positions):
        """
        Analyze how broadly the special token influences the sequence.
        With RoPE, influence should be position-agnostic.
        """
        if len(special_positions) == 0:
            return 0.0

        # Measure variance in representation changes around special tokens
        influences = []
        for pos in special_positions:
            # Get representations in a window around the special token
            window_size = 10
            start = max(0, pos - window_size)
            end = min(hidden_states.shape[1], pos + window_size + 1)

            if end - start > 1:
                window_reprs = hidden_states[0, start:end]
                # Measure how much representations vary in this window
                variance = window_reprs.var(dim=0).mean().item()
                influences.append(variance)

        return sum(influences) / len(influences) if influences else 0.0

    def _measure_rope_coherence(self, hidden_states, attention_mask, special_token):
        """
        Measure how coherently the special token integrates with content
        across the full sequence, leveraging RoPE's distance-agnostic properties.
        """
        # Get mean representation
        mask_expanded = attention_mask.unsqueeze(-1)
        mean_repr = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

        # Measure how each position relates to the mean
        # With good RoPE utilization, this should be consistent
        similarities = []
        for i in range(hidden_states.shape[1]):
            if attention_mask[0, i]:
                sim = torch.cosine_similarity(
                    hidden_states[0, i].unsqueeze(0),
                    mean_repr,
                    dim=-1
                ).item()
                similarities.append(sim)

        # Higher mean with lower variance indicates better integration
        if similarities:
            mean_sim = sum(similarities) / len(similarities)
            std_sim = torch.tensor(similarities).std().item()
            return mean_sim * (1.0 / (1.0 + std_sim))

        return 0.0

    @torch.no_grad()
    def generate_symbolic_fingerprint(
            self,
            embedding_manager,
            projection_dim: int = 768,
            use_delta: bool = True,
            use_pooled: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Generate a fingerprint showing which special tokens best match the prompt content.
        SHUNT tokens go in position 2, symbolic tokens in position 1.
        """
        # Get the base tokenized prompt with alpha weighting
        input_ids, alpha_mask = self.get_tokenized_tensor_with_alpha()

        # Move to device
        input_ids = input_ids.to(self.model.device)
        alpha_mask = alpha_mask.to(self.model.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Get base prompt outputs
        base_outputs = self.model(input_ids, attention_mask=attention_mask)
        base_hidden = base_outputs.last_hidden_state
        base_pooled = base_outputs.pooler_output

        # Apply alpha mask for weighted representation
        base_weighted = (base_hidden * alpha_mask.unsqueeze(-1)).sum(dim=1)
        base_weighted = base_weighted / alpha_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Use pooler output if available, otherwise use weighted representation
        if base_pooled is None:
            base_pooled = base_weighted

        # Ensure proper shape
        if base_pooled.dim() == 3:
            base_pooled = base_pooled.squeeze(1)

        results = []

        for special_token in BEATRIX_SPECIAL_TOKENS_AND_SHUNTS:
            throw_exception_if_processing_interrupted()

            # Determine if this is a SHUNT token or symbolic token
            is_shunt = special_token.startswith("[SHUNT_")

            # Tokenize special token and padding token
            special_token_ids = self.tokenizer.encode(special_token, add_special_tokens=False)
            pad_token_id = self.tokenizer.pad_token_id

            # Create token list based on type
            if is_shunt:
                # SHUNT: [PAD] [SHUNT] rest_of_prompt
                token_list = [pad_token_id] + special_token_ids + input_ids[0].tolist()
                special_token_position = len(special_token_ids)  # Position after padding
            else:
                # Symbolic: <token> [PAD] rest_of_prompt
                token_list = special_token_ids + [pad_token_id] + input_ids[0].tolist()
                special_token_position = 0  # First position

            # Handle length constraints
            if len(token_list) > self.max_length:
                token_list = token_list[:self.max_length]

            modified_ids = torch.tensor([token_list], dtype=torch.long, device=input_ids.device)

            # Pad if necessary
            if modified_ids.shape[1] < self.max_length:
                pad_len = self.max_length - modified_ids.shape[1]
                modified_ids = torch.nn.functional.pad(
                    modified_ids,
                    (0, pad_len),
                    value=self.tokenizer.pad_token_id
                )

            # Create attention mask and run through model
            attention_mask_modified = (modified_ids != self.tokenizer.pad_token_id).long()

            modified_outputs = self.model(modified_ids, attention_mask=attention_mask_modified)
            modified_hidden = modified_outputs.last_hidden_state
            modified_pooled = modified_outputs.pooler_output

            # Get special token representation at its specific position
            special_token_repr = modified_hidden[:, special_token_position, :]

            # Use pooler output if available
            if modified_pooled is None:
                # Fallback to mean pooling
                modified_pooled = (modified_hidden * attention_mask_modified.unsqueeze(-1)).sum(dim=1)
                modified_pooled = modified_pooled / attention_mask_modified.sum(dim=1, keepdim=True).clamp(min=1)

            # Ensure proper shape
            if modified_pooled.dim() == 3:
                modified_pooled = modified_pooled.squeeze(1)

            # Calculate similarity score
            sim = torch.cosine_similarity(modified_pooled, base_pooled, dim=-1)
            similarity = sim.mean().item() if sim.numel() > 1 else sim.item()

            # Token coherence with content (excluding padding and special token)
            content_mask = attention_mask_modified.clone()
            content_mask[0, special_token_position] = 0  # Exclude special token
            if is_shunt:
                content_mask[0, 0] = 0  # Exclude padding at position 0
            else:
                content_mask[0, len(special_token_ids)] = 0  # Exclude padding after special token

            if content_mask.sum() > 0:
                content_repr = (modified_hidden * content_mask.unsqueeze(-1)).sum(dim=1)
                content_repr = content_repr / content_mask.sum(dim=1, keepdim=True)
                sim = torch.cosine_similarity(special_token_repr, content_repr, dim=-1)
                coherence = sim.mean().item() if sim.numel() > 1 else sim.item()
            else:
                coherence = 0.0

            # Combined score
            final_score = similarity * 0.6 + coherence * 0.4

            results.append((special_token, final_score))

        return sorted(results, key=lambda x: x[1], reverse=True)