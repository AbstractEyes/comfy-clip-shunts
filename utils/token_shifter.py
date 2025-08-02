import torch

class TokenShifter:


    @staticmethod
    def token_similarity_score(tokenizer,
                               prompt1: str,
                               prompt2: str,
                               device=None) -> torch.Tensor:
        ...


    @staticmethod
    def slice(tokenizer,
              prompt: str,
              embedding_layer = None,
              top_k: int = 25,
              bottom_k: int = 25,
              max_len: int = 77,
              device=None) -> torch.Tensor:
        """
        Returns reduced token tensor shaped [max_len, D] based on embedding norms.
        """
        # Tokenize and embed full prompt
        token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device)
        with torch.no_grad():
            embeddings = embedding_layer(token_ids)  # [T, D]

        # Convert to string tokens
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        if embeddings.shape[0] <= max_len:
            return embeddings  # Already short enough

        # Compute norm and select top/bottom
        norms = embeddings.norm(dim=1)
        sorted_idx = torch.argsort(norms, descending=True)
        top_idx = sorted_idx[:top_k]
        bottom_idx = sorted_idx[-bottom_k:]

        selected_idx = torch.unique(torch.cat([top_idx, bottom_idx]), sorted=False).tolist()
        selected_idx = sorted(selected_idx)

        # Fill in remaining space
        if len(selected_idx) < max_len:
            middle_idx = [i for i in range(len(tokens)) if i not in selected_idx]
            for idx in middle_idx:
                selected_idx.append(idx)
                if len(selected_idx) >= max_len:
                    break
        else:
            selected_idx = selected_idx[:max_len]

        return embeddings[selected_idx]  # Final tensor: [≤77, D]
