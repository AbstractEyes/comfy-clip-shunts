class AlucardShapeError(Exception):
    """
    Raised when Alucard receives tensors whose shapes make the
    formulaic contract impossible to satisfy.

    Example message:
        Shape mismatch @ FieldWalker.walk
          a : torch.Size([1,2048,768])
          b : torch.Size([1,  77,768])
        Required: identical [B,T,D] after optional projection/alignment.
        Hint   : set `override_context_window=True`
                 or check `sliding_window_size` / `force_projection_in`.
    """
    pass

# ------------------------------------------------------------------ #
def validate_shapes(a, b):
    if a.shape != b.shape:
        raise AlucardShapeError(
            f"Shape mismatch @ FieldWalker.walk\n"
            f"  a : {tuple(a.shape)}\n"
            f"  b : {tuple(b.shape)}\n"
            "Required: identical [B,T,D] after optional projection/alignment."
        )
# ------------------------------------------------------------------ #