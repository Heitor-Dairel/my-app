def rgb_to_bgr(r: int, g: int, b: int) -> tuple[int, int, int]:
    r"""
    Convert an RGB color to BGR format.

    Args:
        r (int): Red component (0-255).
        g (int): Green component (0-255).
        b (int): Blue component (0-255).

    Returns:
        return (tuple[int, int, int]): The color represented in BGR format as (B, G, R).

    Notes:
        - Useful for OpenCV, which expects colors in BGR order instead of RGB.
    """
    return (b, g, r)
