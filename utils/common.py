def snake_to_camel(snake_str):
    """Convert snake_case to CamelCase (PascalCase)."""
    components = snake_str.split("_")
    return "".join(x.capitalize() for x in components)


def print_colored(message, color=None):
    """Print a message in the specified color."""
    # ANSI escape sequences for colors
    colors = {
        "gray": "\033[90m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }

    # Commonly used levels that map to specific colors
    level_colors = {
        "info": colors["blue"],
        "warn": colors["yellow"],
        "error": colors["red"],
        "success": colors["green"],
    }

    # If the color or level key is recognized, apply the color, else use no color (reset)
    color_code = colors.get(color) or level_colors.get(color, colors["reset"])  # type: ignore

    # Print the message with the corresponding color
    print(f"{color_code}{message}{colors['reset']}")
