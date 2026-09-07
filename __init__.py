"""R4 ComfyUI package. Legacy R3 nodes are intentionally not registered."""
from .comfy_offline_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .runtime_manager import start_background_update

WEB_DIRECTORY = "./web"
start_background_update()
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
