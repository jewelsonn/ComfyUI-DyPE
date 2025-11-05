import torch
from comfy_api.latest import ComfyExtension, io
from .src.patch import apply_dype_to_flux

# Try to import ComfyFluxWrapper with multiple possible paths
ComfyFluxWrapper = None
NUNCHAKU_FLUX_AVAILABLE = False

try:
    # Try relative import first (if running from Nunchaku directory)
    from wrappers.flux import ComfyFluxWrapper
    NUNCHAKU_FLUX_AVAILABLE = True
except ImportError:
    try:
        # Try absolute import from parent directory (if DyPE is in custom_nodes)
        import sys
        import os
        # Add parent directory to path to find wrappers
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from wrappers.flux import ComfyFluxWrapper
        NUNCHAKU_FLUX_AVAILABLE = True
    except ImportError:
        # Fallback: use string-based detection
        try:
            import sys
            import os
            nunchaku_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if nunchaku_dir not in sys.path:
                sys.path.insert(0, nunchaku_dir)
            from wrappers.flux import ComfyFluxWrapper
            NUNCHAKU_FLUX_AVAILABLE = True
        except ImportError:
            # Keep ComfyFluxWrapper as None, we'll use string-based detection
            pass


def is_nunchaku_flux_model(model):
    """Check if the model is a Nunchaku FLUX model."""
    if not NUNCHAKU_FLUX_AVAILABLE:
        return False
    
    # Check if the diffusion model is a ComfyFluxWrapper
    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        return isinstance(model.model.diffusion_model, ComfyFluxWrapper)
    return False


def get_model_type(model):
    """Get the type of model for compatibility checks."""
    if is_nunchaku_flux_model(model):
        return "nunchaku_flux"
    else:
        return "standard_flux"

class DyPE_FLUX(io.ComfyNode):
    """
    Applies DyPE (Dynamic Position Extrapolation) to a FLUX model.
    This allows generating images at resolutions far beyond the model's training scale
    by dynamically adjusting positional encodings and the noise schedule.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DyPE_FLUX",
            display_name="DyPE for FLUX",
            category="model_patches/unet",
            description="Applies DyPE (Dynamic Position Extrapolation) to a FLUX model for ultra-high-resolution generation.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The FLUX model to patch with DyPE.",
                ),
                io.Int.Input(
                    "width",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image width. Must match the width of your empty latent."
                ),
                io.Int.Input(
                    "height",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image height. Must match the height of your empty latent."
                ),
                io.Combo.Input(
                    "method",
                    options=["yarn", "ntk", "base"],
                    default="yarn",
                    tooltip="Position encoding extrapolation method (YARN recommended).",
                ),
                io.Boolean.Input(
                    "enable_dype",
                    default=True,
                    label_on="Enabled",
                    label_off="Disabled",
                    tooltip="Enable or disable Dynamic Position Extrapolation for RoPE.",
                ),
                io.Float.Input(
                    "dype_exponent",
                    default=2.0, min=0.0, max=4.0, step=0.1,
                    optional=True,
                    tooltip="Controls DyPE strength over time (λt). 2.0=Exponential (best for 4K+), 1.0=Linear, 0.5=Sub-linear (better for ~2K)."
                ),
                io.Float.Input(
                    "base_shift",
                    default=0.5, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Base shift for the noise schedule (mu). Default is 0.5."
                ),
                io.Float.Input(
                    "max_shift",
                    default=1.15, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Max shift for the noise schedule (mu) at high resolutions. Default is 1.15."
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="The FLUX model patched with DyPE.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, width: int, height: int, method: str, enable_dype: bool, dype_exponent: float = 2.0, base_shift: float = 0.5, max_shift: float = 1.15) -> io.NodeOutput:
        """
        Clones the model and applies the DyPE patch for both the noise schedule and positional embeddings.
        Supports both standard FLUX models and Nunchaku FLUX models.
        """
        # Automatic model detection - no need to check compatibility manually
        # The apply_dype_to_flux function will handle detection and error cases
        
        patched_model = apply_dype_to_flux(model, width, height, method, enable_dype, dype_exponent, base_shift, max_shift)
        return io.NodeOutput(patched_model)

class DyPEExtension(ComfyExtension):
    """Registers the DyPE node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DyPE_FLUX]

async def comfy_entrypoint() -> DyPEExtension:
    return DyPEExtension()