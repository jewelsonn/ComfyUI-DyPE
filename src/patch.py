import torch
import torch.nn as nn
import math
import types
from comfy.model_patcher import ModelPatcher
from comfy import model_sampling
from .rope import get_1d_rotary_pos_embed

# Global debug flag for controlling verbose logging
DYPE_PATCH_DEBUG = False

def _debug_print(*args, **kwargs):
    """Conditionally print debug messages based on the global debug flag."""
    if DYPE_PATCH_DEBUG:
        print(*args, **kwargs)

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


class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', dype: bool = True, dype_exponent: float = 2.0): # Add dype_exponent
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method
        self.dype = dype if method != 'base' else False
        self.dype_exponent = dype_exponent
        self.current_timestep = 1.0
        self.base_resolution = 1024
        self.base_patches = (self.base_resolution // 8) // 2

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb_parts = []
        pos = ids.float()
        freqs_dtype = torch.bfloat16

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'repeat_interleave_real': True, 'use_real': True, 'freqs_dtype': freqs_dtype}
            
            # Pass the exponent to the RoPE function
            dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_exponent': self.dype_exponent}

            if i > 0:
                max_pos = axis_pos.max().item()
                current_patches = int(max_pos + 1)

                if self.method == 'yarn' and current_patches > self.base_patches:
                    max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, yarn=True, max_pe_len=max_pe_len, ori_max_pe_len=self.base_patches, **dype_kwargs)
                elif self.method == 'ntk' and current_patches > self.base_patches:
                    base_ntk_scale = (current_patches / self.base_patches)
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, ntk_factor=base_ntk_scale, **dype_kwargs)
                else:
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs)
            else:
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs)

            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)

def is_nunchaku_flux_model(model):
    """Check if the model is a Nunchaku FLUX model using multiple methods."""
    if not hasattr(model, 'model') or not hasattr(model.model, 'diffusion_model'):
        return False
    
    diffusion_model = model.model.diffusion_model
    
    # Method 1: Check class type if available
    if NUNCHAKU_FLUX_AVAILABLE and ComfyFluxWrapper is not None:
        if isinstance(diffusion_model, ComfyFluxWrapper):
            return True
    
    # Method 2: Check class name as string (fallback)
    if hasattr(diffusion_model, '__class__'):
        class_name = diffusion_model.__class__.__name__
        if class_name == "ComfyFluxWrapper":
            return True
    
    # Method 3: Check for specific attributes that indicate Nunchaku FLUX
    if (hasattr(diffusion_model, 'model') and 
        hasattr(diffusion_model, 'config') and
        hasattr(diffusion_model.model, 'pe_embedder')):
        # This pattern matches the Nunchaku FLUX structure
        return True
    
    return False

def create_nunchaku_forward_wrapper(wrapper_instance, pe_embedder, enable_dype=True, debug=False):
    """
    Create a wrapper for ComfyFluxWrapper.forward that applies DyPE to generated img_ids.
    
    This is the CORRECT approach for Nunchaku models.
    
    Args:
        wrapper_instance: The wrapper instance
        pe_embedder: The DyPE positional embedder
        enable_dype: Whether DyPE is enabled
        debug: Enable verbose debug logging (default: False for silent operation)
    """
    import torch
    from comfy.ldm.common_dit import pad_to_patch_size
    from einops import rearrange, repeat
    
    # Update global debug flag for this wrapper instance
    global DYPE_PATCH_DEBUG
    original_debug_state = DYPE_PATCH_DEBUG
    DYPE_PATCH_DEBUG = debug
    
    # Save the original forward method
    original_forward = wrapper_instance.forward
    
    def dype_enhanced_forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        """Enhanced forward method that applies DyPE to img_ids before calling the model."""
        try:
            _debug_print(f"🎯 DyPE: Enhanced Nunchaku forward called for timestep {timestep}")
            
            # Handle timestep normalization for DyPE
            if enable_dype and timestep is not None:
                sigma = None
                if hasattr(timestep, 'item'):
                    sigma = timestep.item()
                elif isinstance(timestep, (int, float)):
                    sigma = float(timestep)
                elif hasattr(timestep, '__len__') and len(timestep) > 0:
                    if hasattr(timestep[0], 'item'):
                        sigma = timestep[0].item()
                    else:
                        sigma = float(timestep[0])
                
                if sigma is not None:
                    sigma_max = 1.0  # Fallback
                    try:
                        if hasattr(self.model, 'model_sampling'):
                            sigma_max = self.model.model_sampling.sigma_max.item()
                    except:
                        pass
                    
                    if sigma_max > 0:
                        normalized_timestep = min(max(sigma / sigma_max, 0.0), 1.0)
                        pe_embedder.set_timestep(normalized_timestep)
                        _debug_print(f"🎯 DyPE: Set normalized timestep to {normalized_timestep:.3f}")
            
            # Generate img_ids using the SAME logic as ComfyFluxWrapper.process_img
            bs, c, h_orig, w_orig = x.shape
            patch_size = self.config.get("patch_size", 2)
            h_len = (h_orig + (patch_size // 2)) // patch_size
            w_len = (w_orig + (patch_size // 2)) // patch_size
            
            # Process image exactly like ComfyFluxWrapper does
            x_padded = pad_to_patch_size(x, (patch_size, patch_size))
            img = rearrange(x_padded, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            
            # Generate img_ids with the same logic
            h_offset = 0
            w_offset = 0
            index = 0
            
            h_offset = (h_offset + (patch_size // 2)) // patch_size
            w_offset = (w_offset + (patch_size // 2)) // patch_size
            
            img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
            img_ids[:, :, 0] = img_ids[:, :, 1] + index
            img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
                h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype
            ).unsqueeze(1)
            img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
                w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype
            ).unsqueeze(1)
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
            
            _debug_print(f"🎯 DyPE: Generated img_ids with shape {img_ids.shape}, max_pos={img_ids.max().item():.1f}")
            
            # Apply DyPE enhancement to img_ids
            if enable_dype:
                try:
                    _debug_print(f"🎯 DyPE: Applying DyPE enhancement to img_ids")
                    
                    # Get DyPE enhanced embeddings
                    dype_embeddings = pe_embedder(img_ids)
                    
                    # Apply frequency scaling based on DyPE timestep and embeddings
                    freq_scale = 1.0
                    if hasattr(pe_embedder, 'current_timestep'):
                        timestep_norm = pe_embedder.current_timestep
                        base_freq = 1.0 + 0.05 * timestep_norm  # Conservative scaling
                        
                        # Scale based on the actual DyPE embedding variance
                        dype_variance = dype_embeddings.var().item()
                        img_variance = img_ids.var().item()
                        if img_variance > 0:
                            freq_scale = base_freq * (1.0 + 0.1 * timestep_norm * dype_variance / (img_variance + 1e-8))
                    
                    # Apply frequency scaling to Y and X spatial coordinates only
                    if img_ids.shape[-1] >= 3:
                        enhanced_img_ids = img_ids.clone()
                        enhanced_img_ids[..., 1:3] = enhanced_img_ids[..., 1:3] * freq_scale
                        
                        _debug_print(f"🎯 DyPE: Applied frequency scaling (scale={freq_scale:.3f}) to spatial coordinates")
                        _debug_print(f"🎯 DyPE: Enhanced img_ids max_pos={enhanced_img_ids.max().item():.1f}")
                        
                        # Use the enhanced img_ids
                        img_ids = enhanced_img_ids
                        
                except Exception as e:
                    # Keep error logging but make it conditional on debug flag
                    if debug:
                        print(f"⚠️ DyPE: Error applying enhancement: {e}, using original img_ids")
            
            # Create a custom forward function that uses our enhanced img_ids
            def enhanced_model_call(**model_kwargs):
                # Inject our enhanced img_ids
                model_kwargs['img_ids'] = img_ids
                # Remove any img_ids that might be passed in kwargs to avoid conflicts
                model_kwargs.pop('img_ids_dype', None)
                return self.model(**model_kwargs)
            
            # Temporarily replace the model's forward method
            original_model_forward = self.model.forward
            self.model.forward = enhanced_model_call
            
            try:
                # Call the original forward with our modifications
                result = original_forward(x, timestep, context, y, guidance, control, transformer_options, **kwargs)
                return result
            finally:
                # Restore the original model forward
                self.model.forward = original_model_forward
            
        except Exception as e:
            # Keep critical error logging but make it conditional on debug flag
            if debug:
                print(f"❌ DyPE: Critical error in enhanced forward: {e}")
                print("🔄 Falling back to original forward")
            return original_forward(x, timestep, context, y, guidance, control, transformer_options, **kwargs)
        finally:
            # Restore original debug state
            DYPE_PATCH_DEBUG = original_debug_state
    
    return dype_enhanced_forward

def apply_dype_to_flux(model: ModelPatcher, width: int, height: int, method: str, enable_dype: bool, dype_exponent: float, base_shift: float, max_shift: float, model_type: str = None) -> ModelPatcher:
    """Apply DyPE to FLUX model with automatic Nunchaku FLUX detection.
    
    Args:
        model: The model patcher instance (standard FLUX or Nunchaku FLUX)
        width: Target image width
        height: Target image height
        method: Position encoding extrapolation method ('yarn', 'ntk', 'base')
        enable_dype: Whether to enable DyPE
        dype_exponent: DyPE strength exponent
        base_shift: Base noise schedule shift
        max_shift: Maximum noise schedule shift
        model_type: Type of model ('standard_flux', 'nunchaku_flux', or None for auto-detection)
    
    Returns:
        Patched model patcher
    """
    m = model.clone()
    
    # Auto-detect model type if not provided
    if model_type is None:
        if is_nunchaku_flux_model(model):
            model_type = "nunchaku_flux"
        else:
            model_type = "standard_flux"
    
    _debug_print(f"🔧 Applying DyPE to {model_type} model for resolution {width}x{height}")
    
    # Handle model sampling patches
    if not hasattr(m.model.model_sampling, "_dype_patched"):
        model_sampler = m.model.model_sampling
        if isinstance(model_sampler, model_sampling.ModelSamplingFlux):
            # Get patch size (same for both model types)
            if hasattr(m.model.diffusion_model, 'patch_size'):
                patch_size = m.model.diffusion_model.patch_size
            else:
                # Fallback for Nunchaku models
                patch_size = 2
            
            latent_h, latent_w = height // 8, width // 8
            padded_h, padded_w = math.ceil(latent_h / patch_size) * patch_size, math.ceil(latent_w / patch_size) * patch_size
            image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)
            base_seq_len, max_seq_len = 256, 4096
            slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            intercept = base_shift - slope * base_seq_len
            dype_shift = image_seq_len * slope + intercept

            def patched_sigma_func(self, timestep):
                return model_sampling.flux_time_shift(dype_shift, 1.0, timestep)

            model_sampler.sigma = types.MethodType(patched_sigma_func, model_sampler)
            model_sampler._dype_patched = True
            _debug_print(f"✅ Applied DyPE noise schedule shift: {dype_shift:.4f} for {image_seq_len} patches")

    # Get positional embedding parameters with proper error handling
    theta, axes_dim = None, None
    
    try:
        if model_type == "nunchaku_flux":
            # CORRECTED: Proper parameter extraction for Nunchaku FLUX models
            wrapper = m.model.diffusion_model
            
            # Debug: Print available attributes for troubleshooting
            _debug_print(f"🔍 Debug: Wrapper type: {type(wrapper).__name__}")
            
            if hasattr(wrapper, 'model'):
                _debug_print(f"🔍 Debug: Wrapper.model type: {type(wrapper.model).__name__}")
                # Check for actual positional embedding parameters in the Nunchaku model
                potential_attrs = [attr for attr in dir(wrapper.model) if any(keyword in attr.lower() for keyword in ['rope', 'pos', 'embed', 'theta', 'axes'])]
                if potential_attrs:
                    _debug_print(f"🔍 Debug: Found potential positional embedding attributes: {potential_attrs}")
                else:
                    _debug_print("🔍 Debug: No direct positional embedding attributes found")
            
            # CORRECTED: Properly extract parameters from Nunchaku model config
            if hasattr(wrapper, 'config') and wrapper.config:
                config = wrapper.config
                theta = config.get('theta', None)
                axes_dim = config.get('axes_dim', None)
                if theta is not None and axes_dim is not None:
                    _debug_print(f"✅ Extracted theta={theta}, axes_dim={axes_dim} from wrapper.config")
                else:
                    _debug_print("⚠️ Config missing required parameters, trying alternative extraction")
            
            # FALLBACK: Try to extract from the underlying Nunchaku model if config extraction failed
            if theta is None or axes_dim is None:
                if hasattr(wrapper, 'model') and hasattr(wrapper.model, 'config'):
                    model_config = getattr(wrapper.model, 'config', {})
                    if theta is None:
                        theta = model_config.get('theta', None)
                    if axes_dim is None:
                        axes_dim = model_config.get('axes_dim', None)
                    if theta is not None and axes_dim is not None:
                        _debug_print(f"✅ Extracted theta={theta}, axes_dim={axes_dim} from wrapper.model.config")
            
            # FALLBACK: Try to extract from model parameters directly if config extraction failed
            if theta is None or axes_dim is None:
                try:
                    # These are standard values for FLUX models based on the model_configs
                    theta = 10000
                    axes_dim = [16, 56, 56]
                    
                    # Try to get actual values from the underlying model if possible
                    if hasattr(wrapper, 'model'):
                        # Nunchaku models typically use these values
                        if hasattr(wrapper.model, 'theta'):
                            theta = wrapper.model.theta
                        if hasattr(wrapper.model, 'axes_dim'):
                            axes_dim = wrapper.model.axes_dim
                    
                    _debug_print(f"✅ Using calculated parameters: theta={theta}, axes_dim={axes_dim}")
                except Exception as e:
                    _debug_print(f"⚠️ Error extracting parameters, using defaults: {e}")
                    theta, axes_dim = 10000, [16, 56, 56]
                    
        else:
            # Standard FLUX model - extract from pe_embedder
            orig_embedder = m.model.diffusion_model.pe_embedder
            theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
            _debug_print(f"✅ Extracted theta={theta}, axes_dim={axes_dim} from standard FLUX pe_embedder")
            
    except AttributeError as e:
        if model_type == "nunchaku_flux":
            # Use robust defaults for Nunchaku FLUX models
            theta, axes_dim = 10000, [16, 56, 56]
            _debug_print(f"⚠️ Could not detect positional embedding parameters for Nunchaku FLUX model, using defaults")
            _debug_print(f"🔍 Debug: AttributeError details: {e}")
        else:
            raise ValueError("The provided model is not a compatible FLUX model.") from e
    
    if theta is None or axes_dim is None:
        raise ValueError(f"Failed to extract required parameters (theta={theta}, axes_dim={axes_dim}) for {model_type} model")

    new_pe_embedder = FluxPosEmbed(theta, axes_dim, method, enable_dype, dype_exponent)
    
    # Apply the positional embedding patch with improved Nunchaku handling
    if model_type == "nunchaku_flux":
        # CORRECTED: Patch the ComfyFluxWrapper.forward method (the method that gets called)
        wrapper = m.model.diffusion_model
        
        _debug_print("🔧 Creating DyPE wrapper for Nunchaku FLUX model (patching forward method)")
        
        # Import the improved Nunchaku compatibility wrapper
        from .nunchaku_compat import create_nunchaku_dype_wrapper
        
        # Create enhanced forward method
        enhanced_forward = create_nunchaku_forward_wrapper(wrapper, new_pe_embedder, enable_dype, debug=False)
        
        # Replace the forward method of the wrapper
        wrapper.forward = types.MethodType(enhanced_forward, wrapper)
        
        # Also add model patcher patch
        m.add_object_patch("diffusion_model.forward", wrapper.forward)
        
        _debug_print("✅ Successfully applied DyPE to Nunchaku FLUX model - patched ComfyFluxWrapper.forward")
        _debug_print("🎯 This ensures DyPE is applied to the img_ids that are actually generated and used")
        
    else:
        # Standard FLUX model
        m.add_object_patch("diffusion_model.pe_embedder", new_pe_embedder)
        _debug_print("✅ Applied DyPE to standard FLUX model via pe_embedder replacement")
    
    # Get sigma_max for the wrapper function
    sigma_max = m.model.model_sampling.sigma_max.item()
    _debug_print(f"📊 Using sigma_max={sigma_max} for timestep normalization")

    def dype_wrapper_function(model_function, args_dict):
        if enable_dype:
            timestep_tensor = args_dict.get("timestep")
            if timestep_tensor is not None and timestep_tensor.numel() > 0:
                current_sigma = timestep_tensor.item()
                if sigma_max > 0:
                    normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                    new_pe_embedder.set_timestep(normalized_timestep)
        
        input_x, c = args_dict.get("input"), args_dict.get("c", {})
        return model_function(input_x, args_dict.get("timestep"), **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)
    
    _debug_print(f"🎉 DyPE successfully applied to {model_type} model for {width}x{height} resolution!")
    return m


def set_dype_patch_debug_mode(enabled=True):
    """
    Set the global debug mode for DyPE patch operations.
    
    Args:
        enabled: Whether to enable verbose debug logging (default: True)
    """
    global DYPE_PATCH_DEBUG
    DYPE_PATCH_DEBUG = enabled


def get_dype_patch_debug_mode():
    """
    Get the current debug mode for DyPE patch operations.
    
    Returns:
        bool: Current debug mode setting
    """
    global DYPE_PATCH_DEBUG
    return DYPE_PATCH_DEBUG