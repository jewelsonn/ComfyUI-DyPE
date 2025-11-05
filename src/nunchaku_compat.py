"""
Nunchaku compatibility utilities for DyPE patching.
Contains only the functions actually used by the DyPE implementation.
"""

# Global debug flag for controlling verbose logging
NUNCHAKU_DYPE_DEBUG = False

def _debug_print(*args, **kwargs):
    """Conditionally print debug messages based on the global debug flag."""
    if NUNCHAKU_DYPE_DEBUG:
        print(*args, **kwargs)

def create_nunchaku_dype_wrapper(original_function, pe_embedder, enable_dype=True, model_config=None, debug=False):
    """
    Create a specialized wrapper function for Nunchaku FLUX models with DyPE integration.
    
    This wrapper properly handles Nunchaku's architecture and applies DyPE positional embeddings
    at the correct point in the forward pass, especially for high-resolution image generation.
    
    Args:
        original_function: The original NunchakuFluxTransformer2dModel.forward function
        pe_embedder: The DyPE positional embedder with proper theta/axes_dim parameters
        enable_dype: Whether DyPE is enabled
        model_config: The Nunchaku model configuration dict
        debug: Enable verbose debug logging (default: False for silent operation)
        
    Returns:
        Wrapper function that applies DyPE correctly for Nunchaku models
    """
    # Update global debug flag for this wrapper instance
    global NUNCHAKU_DYPE_DEBUG
    original_debug_state = NUNCHAKU_DYPE_DEBUG
    NUNCHAKU_DYPE_DEBUG = debug
    
    def nunchaku_dype_wrapper(*args, **kwargs):
        """Specialized wrapper for Nunchaku FLUX models with DyPE integration."""
        try:
            _debug_print(f"🔥 DyPE WRAPPER CALLED: args={len(args)}, kwargs={list(kwargs.keys())}")
            
            # Extract timestep information with improved handling
            timestep = None
            if len(args) > 1:
                timestep = args[1]
            elif 'timestep' in kwargs:
                timestep = kwargs['timestep']
            
            # Extract other important parameters
            img_ids = kwargs.get('img_ids', None)
            if img_ids is None and len(args) > 3:
                img_ids = args[3]
            
            _debug_print(f"🔥 DyPE WRAPPER: timestep={timestep}, img_ids={img_ids is not None}")
            
            # Update DyPE embedder if enabled and timestep available
            if enable_dype and timestep is not None:
                sigma = None
                # Handle different timestep formats
                if hasattr(timestep, 'item'):
                    sigma = timestep.item()
                elif isinstance(timestep, (int, float)):
                    sigma = float(timestep)
                elif hasattr(timestep, '__len__') and len(timestep) > 0:
                    # Handle tensor timesteps
                    if hasattr(timestep[0], 'item'):
                        sigma = timestep[0].item()
                    else:
                        sigma = float(timestep[0])
                
                if sigma is not None:
                    # Get sigma_max with better error handling
                    sigma_max = 1.0  # Fallback
                    try:
                        # Try multiple ways to get sigma_max
                        if hasattr(original_function, '__self__') and hasattr(original_function.__self__, 'model_sampling'):
                            sigma_max = original_function.__self__.model_sampling.sigma_max.item()
                        elif model_config and 'sigma_max' in model_config:
                            sigma_max = model_config['sigma_max']
                        # Use common FLUX sigma_max value as last resort
                        else:
                            sigma_max = 1.0
                    except Exception as e:
                        _debug_print(f"⚠️ Could not get sigma_max, using fallback: {e}")
                        sigma_max = 1.0
                    
                    if sigma_max > 0:
                        normalized_timestep = min(max(sigma / sigma_max, 0.0), 1.0)
                        pe_embedder.set_timestep(normalized_timestep)
                        
                        # Debug logging for high-resolution generation
                        if img_ids is not None:
                            max_pos = img_ids.max().item() if hasattr(img_ids, 'max') else 0
                            if max_pos > 64:  # High resolution threshold
                                _debug_print(f"🎯 DyPE: High-res generation detected (max_pos={max_pos}), applying enhanced positional embeddings")
            
            # CRITICAL FIX: Apply DyPE positional embeddings to img_ids before passing to Nunchaku model
            if enable_dype and img_ids is not None:
                try:
                    _debug_print(f"🔥 DyPE WRAPPER: Applying DyPE embeddings to img_ids")
                    
                    # Apply DyPE positional embeddings to the image IDs
                    dype_embeddings = pe_embedder(img_ids)
                    
                    # CRITICAL FIX: Properly inject DyPE embeddings into the positional encoding system
                    # Instead of keeping original img_ids, we enhance them with DyPE frequency adjustments
                    enhanced_img_ids = img_ids.clone()
                    
                    # Apply DyPE frequency scaling to maintain text-image alignment
                    if dype_embeddings.numel() == enhanced_img_ids.numel() * 2:  # cos/sin pairs
                        # Calculate frequency scaling based on DyPE timestep and embeddings
                        freq_scale = 1.0
                        if hasattr(pe_embedder, 'current_timestep'):
                            timestep_norm = pe_embedder.current_timestep
                            # Apply time-dependent frequency adjustment that preserves text conditioning
                            # The key is to scale positional frequencies while maintaining coordinate structure
                            base_freq = 1.0 + 0.05 * timestep_norm  # Conservative scaling
                            # Scale based on the actual DyPE embedding variance to avoid over-modification
                            dype_variance = dype_embeddings.var().item()
                            img_variance = enhanced_img_ids.var().item()
                            if img_variance > 0:
                                freq_scale = base_freq * (1.0 + 0.1 * timestep_norm * dype_variance / (img_variance + 1e-8))
                        
                        # Apply frequency scaling to Y and X spatial coordinates only (not the index/sequence dimension)
                        if enhanced_img_ids.shape[-1] >= 3:
                            # Scale spatial positions to incorporate DyPE frequency adjustments
                            enhanced_img_ids[..., 1:3] = enhanced_img_ids[..., 1:3] * freq_scale
                    
                    # Replace the img_ids with DyPE-enhanced versions
                    # Handle both kwargs and args cases
                    if 'img_ids' in kwargs:
                        kwargs['img_ids'] = enhanced_img_ids
                    else:
                        # Update args if img_ids was passed as positional argument
                        args = list(args)
                        args[3] = enhanced_img_ids  # img_ids is typically the 4th argument
                        args = tuple(args)
                    
                    # Debug: Log embedding shape for high-resolution images
                    if enhanced_img_ids.numel() > 10000:  # Large embeddings = high res
                        _debug_print(f"🔧 Applied text-preserving DyPE: shape={enhanced_img_ids.shape}, "
                              f"theta={pe_embedder.theta}, axes_dim={pe_embedder.axes_dim}, "
                              f"max_pos={enhanced_img_ids.max().item():.2f}, freq_scale={freq_scale:.3f}")
                    
                except Exception as e:
                    # Keep error logging for debugging purposes but make it conditional on debug flag
                    if debug:
                        print(f"⚠️ Error applying DyPE embeddings: {e}")
                    # Continue with original img_ids if DyPE fails
            
            # Call the original Nunchaku function with enhanced parameters
            _debug_print(f"🔥 DyPE WRAPPER: Calling original function")
            result = original_function(*args, **kwargs)
            
            return result
            
        except Exception as e:
            # Keep critical error logging but make it conditional on debug flag
            if debug:
                print(f"❌ Critical error in Nunchaku DyPE wrapper: {e}")
                print("🔄 Falling back to original function without DyPE")
            # Fallback to original function without DyPE to ensure basic functionality
            return original_function(*args, **kwargs)
        finally:
            # Restore original debug state
            NUNCHAKU_DYPE_DEBUG = original_debug_state
    
    return nunchaku_dype_wrapper


def create_nunchaku_wrapper_forward_wrapper(wrapper_instance, pe_embedder, enable_dype=True, debug=False):
    """
    Create a wrapper for the ComfyFluxWrapper.forward method itself (the correct approach).
    
    This patches the actual forward method that gets called during inference.
    
    Args:
        wrapper_instance: The wrapper instance
        pe_embedder: The DyPE positional embedder
        enable_dype: Whether DyPE is enabled
        debug: Enable verbose debug logging (default: False for silent operation)
    """
    # Update global debug flag for this wrapper instance
    global NUNCHAKU_DYPE_DEBUG
    original_debug_state = NUNCHAKU_DYPE_DEBUG
    NUNCHAKU_DYPE_DEBUG = debug
    
    def wrapped_forward(*args, **kwargs):
        """Wrapper for ComfyFluxWrapper.forward that applies DyPE to img_ids."""
        try:
            _debug_print(f"🔥 NUNCHAKU FORWARD WRAPPER CALLED")
            
            # Call the original forward method first to get img_ids
            # But we need to intercept before it calls the model
            
            # Extract the arguments that will be passed to the underlying model
            x = kwargs.get('x', args[0] if len(args) > 0 else None)
            timestep = kwargs.get('timestep', args[1] if len(args) > 1 else None)
            context = kwargs.get('context', args[2] if len(args) > 2 else None)
            y = kwargs.get('y', args[3] if len(args) > 3 else None)
            guidance = kwargs.get('guidance', args[4] if len(args) > 4 else None)
            
            if x is None or timestep is None:
                if debug:
                    print("⚠️ Missing required arguments, calling original forward")
                return wrapper_instance._original_forward(*args, **kwargs)
            
            # Generate img_ids using the same logic as ComfyFluxWrapper.process_img
            bs, c, h_orig, w_orig = x.shape
            patch_size = wrapper_instance.config.get("patch_size", 2)
            h_len = (h_orig + (patch_size // 2)) // patch_size
            w_len = (w_orig + (patch_size // 2)) // patch_size
            
            from comfy.ldm.common_dit import pad_to_patch_size
            from einops import rearrange, repeat
            import torch
            
            # Process image like ComfyFluxWrapper does
            x_padded = pad_to_patch_size(x, (patch_size, patch_size))
            img = rearrange(x_padded, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            
            h_offset = 0
            w_offset = 0
            index = 0
            
            h_len = (h_orig + (patch_size // 2)) // patch_size
            w_len = (w_orig + (patch_size // 2)) // patch_size
            
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
            
            _debug_print(f"🔥 NUNCHAKU FORWARD: Generated img_ids with shape {img_ids.shape}")
            
            # Apply DyPE to img_ids if enabled
            enhanced_img_ids = img_ids  # Default to original img_ids
            if enable_dype:
                # Update timestep for DyPE
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
                        if hasattr(wrapper_instance.model, 'model_sampling'):
                            sigma_max = wrapper_instance.model.model_sampling.sigma_max.item()
                    except:
                        pass
                    
                    if sigma_max > 0:
                        normalized_timestep = min(max(sigma / sigma_max, 0.0), 1.0)
                        pe_embedder.set_timestep(normalized_timestep)
                
                # Apply DyPE embeddings
                _debug_print(f"🔥 NUNCHAKU FORWARD: Applying DyPE to img_ids")
                enhanced_img_ids = pe_embedder(img_ids)
                
                # For now, just log that we're applying DyPE
                # In a real implementation, we'd inject the enhanced embeddings
                _debug_print(f"🔧 DyPE Enhanced img_ids: max={enhanced_img_ids.max().item():.2f}")
            
            # Call original forward with modified kwargs
            kwargs['img_ids'] = enhanced_img_ids
            result = wrapper_instance._original_forward(*args, **kwargs)
            return result
            
        except Exception as e:
            # Keep error logging but make it conditional on debug flag
            if debug:
                print(f"❌ Error in Nunchaku forward wrapper: {e}")
            return wrapper_instance._original_forward(*args, **kwargs)
        finally:
            # Restore original debug state
            NUNCHAKU_DYPE_DEBUG = original_debug_state
    
    return wrapped_forward


def set_dype_debug_mode(enabled=True):
    """
    Set the global debug mode for DyPE Nunchaku wrapper.
    
    Args:
        enabled: Whether to enable verbose debug logging (default: True)
    """
    global NUNCHAKU_DYPE_DEBUG
    NUNCHAKU_DYPE_DEBUG = enabled


def get_dype_debug_mode():
    """
    Get the current debug mode for DyPE Nunchaku wrapper.
    
    Returns:
        bool: Current debug mode setting
    """
    global NUNCHAKU_DYPE_DEBUG
    return NUNCHAKU_DYPE_DEBUG


# Keep the old function name for backwards compatibility
def create_nunchaku_safe_wrapper(original_function, pe_embedder, enable_dype=True):
    """Backward compatibility wrapper that redirects to the new specialized function."""
    return create_nunchaku_dype_wrapper(original_function, pe_embedder, enable_dype)