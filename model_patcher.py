"""
This module wraps the ComfyUI model patcher for Nunchaku models to load and unload the model correctly.
"""

import logging

import torch
from comfy.model_patcher import ModelPatcher

logger = logging.getLogger(__name__)


class NunchakuModelPatcher(ModelPatcher):
    """
    This class extends the ComfyUI ModelPatcher to provide custom logic for loading and unloading the model correctly.
    
    It automatically:
    - Reinitializes the C++ QuantizedFluxModel when loading to GPU (before sampling)
    - Resets the C++ QuantizedFluxModel when unloading from GPU (after sampling)
    
    This ensures minimal CUDA memory usage when the model is not actively being used.
    """

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """
        Load the diffusion model onto the specified device.
        
        Before loading, reinitializes the C++ model if it was previously reset.

        Parameters
        ----------
        device_to : torch.device or str, optional
            The device to which the diffusion model should be moved.
        lowvram_model_memory : int, optional
            Not used in this implementation.
        force_patch_weights : bool, optional
            Not used in this implementation.
        full_load : bool, optional
            Not used in this implementation.
        """
        # Reinitialize the C++ model before loading to device
        logger.info("NunchakuModelPatcher.load() - reinitializing C++ model before loading to GPU")
        self._reinit_if_needed()
        
        with self.use_ejected():
            # For NunchakuFluxTransformer2dModel (wrapped in ComfyFluxWrapper)
            # The transformer has to_safely, but the wrapper doesn't
            # Access the underlying transformer model
            wrapper = self.model.diffusion_model
            if hasattr(wrapper, 'model') and hasattr(wrapper.model, 'to_safely'):
                # This is a ComfyFluxWrapper wrapping a NunchakuFluxTransformer2dModel
                wrapper.model.to_safely(device_to)
            else:
                # Fallback to regular to() if to_safely doesn't exist
                self.model.diffusion_model.to(device_to)

    def detach(self, unpatch_all: bool = True):
        """
        Detach the model and move it to the offload device.
        
        After detaching, resets the C++ model to free CUDA memory.

        Parameters
        ----------
        unpatch_all : bool, optional
            If True, unpatch all model components (default is True).
        """
        logger.info("NunchakuModelPatcher.detach() - resetting C++ model after unloading from GPU")
        
        self.eject_model()
        
        # For NunchakuFluxTransformer2dModel (wrapped in ComfyFluxWrapper)
        wrapper = self.model.diffusion_model
        if hasattr(wrapper, 'model') and hasattr(wrapper.model, 'to_safely'):
            # This is a ComfyFluxWrapper wrapping a NunchakuFluxTransformer2dModel
            wrapper.model.to_safely(self.offload_device)
        else:
            # Fallback to regular to() if to_safely doesn't exist
            self.model.diffusion_model.to(self.offload_device)
        
        # Reset the C++ model after offloading to free CUDA memory
        self.model.diffusion_model.reset_quantized_model()

    def _reinit_if_needed(self):
        """
        Reinitialize the C++ QuantizedFluxModel if needed.
        
        Retrieves the stored configuration from the transformer and calls reinit_quantized_model().
        """
        try:
            wrapper = self.model.diffusion_model
            transformer = wrapper.model
            
            # Get the stored configuration
            attention_impl = getattr(transformer, '_attention_impl', 'flashattn2')
            offload = getattr(transformer, '_offload', False)
            
            # Check if model uses fp4 from stored metadata or config
            # The precision is typically stored when loading the model
            use_fp4 = getattr(transformer, '_use_fp4', False)
            
            # Determine bf16 setting from dtype
            bf16 = (wrapper.dtype == torch.bfloat16) if hasattr(wrapper, 'dtype') else True
            
            logger.info(f"Reinitializing with: use_fp4={use_fp4}, offload={offload}, bf16={bf16}, attention_impl={attention_impl}")
            wrapper.reinit_quantized_model(
                use_fp4=use_fp4,
                offload=offload,
                bf16=bf16,
                attention_impl=attention_impl
            )
        except Exception as e:
            logger.error(f"Failed to reinitialize model in _reinit_if_needed: {e}")
            import traceback
            traceback.print_exc()
