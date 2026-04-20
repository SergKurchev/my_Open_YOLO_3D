import torch
from loguru import logger
import numpy as np

def convert_minkowski_to_spconv(state_dict):
    """
    Converts a state_dict from MinkowskiEngine based model to spconv based model.
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # MinkowskiEngine uses 'kernel' for weights in layers
        if "backbone" in key and "kernel" in key:
            # 1. Remap key name: kernel -> weight
            new_key = key.replace("kernel", "weight")
            
            # 2. Reshape and Permute weight
            # ME shape: [Vol, In, Out] where Vol = K*K*K
            # spconv shape: [K, K, K, In, Out]
            
            old_shape = value.shape
            if len(old_shape) == 3:
                vol, cin, cout = old_shape
                k = int(round(vol**(1/3)))
                if k**3 == vol:
                    # Standard 3D cubic kernel
                    # ME stores as [Vol, Cin, Cout]
                    # We need to reshape to [k, k, k, Cin, Cout]
                    # The ordering in ME is usually (x, y, z) lexicographical
                    new_value = value.view(k, k, k, cin, cout)
                    new_state_dict[new_key] = new_value
                    logger.debug(f"Converted Conv weight: {key} ({old_shape}) -> {new_key} ({new_value.shape})")
                else:
                    # Non-cubic kernel or something else, handle as fallback
                    new_state_dict[new_key] = value
                    logger.warning(f"Non-cubic kernel detected for {key} (vol={vol}). Copying as-is.")
            else:
                new_state_dict[new_key] = value
        
        elif "backbone" in key and ("running_mean" in key or "running_var" in key or "weight" in key or "bias" in key):
            # Normalization layers or bias
            # spconv uses BatchNorm1d on features, which has identical key names to MinkowskiBatchNorm
            new_state_dict[key] = value
            
        else:
            # Other parameters (e.g. Mask3D heads)
            new_state_dict[key] = value
            
    return new_state_dict

def adapt_checkpoint(checkpoint_path):
    """
    Loads a checkpoint and adapts it if it's from MinkowskiEngine.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        
        # Heuristic check: if most backbone layers have 'kernel' instead of 'weight'
        has_kernel = any("backbone" in k and "kernel" in k for k in state_dict.keys())
        
        if has_kernel:
            logger.info(f"Detected MinkowskiEngine weights in {checkpoint_path}. Adapting to spconv...")
            ckpt["state_dict"] = convert_minkowski_to_spconv(state_dict)
        else:
            logger.info(f"Checkpoint {checkpoint_path} already seems compatible with spconv (or no backbone weights found).")
            
    return ckpt
