import tensorflow as tf
import argparse
import os
import sys
import yaml
from pathlib import Path
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

# Add the project root to Python path
root_path = str(Path(__file__).parent)
sys.path.append(root_path)

# Import model building utilities
import cebed.models_with_ssl as cm_ssl
import cebed.models as cm_regular

def load_model_from_weights(model_path, model_name=None, experiment_name=None):
    """
    Load a model from weights file by reconstructing the model architecture.
    
    Args:
        model_path: Path to the .h5 file, .ckpt file, or directory containing checkpoints
        model_name: Name of the model class (will be extracted from config if None)
        experiment_name: Experiment configuration name (will be extracted from config if None)
    
    Returns:
        Loaded TensorFlow model
    """
    try:
        # Handle different path formats
        if os.path.isdir(model_path):
            # If it's a directory, look for checkpoint files and config
            ckpt_path = os.path.join(model_path, "cp.ckpt")
            config_path = os.path.join(model_path, "config.yaml")
            
            if os.path.exists(ckpt_path + ".index"):
                weights_path = ckpt_path
            else:
                # Look for other checkpoint patterns
                import glob
                ckpt_files = glob.glob(os.path.join(model_path, "*.ckpt.index"))
                if ckpt_files:
                    weights_path = ckpt_files[0].replace(".index", "")
                else:
                    raise FileNotFoundError(f"No checkpoint files found in {model_path}")
            
            # Extract model name and experiment name from config.yaml if not provided
            if model_name is None or experiment_name is None:
                if os.path.exists(config_path):
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    if model_name is None:
                        model_name = config.get('model_name', 'ReconMAE')
                    if experiment_name is None:
                        experiment_name = config.get('experiment_name', 'siso_1_umi_block_1_ps2_p72')
                else:
                    # Default values if config not found
                    if model_name is None:
                        model_name = "ReconMAE"
                    if experiment_name is None:
                        experiment_name = "siso_1_umi_block_1_ps2_p72"
                        
        elif model_path.endswith(".ckpt"):
            weights_path = model_path
        elif model_path.endswith(".h5"):
            weights_path = model_path
        else:
            # Check if it's a checkpoint path without extension
            if os.path.exists(model_path + ".index"):
                weights_path = model_path
            else:
                raise FileNotFoundError(f"Cannot find model weights at {model_path}")
        
        # Set default values if still None
        if model_name is None:
            model_name = "ReconMAE"
        if experiment_name is None:
            experiment_name = "siso_1_umi_block_1_ps2_p72"
        
        # Get model hyperparameters from config
        # Try SSL models first, then regular models
        try:
            model_hparams = cm_ssl.get_model_hparams(model_name, experiment_name)
            model_class = cm_ssl.get_model_class(model_name)
            is_ssl_model = True
        except:
            try:
                model_hparams = cm_regular.get_model_hparams(model_name, experiment_name)
                model_class = cm_regular.get_model_class(model_name)
                is_ssl_model = False
            except:
                # Handle special cases like DnCNN that don't follow the BaseModel pattern
                if model_name == "DnCNN":
                    model_class = cm_regular.get_model_class(model_name)
                    # DnCNN doesn't need hyperparameters from config
                    model = model_class()
                    input_tensor = tf.ones([1, 14, 72, 2])  # Standard input for DnCNN
                    model(input_tensor)
                    
                    # Load weights
                    if weights_path.endswith(".h5"):
                        model.load_weights(weights_path)
                    else:
                        model.load_weights(weights_path).expect_partial()
                    return model
                else:
                    raise ValueError(f"Model {model_name} not found in either SSL or regular models")
        
        # Initialize model (for BaseModel-derived classes)
        model = model_class(model_hparams)
        
        # Set up model architecture based on model type
        if model_name == "ReconMAE":
            # Create a more realistic pilot mask - similar to the actual dataset
            pilot_mask = tf.zeros([1, 1, 14, 72])  # [batch, 1, ns, nf]
            # Set pilot symbols at positions 2 and 9 (typical pilot positions)
            pilot_mask = tf.tensor_scatter_nd_update(pilot_mask, [[0, 0, 2]], [tf.ones([72])])
            pilot_mask = tf.tensor_scatter_nd_update(pilot_mask, [[0, 0, 9]], [tf.ones([72])])
            model.set_mask(pilot_mask)
            
            # Create dummy inputs to build the model
            main_input = tf.ones([1, 2, 72, 2])  # [batch, nps, nfs, c]
            aux_input = tf.ones([1, 2, 72, 2])   # [batch, nps, nfs, c]
            
            # Create a dummy mask (for aux input) - shape [batch, ns, nf]
            mask = tf.zeros([1, 14, 72])  # [batch, ns, nf]
            # Set the same pilot positions as above
            mask = tf.tensor_scatter_nd_update(mask, [[0, 2]], [tf.ones([72])])
            mask = tf.tensor_scatter_nd_update(mask, [[0, 9]], [tf.ones([72])])
            
            # Build the model by calling it with dummy inputs
            model((main_input, (aux_input, mask)))
            
        elif model_name == "ReconMAE_MainOnly":
            # For main-only model, it still expects the tuple format but only uses main branch
            pilot_mask = tf.zeros([1, 1, 14, 72])  # [batch, 1, ns, nf]
            # Set pilot symbols at positions 2 and 9 (typical pilot positions)
            pilot_mask = tf.tensor_scatter_nd_update(pilot_mask, [[0, 0, 2]], [tf.ones([72])])
            pilot_mask = tf.tensor_scatter_nd_update(pilot_mask, [[0, 0, 9]], [tf.ones([72])])
            model.set_mask(pilot_mask)
            
            # ReconMAE_MainOnly still expects tuple input format: (main_input, (aux_input, mask))
            main_input = tf.ones([1, 2, 72, 2])  # [batch, nps, nfs, c]
            aux_input = tf.ones([1, 2, 72, 2])   # [batch, nps, nfs, c]
            mask = tf.zeros([1, 14, 72])  # [batch, ns, nf]
            # Set the same pilot positions as above
            mask = tf.tensor_scatter_nd_update(mask, [[0, 2]], [tf.ones([72])])
            mask = tf.tensor_scatter_nd_update(mask, [[0, 9]], [tf.ones([72])])
            
            # Build the model by calling it with tuple inputs
            model((main_input, (aux_input, mask)))
            
        elif model_name == "ReconMAEV2":
            input_shape_main, input_shape_aux = [2, 72, 2], [2, 72, 2]
            main_low_input = tf.ones([1, input_shape_main[0], input_shape_main[1], input_shape_main[2]])
            aux_low_input = tf.ones([1, input_shape_aux[0], input_shape_aux[1], input_shape_aux[2]])
            model((main_low_input, aux_low_input))
        elif model_name == "ReconMAEX":            
            input_shape_main, input_shape_aux = [2, 72, 2], ([2, 72, 2], [14, 72])
            main_low_input = tf.ones([1, input_shape_main[0], input_shape_main[1], input_shape_main[2]])
            aux_one_input = tf.ones([1, input_shape_aux[0][0], input_shape_aux[0][1], input_shape_aux[0][2]])
            aux_two_input = tf.ones([1, input_shape_aux[1][0], input_shape_aux[1][1]])
            model((main_low_input, (aux_one_input, aux_two_input)))
        elif model_name in ["MaeFixMask", "MaeRandomMask"]:
            input_shape_main, input_shape_aux = [2, 72, 2], [14, 72]
            main_low_input = tf.ones([1, input_shape_main[0], input_shape_main[1], input_shape_main[2]])
            aux_low_input = tf.ones([1, input_shape_aux[0], input_shape_aux[1]])
            model((main_low_input, aux_low_input))
            
        else:
            # For other models, try to build with standard input
            try:
                # Try to get input dimensions from hyperparameters
                if hasattr(model, 'hparams') and 'input_dim' in model.hparams:
                    input_dim = model.hparams['input_dim']
                    input_shape = [1] + input_dim
                else:
                    # Use standard input shape for models without explicit input_dim
                    if model_name in ["HA02", "HA03"]:
                        input_shape = [1, 2, 72, 2]  # Pilot-based input
                    else:
                        input_shape = [1, 2, 72, 2]  # Default
                
                input_tensor = tf.ones(input_shape)
                model(input_tensor)
            except Exception as e:
                print(f"Error building model with standard input: {e}")
                # If that fails, try different input shapes based on model type
                if "HA" in model_name:
                    # HA models typically expect [batch, nps, nf, c] input from hyperparams
                    input_shape = [1, 2, 72, 2]
                    input_tensor = tf.ones(input_shape)
                    model(input_tensor)
                else:
                    # For other models, use a generic approach
                    input_shape = [1, 2, 72, 2]
                    input_tensor = tf.ones(input_shape)
                    model(input_tensor)
        
        # Load weights
        if weights_path.endswith(".h5"):
            model.load_weights(weights_path)
        else:
            # For checkpoint files, use expect_partial() to handle partial loading
            model.load_weights(weights_path).expect_partial()
        
        return model
        
    except Exception as e:
        print(f"Error loading model from weights: {e}")
        print("Available SSL models:", cm_ssl.MODELS if hasattr(cm_ssl, 'MODELS') else "N/A")
        print("Available regular models:", cm_regular.MODELS if hasattr(cm_regular, 'MODELS') else "N/A")
        return None

def get_flops(model, batch_size=1):
    """
    Calculate FLOPs for a tf.keras.Model or a tf.function.
    Args:
        model: A tf.keras.Model object.
        batch_size: The batch size to use for the calculation.
    Returns:
        The total number of FLOPs.
    """
    # Handle different input structures for different models
    model_class_name = model.__class__.__name__
    
    try:
        # More specific checks first
        if 'ReconMAEX' in model_class_name:
            main_input = tf.ones([batch_size, 2, 72, 2])
            aux_input_x1 = tf.ones([batch_size, 2, 72, 2])
            aux_input_x2 = tf.zeros([batch_size, 14, 72])
            
            @tf.function
            def model_func(inputs):
                return model(inputs)
            
            concrete_func = model_func.get_concrete_function(
                (tf.TensorSpec(main_input.shape, main_input.dtype),
                 (tf.TensorSpec(aux_input_x1.shape, aux_input_x1.dtype),
                  tf.TensorSpec(aux_input_x2.shape, aux_input_x2.dtype)))
            )
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        elif 'ReconMAE_MainOnly' in model_class_name:
            input_shape = [batch_size, 2, 72, 2]
            
            @tf.function
            def model_func(x):
                return model(x)
            
            concrete_func = model_func.get_concrete_function(
                tf.TensorSpec(input_shape, tf.float32)
            )
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        elif 'ReconMAE' in model_class_name:
            main_input = tf.ones([batch_size, 2, 72, 2])
            aux_input = tf.ones([batch_size, 2, 72, 2])
            mask = tf.zeros([batch_size, 14, 72])  # [batch, ns, nf]
            # Set pilot positions
            mask = tf.tensor_scatter_nd_update(mask, [[0, 2]], [tf.ones([72])])
            
            # Create a concrete function with the specific input structure
            @tf.function
            def model_func(main_input, aux_input, mask):
                return model((main_input, (aux_input, mask)))
            
            # Get concrete function for FLOPs calculation
            concrete_func = model_func.get_concrete_function(
                tf.TensorSpec(main_input.shape, main_input.dtype),
                tf.TensorSpec(aux_input.shape, aux_input.dtype),
                tf.TensorSpec(mask.shape, mask.dtype)
            )
            
            # Convert to constants for FLOPs calculation
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
        elif 'MaeFixMask' in model_class_name or 'MaeRandomMask' in model_class_name:
            # MAE models with mask inputs
            main_input = tf.ones([batch_size, 2, 72, 2])  # [batch, nps, nfs, c]
            mask = tf.zeros([batch_size, 14, 72])  # [batch, ns, nf]
            mask = tf.tensor_scatter_nd_update(mask, [[0, 2]], [tf.ones([72])])
            
            @tf.function
            def model_func(main_input, mask):
                return model((main_input, mask))
            
            concrete_func = model_func.get_concrete_function(
                tf.TensorSpec(main_input.shape, main_input.dtype),
                tf.TensorSpec(mask.shape, mask.dtype)
            )
            
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
        else:
            # Standard models that expect single tensor input
            input_shape = [batch_size, 14, 72, 2]  # Standard input shape for most models
            
            # Adjust input shape based on model type
            if 'DnCNN' in model_class_name:
                # DnCNN expects full channel input
                input_shape = [batch_size, 14, 72, 2]
            elif any(name in model_class_name for name in ['HA02', 'HA03', 'ChannelNet', 'ReEsNet', 'DenoiseNet']):
                # These models typically work with pilot-based input
                input_shape = [batch_size, 2, 72, 2]  # [batch, nps, nfs, c]
            
            # Create concrete function with the appropriate input shape
            @tf.function
            def model_func(x):
                return model(x)
            
            concrete_func = model_func.get_concrete_function(
                tf.TensorSpec(input_shape, tf.float32)
            )
            
            # Convert the variables in the concrete function to constants
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
    except Exception as e:
        print(f"Error creating concrete function for {model_class_name}: {e}")
        # Fallback to standard approach
        input_shape = [batch_size, 2, 72, 2]
        
        @tf.function
        def model_func(x):
            return model(x)
        
        concrete_func = model_func.get_concrete_function(
            tf.TensorSpec(input_shape, tf.float32)
        )
        
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    # Create a graph from the graph_def and calculate FLOPs
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        # Use the TF profiler to calculate FLOPs
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops

def get_memory_usage(model, batch_size=1):
    """
    Calculate memory usage for a model with improved estimation.
    """
    # Get model parameters memory
    total_params = model.count_params()
    param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    # Estimate activation memory (improved approximation)
    activation_memory_mb = 0
    model_class_name = model.__class__.__name__
    
    try:
        # Use model-specific input shapes for more accurate memory estimation
        if 'ReconMAEX' in model_class_name:
            # Main branch + Aux branch
            activation_memory_mb = (batch_size * 2 * 72 * 2 * 4) * 2 / (1024*1024)
        elif 'ReconMAE' in model_class_name:
            # Main branch + Aux branch
            activation_memory_mb = (batch_size * 2 * 72 * 2 * 4) * 2 / (1024*1024)
        elif 'DnCNN' in model_class_name:
            # Full channel input
            activation_memory_mb = batch_size * 14 * 72 * 2 * 4 / (1024 * 1024)
        elif any(name in model_class_name for name in ['HA02', 'HA03']):
            # Transformer-based models
            # Input: [batch, 2, 72, 2] -> [batch, 2, 144] (tokenized)
            # Encoder: [batch, 2, 144] -> [batch, 2, 144]
            # Decoder: [batch, 144, 2] -> [batch, 14, 72, 2]
            activation_memory_mb = (
                batch_size * (2 * 144 * 2 + 2 * 144 + 14 * 72 * 2) * 4 / (1024 * 1024)
            )
        elif 'ChannelNet' in model_class_name:
            # ChannelNet has both SR and denoising components
            # Input: [batch, 2, 72, 2]
            # SR component: multiple conv layers
            # Denoising component: DnCNN-like architecture
            activation_memory_mb = batch_size * 14 * 72 * 128 * 4 / (1024 * 1024)
        else:
            # Generic estimation for other models
            for layer in model.layers:
                if hasattr(layer, 'output_shape') and layer.output_shape:
                    # Calculate activation size
                    shape = layer.output_shape
                    if shape and len(shape) > 1:
                        size = batch_size
                        for dim in shape[1:]:  # Skip batch dimension
                            if dim is not None:
                                size *= dim
                        activation_memory_mb += size * 4 / (1024 * 1024)  # 4 bytes per float32
    except Exception as e:
        print(f"Warning: Error estimating activation memory: {e}")
        # Fallback to basic estimation
        activation_memory_mb = batch_size * 14 * 72 * 32 * 4 / (1024 * 1024)  # Basic estimate
    
    total_memory_mb = param_memory_mb + activation_memory_mb
    
    return {
        'param_memory_mb': param_memory_mb,
        'activation_memory_mb': activation_memory_mb,
        'total_memory_mb': total_memory_mb
    }

def calculate_efficiency_metrics(model, flops, batch_size=1):
    """
    Calculate various efficiency metrics for a model.
    """
    # Basic metrics
    total_params = model.count_params()
    memory_info = get_memory_usage(model, batch_size)
    
    # Efficiency metrics
    flops_per_param = flops / total_params if total_params > 0 else 0
    flops_per_mb = flops / memory_info['total_memory_mb'] if memory_info['total_memory_mb'] > 0 else 0
    
    return {
        'total_params': total_params,
        'flops': flops,
        'flops_per_param': flops_per_param,
        'flops_per_mb': flops_per_mb,
        'memory_mb': memory_info['total_memory_mb'],
        'param_memory_mb': memory_info['param_memory_mb'],
        'activation_memory_mb': memory_info['activation_memory_mb']
    }

def get_layer_flops(model, layer_name, batch_size=1):
    """
    Calculate FLOPs for a specific layer in a model.
    
    Args:
        model: TensorFlow model
        layer_name: Name of the layer to calculate FLOPs for
        batch_size: Batch size for calculation
        
    Returns:
        FLOPs for the specified layer
    """
    # Find the layer by name
    layer = None
    for l in model.layers:
        if l.name == layer_name:
            layer = l
            break
    
    if layer is None:
        print(f"Layer '{layer_name}' not found in model")
        return 0
    
    # Create a simplified model with just this layer
    try:
        # Get the layer's input shape
        input_shape = layer.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        # Create dummy input
        dummy_input = tf.ones([batch_size] + list(input_shape[1:]))
        
        # Create a function that calls just this layer
        @tf.function
        def layer_func(x):
            return layer(x)
        
        concrete_func = layer_func.get_concrete_function(
            tf.TensorSpec(dummy_input.shape, dummy_input.dtype)
        )
        
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        # Calculate FLOPs
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            
            return flops.total_float_ops
    
    except Exception as e:
        print(f"Error calculating FLOPs for layer '{layer_name}': {e}")
        return 0

def get_component_flops(model, batch_size=1):
    """
    Calculate FLOPs for major components of different model architectures.
    
    Args:
        model: TensorFlow model
        batch_size: Batch size for calculation
        
    Returns:
        Dictionary with component-wise FLOPs
    """
    model_class_name = model.__class__.__name__
    component_flops = {}
    
    try:
        if 'ReconMAEX' in model_class_name:
            component_flops = _get_reconmaex_component_flops(model, batch_size)
        elif 'ReconMAE_MainOnly' in model_class_name:
            component_flops = _get_reconmae_main_only_component_flops(model, batch_size)
        elif 'ReconMAE' in model_class_name:
            # ReconMAE has shared encoder and separate decoders
            component_flops.update(_get_reconmae_component_flops(model, batch_size))
        elif 'HA02' in model_class_name or 'HA03' in model_class_name:
            # HA02/HA03 have transformer encoder and decoder
            component_flops.update(_get_ha_component_flops(model, batch_size))
        elif 'ReEsNet' in model_class_name or 'InReEsNet' in model_class_name or 'MReEsNet' in model_class_name:
            # ResNet-based models
            component_flops.update(_get_resnet_component_flops(model, batch_size))
        elif 'DnCNN' in model_class_name:
            # DnCNN model
            component_flops.update(_get_dncnn_component_flops(model, batch_size))
        elif 'ChannelNet' in model_class_name:
            # ChannelNet model
            component_flops.update(_get_channelnet_component_flops(model, batch_size))
        elif 'MaeFixMask' in model_class_name or 'MaeRandomMask' in model_class_name:
            # MAE models
            component_flops.update(_get_mae_component_flops(model, batch_size))
        else:
            # Generic approach - try to identify common layer patterns
            component_flops.update(_get_generic_component_flops(model, batch_size))
    
    except Exception as e:
        print(f"Error calculating component FLOPs for {model_class_name}: {e}")
        component_flops = {"error": f"Failed to calculate: {e}"}
    
    return component_flops

def _get_reconmae_component_flops(model, batch_size):
    """Calculate component FLOPs for ReconMAE model"""
    components = {}
    
    # Prepare inputs for ReconMAE
    main_input = tf.ones([batch_size, 2, 72, 2])  # [batch, nps, nfs, c]
    aux_input = tf.ones([batch_size, 2, 72, 2])   # [batch, nps, nfs, c]
    mask = tf.zeros([batch_size, 14, 72])  # [batch, ns, nf]
    # Set pilot positions
    mask = tf.tensor_scatter_nd_update(mask, [[0, 2]], [tf.ones([72])])
    mask = tf.tensor_scatter_nd_update(mask, [[0, 9]], [tf.ones([72])])
    
    # Calculate encoder FLOPs
    try:
        # Get shared encoder
        encoder = model.encoder
        
        # Tokenize main input
        main_tokenized = model.tokenize_input(main_input)
        
        @tf.function
        def encoder_func(x):
            return encoder(x)
        
        concrete_func = encoder_func.get_concrete_function(
            tf.TensorSpec(main_tokenized.shape, main_tokenized.dtype)
        )
        
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            components['encoder'] = flops.total_float_ops
            
    except Exception as e:
        components['encoder'] = f"Error: {e}"
    
    # Calculate main decoder FLOPs
    try:
        main_decoder = model.main_decoder
        
        # Get decoder input shape
        main_latent = model.tokenize_input(main_input)
        main_latent = model.encoder(main_latent)
        main_latent = model.post_encoder_reshape(main_latent)
        main_expanded = model.main_expand_batch(main_latent)
        
        @tf.function
        def main_decoder_func(x):
            return main_decoder(x)
        
        concrete_func = main_decoder_func.get_concrete_function(
            tf.TensorSpec(main_expanded.shape, main_expanded.dtype)
        )
        
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            components['main_decoder'] = flops.total_float_ops
            
    except Exception as e:
        components['main_decoder'] = f"Error: {e}"
    
    # Calculate aux decoder FLOPs
    try:
        aux_decoder = model.aux_decoder
        
        # Get aux decoder input shape
        aux_latent = model.tokenize_input(aux_input)
        aux_latent = model.encoder(aux_latent)
        aux_latent = model.post_encoder_reshape(aux_latent)
        aux_expanded = model.aux_expand_batch(aux_latent, mask)
        
        @tf.function
        def aux_decoder_func(x):
            return aux_decoder(x)
        
        concrete_func = aux_decoder_func.get_concrete_function(
            tf.TensorSpec(aux_expanded.shape, aux_expanded.dtype)
        )
        
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            components['aux_decoder'] = flops.total_float_ops
            
    except Exception as e:
        components['aux_decoder'] = f"Error: {e}"
    
    return components

def _get_reconmaex_component_flops(model, batch_size):
    """Calculate component FLOPs for ReconMAEX model"""
    components = {}
    
    # Prepare inputs for ReconMAEX
    main_input = tf.ones([batch_size, 2, 72, 2])  # [batch, nps, nfs, c]
    aux_input_x1 = tf.ones([batch_size, 2, 72, 2])   # [batch, nps, nfs, c]
    aux_input_x2 = tf.zeros([batch_size, 14, 72])  # [batch, ns, nf]
    
    # Calculate encoder FLOPs
    try:
        # Get shared encoder
        encoder = model.encoder
        
        # Tokenize main input
        main_tokenized = model.tokenize_input(main_input)
        
        @tf.function
        def encoder_func(x):
            return encoder(x)
        
        concrete_func = encoder_func.get_concrete_function(
            tf.TensorSpec(main_tokenized.shape, main_tokenized.dtype)
        )
        
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            components['encoder'] = flops.total_float_ops
            
    except Exception as e:
        components['encoder'] = f"Error: {e}"
    
    # Calculate main decoder FLOPs
    try:
        main_decoder = model.main_decoder
        
        # Get decoder input shape
        main_latent = model.tokenize_input(main_input)
        main_latent = model.encoder(main_latent)
        main_latent = model.post_encoder_reshape(main_latent)
        main_expanded = model.main_expand_batch(main_latent)
        
        @tf.function
        def main_decoder_func(x):
            return main_decoder(x)
        
        concrete_func = main_decoder_func.get_concrete_function(
            tf.TensorSpec(main_expanded.shape, main_expanded.dtype)
        )
        
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            components['main_decoder'] = flops.total_float_ops
            
    except Exception as e:
        components['main_decoder'] = f"Error: {e}"
    
    # Calculate aux decoder FLOPs
    try:
        aux_decoder = model.aux_decoder
        
        # Get aux decoder input shape
        aux_latent = model.tokenize_input(aux_input_x1)
        aux_latent = model.encoder(aux_latent)
        aux_latent = model.post_encoder_reshape(aux_latent)
        aux_expanded = model.aux_expand_batch(aux_latent, aux_input_x2)
        
        @tf.function
        def aux_decoder_func(x):
            return aux_decoder(x)
        
        concrete_func = aux_decoder_func.get_concrete_function(
            tf.TensorSpec(aux_expanded.shape, aux_expanded.dtype)
        )
        
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            components['aux_decoder'] = flops.total_float_ops
            
    except Exception as e:
        components['aux_decoder'] = f"Error: {e}"
    
    return components

def _get_reconmae_main_only_component_flops(model, batch_size):
    """Calculate component FLOPs for ReconMAE_MainOnly model"""
    components = {}
    
    # Prepare inputs
    main_input = tf.ones([batch_size, 2, 72, 2])  # [batch, nps, nfs, c]
    
    # Calculate encoder FLOPs
    try:
        encoder = model.encoder
        main_tokenized = model.tokenize_input(main_input)
        
        @tf.function
        def encoder_func(x):
            return encoder(x)
        
        concrete_func = encoder_func.get_concrete_function(
            tf.TensorSpec(main_tokenized.shape, main_tokenized.dtype)
        )
        
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            components['encoder'] = flops.total_float_ops
            
    except Exception as e:
        components['encoder'] = f"Error: {e}"
    
    # Calculate main decoder FLOPs
    try:
        main_decoder = model.main_decoder
        
        # Get decoder input shape
        main_latent = model.tokenize_input(main_input)
        main_latent = model.encoder(main_latent)
        main_latent = model.post_encoder_reshape(main_latent)
        main_expanded = model.main_expand_batch(main_latent)
        
        @tf.function
        def main_decoder_func(x):
            return main_decoder(x)
        
        concrete_func = main_decoder_func.get_concrete_function(
            tf.TensorSpec(main_expanded.shape, main_expanded.dtype)
        )
        
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            components['main_decoder'] = flops.total_float_ops
            
    except Exception as e:
        components['main_decoder'] = f"Error: {e}"
    
    return components

def _get_ha_component_flops(model, batch_size):
    """Calculate component FLOPs for HA02/HA03 models"""
    components = {}
    
    # Prepare input
    input_tensor = tf.ones([batch_size, 2, 72, 2])  # [batch, nps, nfs, c]
    
    # Try to get encoder and decoder components
    try:
        if hasattr(model, 'encoder'):
            encoder = model.encoder
            # Tokenize input for encoder
            reshaped_input = tf.keras.layers.Reshape((-1, input_tensor.shape[-1]))(input_tensor)
            permuted_input = tf.keras.layers.Permute((2, 1))(reshaped_input)
            
            @tf.function
            def encoder_func(x):
                return encoder(x)
            
            concrete_func = encoder_func.get_concrete_function(
                tf.TensorSpec(permuted_input.shape, permuted_input.dtype)
            )
            
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(graph_def, name='')
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                components['encoder'] = flops.total_float_ops
                
    except Exception as e:
        components['encoder'] = f"Error: {e}"
    
    # Try to get decoder components
    try:
        if hasattr(model, 'decoder'):
            decoder = model.decoder
            
            # Get decoder input (after encoder processing)
            reshaped_input = tf.keras.layers.Reshape((-1, input_tensor.shape[-1]))(input_tensor)
            permuted_input = tf.keras.layers.Permute((2, 1))(reshaped_input)
            if hasattr(model, 'encoder'):
                encoder_output = model.encoder(permuted_input)
                decoder_input = tf.keras.layers.Permute((2, 1))(encoder_output)
            else:
                decoder_input = permuted_input
            
            @tf.function
            def decoder_func(x):
                return decoder(x)
            
            concrete_func = decoder_func.get_concrete_function(
                tf.TensorSpec(decoder_input.shape, decoder_input.dtype)
            )
            
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(graph_def, name='')
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                components['decoder'] = flops.total_float_ops
                
    except Exception as e:
        components['decoder'] = f"Error: {e}"
    
    return components

def _get_resnet_component_flops(model, batch_size):
    """Calculate component FLOPs for ResNet-based models (ReEsNet, InReEsNet, MReEsNet)"""
    components = {}
    
    # Prepare input based on model type
    if hasattr(model, 'input_dim'):
        input_shape = [batch_size] + list(model.input_dim)
    else:
        input_shape = [batch_size, 2, 72, 2]  # Default shape
    
    input_tensor = tf.ones(input_shape)
    
    # Calculate FLOPs for different components
    try:
        # Initial convolution
        if hasattr(model, 'conv1'):
            conv1_output = model.conv1(input_tensor)
            
            @tf.function
            def conv1_func(x):
                return model.conv1(x)
            
            concrete_func = conv1_func.get_concrete_function(
                tf.TensorSpec(input_tensor.shape, input_tensor.dtype)
            )
            
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(graph_def, name='')
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                components['initial_conv'] = flops.total_float_ops
                
    except Exception as e:
        components['initial_conv'] = f"Error: {e}"
    
    # Calculate backbone FLOPs
    try:
        if hasattr(model, 'backbone'):
            backbone_input = model.conv1(input_tensor) if hasattr(model, 'conv1') else input_tensor
            
            @tf.function
            def backbone_func(x):
                return model.backbone(x)
            
            concrete_func = backbone_func.get_concrete_function(
                tf.TensorSpec(backbone_input.shape, backbone_input.dtype)
            )
            
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(graph_def, name='')
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                components['backbone'] = flops.total_float_ops
                
    except Exception as e:
        components['backbone'] = f"Error: {e}"
    
    # Calculate upsampling FLOPs
    try:
        if hasattr(model, 'upsampling') and model.upsampling is not None:
            # Get upsampling input
            features = model.conv1(input_tensor) if hasattr(model, 'conv1') else input_tensor
            if hasattr(model, 'backbone'):
                features = model.backbone(features)
                features = tf.keras.layers.Add()([model.conv1(input_tensor), features])
            
            @tf.function
            def upsampling_func(x):
                return model.upsampling(x)
            
            concrete_func = upsampling_func.get_concrete_function(
                tf.TensorSpec(features.shape, features.dtype)
            )
            
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(graph_def, name='')
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                components['upsampling'] = flops.total_float_ops
                
    except Exception as e:
        components['upsampling'] = f"Error: {e}"
    
    return components

def _get_dncnn_component_flops(model, batch_size):
    """Calculate component FLOPs for DnCNN model"""
    components = {}
    
    # DnCNN typically processes full channel input
    input_tensor = tf.ones([batch_size, 14, 72, 2])
    
    # Try to analyze layer by layer
    try:
        layer_flops = {}
        current_input = input_tensor
        
        for i, layer in enumerate(model.layers):
            try:
                @tf.function
                def layer_func(x):
                    return layer(x)
                
                concrete_func = layer_func.get_concrete_function(
                    tf.TensorSpec(current_input.shape, current_input.dtype)
                )
                
                frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
                
                with tf.Graph().as_default() as graph:
                    tf.graph_util.import_graph_def(graph_def, name='')
                    run_meta = tf.compat.v1.RunMetadata()
                    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                    flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                    layer_flops[f'layer_{i}_{layer.name}'] = flops.total_float_ops
                
                # Update current_input for next layer
                current_input = layer(current_input)
                
            except Exception as e:
                layer_flops[f'layer_{i}_{layer.name}'] = f"Error: {e}"
        
        components.update(layer_flops)
        
    except Exception as e:
        components['layers'] = f"Error: {e}"
    
    return components

def _get_channelnet_component_flops(model, batch_size):
    """Calculate component FLOPs for ChannelNet model"""
    components = {}
    
    # ChannelNet typically uses low-dimensional input
    input_tensor = tf.ones([batch_size, 2, 72, 2])
    
    # Try to get major components
    try:
        # Similar to DnCNN, analyze layer by layer
        layer_flops = {}
        current_input = input_tensor
        
        for i, layer in enumerate(model.layers):
            try:
                @tf.function
                def layer_func(x):
                    return layer(x)
                
                concrete_func = layer_func.get_concrete_function(
                    tf.TensorSpec(current_input.shape, current_input.dtype)
                )
                
                frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
                
                with tf.Graph().as_default() as graph:
                    tf.graph_util.import_graph_def(graph_def, name='')
                    run_meta = tf.compat.v1.RunMetadata()
                    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                    flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                    layer_flops[f'layer_{i}_{layer.name}'] = flops.total_float_ops
                
                # Update current_input for next layer
                current_input = layer(current_input)
                
            except Exception as e:
                layer_flops[f'layer_{i}_{layer.name}'] = f"Error: {e}"
        
        components.update(layer_flops)
        
    except Exception as e:
        components['layers'] = f"Error: {e}"
    
    return components

def _get_mae_component_flops(model, batch_size):
    """Calculate component FLOPs for MAE models"""
    components = {}
    
    # Prepare inputs for MAE
    main_input = tf.ones([batch_size, 2, 72, 2])  # [batch, nps, nfs, c]
    mask = tf.zeros([batch_size, 14, 72])  # [batch, ns, nf]
    mask = tf.tensor_scatter_nd_update(mask, [[0, 2]], [tf.ones([72])])
    
    # Calculate encoder FLOPs
    try:
        if hasattr(model, 'encoder'):
            encoder = model.encoder
            
            # Tokenize input for encoder
            reshaped_input = tf.keras.layers.Reshape((-1, main_input.shape[-1]))(main_input)
            permuted_input = tf.keras.layers.Permute((2, 1))(reshaped_input)
            
            @tf.function
            def encoder_func(x):
                return encoder(x)
            
            concrete_func = encoder_func.get_concrete_function(
                tf.TensorSpec(permuted_input.shape, permuted_input.dtype)
            )
            
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(graph_def, name='')
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                components['encoder'] = flops.total_float_ops
                
    except Exception as e:
        components['encoder'] = f"Error: {e}"
    
    # Calculate decoder FLOPs
    try:
        if hasattr(model, 'decoder'):
            decoder = model.decoder
            
            # Get decoder input
            reshaped_input = tf.keras.layers.Reshape((-1, main_input.shape[-1]))(main_input)
            permuted_input = tf.keras.layers.Permute((2, 1))(reshaped_input)
            encoder_output = model.encoder(permuted_input)
            latent = tf.keras.layers.Permute((2, 1))(encoder_output)
            expanded_latent = model.expand_batch(latent) if hasattr(model, 'expand_batch') else latent
            
            @tf.function
            def decoder_func(x):
                return decoder(x)
            
            concrete_func = decoder_func.get_concrete_function(
                tf.TensorSpec(expanded_latent.shape, expanded_latent.dtype)
            )
            
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
            
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(graph_def, name='')
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
                components['decoder'] = flops.total_float_ops
                
    except Exception as e:
        components['decoder'] = f"Error: {e}"
    
    return components

def _get_generic_component_flops(model, batch_size):
    """Generic component FLOPs calculation for unknown models"""
    components = {}
    
    # Try to identify common layer types
    layer_types = {
        'conv': [],
        'dense': [],
        'attention': [],
        'normalization': [],
        'activation': [],
        'other': []
    }
    
    # Categorize layers
    for layer in model.layers:
        layer_name = layer.__class__.__name__.lower()
        if 'conv' in layer_name:
            layer_types['conv'].append(layer)
        elif 'dense' in layer_name or 'linear' in layer_name:
            layer_types['dense'].append(layer)
        elif 'attention' in layer_name:
            layer_types['attention'].append(layer)
        elif 'norm' in layer_name or 'batch' in layer_name:
            layer_types['normalization'].append(layer)
        elif 'activation' in layer_name or 'relu' in layer_name:
            layer_types['activation'].append(layer)
        else:
            layer_types['other'].append(layer)
    
    # Calculate FLOPs for each category
    for category, layers in layer_types.items():
        if layers:
            total_flops = 0
            for layer in layers:
                try:
                    layer_flops = get_layer_flops(model, layer.name, batch_size)
                    if isinstance(layer_flops, int):
                        total_flops += layer_flops
                except:
                    pass
            components[f'{category}_layers'] = total_flops
    
    return components

def main():
    """
    Main function to parse arguments, load model, and calculate FLOPs.
    """
    parser = argparse.ArgumentParser(description='Calculate FLOPs for a Keras model.')
    parser.add_argument('--model_path', type=str, help='Path to the .h5 file or ckpt folder.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for FLOPs calculation.')
    parser.add_argument('--model_name', type=str, default=None, 
                       help='Model class name (will be extracted from config if not provided)')
    parser.add_argument('--experiment_name', type=str, default='siso_1_umi_block_1_ps2_p72',
                       help='Experiment configuration name (will be extracted from config if not provided)')
    parser.add_argument('--components', action='store_true', default=True,
                       help='Calculate component-wise FLOPs breakdown (default: True)')
    parser.add_argument('--no-components', action='store_false', dest='components',
                       help='Skip component-wise FLOPs breakdown')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found at {args.model_path}")
        return

    print(f"Loading model from: {args.model_path}")
    
    # Extract model name and experiment name from config if not provided
    if args.model_name is None or args.experiment_name is None:
        if os.path.isdir(args.model_path):
            config_path = os.path.join(args.model_path, "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                if args.model_name is None:
                    args.model_name = config.get('model_name', 'ReconMAE')
                if args.experiment_name is None:
                    args.experiment_name = config.get('experiment_name', 'siso_1_umi_block_1_ps2_p72')
            else:
                # Default values if config not found
                if args.model_name is None:
                    args.model_name = "ReconMAE"
                if args.experiment_name is None:
                    args.experiment_name = "siso_1_umi_block_1_ps2_p72"
        else:
            # For non-directory paths, use defaults
            if args.model_name is None:
                args.model_name = "ReconMAE"
            if args.experiment_name is None:
                args.experiment_name = "siso_1_umi_block_1_ps2_p72"
    
    print(f"Model name: {args.model_name}")
    print(f"Experiment name: {args.experiment_name}")
    
    try:
        # Suppress experimental feature warnings during model loading
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
        tf.get_logger().setLevel('ERROR')
        
        # Try to load as a complete model first
        try:
            model = tf.keras.models.load_model(args.model_path, compile=False)
            print("Model loaded as complete model")
        except:
            # If that fails, try to load from weights
            print("Loading model from weights...")
            model = load_model_from_weights(args.model_path, args.model_name, args.experiment_name)
            if model is None:
                print("Failed to load model from weights")
                return
            print("Model loaded from weights successfully")
        
        model.summary()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    finally:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        tf.get_logger().setLevel('INFO')

    print(f"\nCalculating FLOPs for batch size: {args.batch_size}")
    flops = get_flops(model, args.batch_size)
    
    print(f"FLOPs: {flops:,}")
    print(f"MFLOPs (Mega): {flops / 1e6:,.2f}")
    print(f"GFLOPs (Giga): {flops / 1e9:,.2f}")
    
    # Calculate component-wise FLOPs
    print(f"\n=== Component-wise FLOPs Analysis ===")
    component_flops = get_component_flops(model, args.batch_size)
    
    total_component_flops = 0
    for component, flops_val in component_flops.items():
        if isinstance(flops_val, int):
            print(f"{component}: {flops_val:,} FLOPs ({flops_val / 1e6:,.2f} MFLOPs)")
            total_component_flops += flops_val
        else:
            print(f"{component}: {flops_val}")
    
    if total_component_flops > 0:
        print(f"\nTotal Component FLOPs: {total_component_flops:,} ({total_component_flops / 1e6:,.2f} MFLOPs)")
        print(f"Component Coverage: {(total_component_flops / flops * 100):.1f}% of total FLOPs")
        
        # Show component breakdown as percentages
        print(f"\n=== Component Breakdown ===")
        for component, flops_val in component_flops.items():
            if isinstance(flops_val, int) and flops_val > 0:
                percentage = (flops_val / total_component_flops) * 100
                print(f"{component}: {percentage:.1f}%")
    
    # Calculate efficiency metrics
    efficiency_metrics = calculate_efficiency_metrics(model, flops, args.batch_size)
    
    print(f"\n=== Efficiency Metrics ===")
    print(f"Total Parameters: {efficiency_metrics['total_params']:,}")
    print(f"Parameter Memory: {efficiency_metrics['param_memory_mb']:.2f} MB")
    print(f"Activation Memory: {efficiency_metrics['activation_memory_mb']:.2f} MB")
    print(f"Total Memory: {efficiency_metrics['memory_mb']:.2f} MB")
    print(f"FLOPs per Parameter: {efficiency_metrics['flops_per_param']:.2f}")
    print(f"FLOPs per MB: {efficiency_metrics['flops_per_mb']:.2f}")
    print(f"Memory Efficiency: {efficiency_metrics['total_params'] / efficiency_metrics['memory_mb']:.2f} params/MB")

if __name__ == '__main__':
    main()
