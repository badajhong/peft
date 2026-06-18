import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import(
    TRANSFORMERS_MODELS_TO_CARA_TARGET_MODULES_MAPPING, # need to update
)
from peft.utils.other import get_pattern_key

from .config import CaraConfig
from .layer import CaraLayer, CaraLinear

class CaraModel(BaseTuner):
    """
    Creates Cayley Rotational Adaptation (CaRA) model from a pretrained model.
    """

    prefix: str = "cara_"
    tuner_layer_cls = CaraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_CARA_TARGET_MODULES_MAPPING

    def _check_new_adapter_config(self, config: CaraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config)
        if config.r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {config.r}")     
    
    def _create_and_replace(
            self,
            cara_config,
            adapter_name,
            target,
            target_name,
            parent, 
            current_key,
            **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        
        kwargs = {
            "r": cara_config.r,
            "noise_alpha": cara_config.noise_alpha,
            "noise_step_interval": cara_config.noise_step_interval,
        }

        if isinstance(target, CaraLinear):
            target.update_layer(
                adapter_name,
                r=kwargs["r"],
                noise_alpha=kwargs["noise_alpha"],
                noise_step_interval=kwargs["noise_step_interval"],
                init_weights=False,
            )
        else:
            new_module = self._create_new_module(cara_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
    
    @staticmethod
    def _create_new_module(cara_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = CaraLinear(target, adapter_name, **kwargs)
        else:
            raise ValueError(f"Unsupported layer type {type(target_base_layer)} for CaRA.")
        
        return new_module