from cebed.models_with_ssl.denoise_net import DenoiseSingleBranch # get the model class into globals()
from cebed.models_with_ssl.recon_net import ReconMAE
from cebed.models_with_ssl.recon_net_v2 import ReconMAEV2
from cebed.models_with_ssl.recon_net_main_only import ReconMAE_MainOnly
from cebed.models_with_ssl.recon_net_v3 import ReconMAEX
import yaml

MODELS = ["DenoiseSingleBranch", "ReconMAE", "ReconMAEV2", "ReconMAE_MainOnly", "ReconMAEX"]


def get_model_class(model_name):

    if model_name not in globals() or model_name not in MODELS:
        raise NotImplementedError("Model not found: {}".format(model_name))

    return globals()[model_name]


def get_model_hparams(model_name, experiment_name): # FIXME: use experiment_name to load hyperparams
    """
    Returns the model hyperparamters
    :param model_name: The name of the model. Musy be in MODELS
    :param experiment_name: The experiment name
    """

    with open(f"./hyperparams/{model_name}.yaml") as f:
        model_hparams = yaml.safe_load(f)

    if "best" in model_hparams[experiment_name]: 
        model_hparams = model_hparams[experiment_name]["best"]
    else:
        model_hparams = model_hparams[experiment_name]["default"]

    return model_hparams 
