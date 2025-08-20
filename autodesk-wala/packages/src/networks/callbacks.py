import logging
from pytorch_lightning.callbacks import Callback
from timm.utils.model import get_state_dict, unwrap_model
from timm.utils.model_ema import ModelEmaV2


# obtained from https://github.com/benihime91/gale/blob/master/gale/collections/callbacks/ema.py#L20
class EMACallback(Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.

    If `use_ema_weights`, then the ema parameters of the network is set after training end.
    """

    def __init__(self, decay=0.9999, use_ema_weights: bool = False):
        self.decay = decay
        self.use_ema_weights = use_ema_weights
        self.ema = None
        self.reload_weight = None
        self.collected_params = None

    def on_fit_start(self, trainer, pl_module):
        "Initialize `ModelEmaV2` from timm to keep a copy of the moving average of the weights"
        self.ema = ModelEmaV2(pl_module, decay=self.decay, device=None)
        print("Entering the training steps")
        logging.info("Entering the training steps")
        print("EMA reloaded weight exists : {}".format(self.reload_weight is not None))
        logging.info(
            "EMA reloaded weight exists : {}".format(self.reload_weight is not None)
        )
        if self.reload_weight is not None:
            print("Reloaded the weight for EMA models")
            logging.info("Reloaded the weight for EMA models")
            self.ema.load_state_dict(self.reload_weight)
            self.reload_weight = None

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.ema = ModelEmaV2(pl_module, decay=self.decay, device=None)
        print("Entering the testing steps")
        logging.info("Entering the testing steps")
        print("EMA reloaded weight exists : {}".format(self.reload_weight is not None))
        logging.info(
            "EMA reloaded weight exists : {}".format(self.reload_weight is not None)
        )
        if self.reload_weight is not None:
            print("Reloaded the weight for EMA models")
            logging.info("Reloaded the weight for EMA models")
            self.ema.load_state_dict(self.reload_weight)
            self.reload_weight = None

    def reload_weight_for_pl_module(self, pl_module):
        self.ema = ModelEmaV2(pl_module, decay=self.decay, device=None)
        print("Entering the testing steps")
        logging.info("Entering the testing steps")
        print("EMA reloaded weight exists : {}".format(self.reload_weight is not None))
        logging.info(
            "EMA reloaded weight exists : {}".format(self.reload_weight is not None)
        )
        if self.reload_weight is not None:
            print("Reloaded the weight for EMA models")
            logging.info("Reloaded the weight for EMA models")
            self.ema.load_state_dict(self.reload_weight)
            self.reload_weight = None

    def copy_to_pl_module(self, pl_module):
        if self.reload_weight is not None:
            print("Reloaded the weight for EMA models")
            logging.info("Reloaded the weight for EMA models")
            self.ema.load_state_dict(self.reload_weight)
            self.reload_weight = None
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())
        logging.info("Resume from EMA for testing...")

    def on_test_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if self.reload_weight is not None:
            print("Reloaded the weight for EMA models")
            logging.info("Reloaded the weight for EMA models")
            self.ema.load_state_dict(self.reload_weight)
            self.reload_weight = None
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())
        logging.info("Resume from EMA for testing...")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        "Update the stored parameters using a moving average"
        # Update currently maintained parameters.
        self.ema.update(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        logging.info("do validation using the stored parameters")
        if self.use_ema_weights and self.ema:
            # save original parameters before replacing with EMA version
            self.store(pl_module.parameters())

            # update the LightningModule with the EMA weights
            # ~ Copy EMA parameters to LightningModule
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        logging.info("Restore original parameters to resume training later")
        if self.use_ema_weights:
            self.restore(pl_module.parameters())

    def on_train_end(self, trainer, pl_module):
        # update the LightningModule with the EMA weights
        if self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())
            msg = "Model weights replaced with the EMA version."
            logging.info(msg)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        print("Loaded the EMA model weights")
        logging.info("Saved the EMA model weights")
        checkpoint["state_dict_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        print("Loaded the EMA model weights")
        logging.info("Loaded the EMA model weights")
        self.reload_weight = checkpoint["state_dict_ema"]

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        if self.collected_params:
            for c_param, param in zip(self.collected_params, parameters):
                param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)
