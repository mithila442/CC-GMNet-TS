"""GMNet implementation — fixed version.

Changes from original:
  1. Removed geotorch dependency (diagonal covariance in GMLayer handles PD)
  2. Fixed Unflatten ordering (batch-first, not bag-first)
  3. Added bag-level aggregation (mean over bag dim) so output is [B, C*K]
  4. Removed full-covariance init code (replaced by diagonal log_var)
"""

import torch
from dlquantification.quantmodule.other.GMLayer import GMLayer
from dlquantification.utils.utils import BaseBagGenerator
import torch.nn.functional as F
import numpy as np

from dlquantification.dlquantification import DLQuantification
from dlquantification.utils.ckareg import CKARegularization


class Power(torch.nn.Module):
    def __init__(self, exponent):
        super(Power, self).__init__()
        self.exponent = exponent

    def forward(self, x):
        return torch.pow(x, self.exponent)


class GMNet_Module(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size_fe,
        dropout_fe,
        bag_size,
        device,
        num_gaussians,
        n_gm_layers,
        gaussian_dimensions,
        cka_regularization,
        n_classes
    ):
        super(GMNet_Module, self).__init__()
        if len(num_gaussians) != n_gm_layers:
            raise ValueError("num_gaussians should be a tuple of the same size as n_gm_layers")
        if len(gaussian_dimensions) != n_gm_layers:
            raise ValueError("gaussian_dimensions should be a tuple of the same size as n_gm_layers")

        if isinstance(hidden_size_fe, int):
            self.hidden_size_fe = [hidden_size_fe]
        else:
            self.hidden_size_fe = hidden_size_fe
        self.n_gm_layers = n_gm_layers
        self.num_gaussians = num_gaussians
        self.gaussian_dimensions = gaussian_dimensions
        self.cka_regularization = cka_regularization
        self.n_classes = n_classes
        self.bag_size = bag_size

        if n_gm_layers == 1:
            self.output_size = num_gaussians[0]
        else:
            self.output_size = 0
            for i in range(n_gm_layers):
                self.output_size += self.num_gaussians[i]

        if cka_regularization != 0 or cka_regularization != 'view':
            self.quant_regularization = True
            self.cka = CKARegularization()
            self.activations = [None] * self.n_gm_layers

            def getActivation(i):
                def hook(model, input, output):
                    self.activations[i] = output.view(-1, self.gaussian_dimensions[i])
                return hook
        else:
            self.quant_regularization = False

        self.gm_modules = torch.nn.ModuleList()
        for i in range(n_gm_layers):
            module = torch.nn.Sequential()
            if gaussian_dimensions[i] is not None:
                previous_size = input_size
                if self.hidden_size_fe is not None:
                    for j, layer_size in enumerate(self.hidden_size_fe):
                        module.add_module(f"hidden_gm_{i}_{j}", torch.nn.Linear(previous_size, layer_size))
                        module.add_module(f"fe_leaky_relu_{i}_{j}", torch.nn.LeakyReLU())
                        module.add_module(f"fe_dropout_{i}_{j}", torch.nn.Dropout(dropout_fe))
                        previous_size = layer_size
                module.add_module(f"linear_gm_{i}", torch.nn.Linear(previous_size, gaussian_dimensions[i]))
            module.add_module(f"sigmoid_gm_{i}", torch.nn.Sigmoid())

            input_dimensions = gaussian_dimensions[i] if gaussian_dimensions[i] is not None else input_size

            if cka_regularization != 0 or cka_regularization == 'view':
                index = list(dict(module.named_children()).keys()).index(f"linear_gm_{i}")
                module[index].register_forward_hook(getActivation(i))

            # === Class-specific Gaussian modeling ===
            total_gaussians = num_gaussians[i]
            gaussians_per_class = total_gaussians // self.n_classes

            gm_layer = GMLayer(
                n_features=input_dimensions,
                num_gaussians=gaussians_per_class,
                num_classes=self.n_classes,
                class_conditioned=True,
                device=device,
            )
            # No geotorch needed — diagonal covariance is always positive via exp()

            module.add_module(f"gm_layer_{i}", gm_layer)
            # NOTE: We handle BatchNorm and bag aggregation manually in forward()
            # to maintain correct batch-first ordering.
            self.gm_modules.append(module)

        # BatchNorm over the C*K likelihood features
        total_features = sum(self.num_gaussians)
        self.bn = torch.nn.BatchNorm1d(total_features)

    def forward(self, input):
        """
        input : [B, bag_size, feat_dim]
        returns : [B, C*K]   (bag-level representation, averaged over bag)
        """
        gm_outputs = []
        for gm_module in self.gm_modules:
            out = gm_module(input)           # [B, bag_size, C*K_i]
            gm_outputs.append(out)

        # Concatenate across GM layers: [B, bag_size, total_C*K]
        cat = torch.cat(gm_outputs, dim=-1)  # [B, bag_size, C*K]

        # Apply BatchNorm (needs [N, features])
        B, M, F = cat.shape
        cat_flat = cat.reshape(B * M, F)      # [B*bag_size, C*K]
        cat_bn   = self.bn(cat_flat)          # [B*bag_size, C*K]
        cat      = cat_bn.reshape(B, M, F)    # [B, bag_size, C*K]

        # === Bag-level aggregation (§3.3, Eq. r_i) ===
        bag_repr = cat.mean(dim=1)            # [B, C*K]

        return bag_repr

    def compute_regularization(self):
        return self.cka_regularization == "view" or self.cka_regularization != 0

    def apply_regularization(self):
        return self.cka_regularization != 0 and self.cka_regularization != "view"

    def get_regularization_term(self):
        if self.cka_regularization != 'view':
            return self.cka_regularization * self.cka.feature_space_linear_cka(self.activations)
        else:
            return self.cka.feature_space_linear_cka(self.activations)

    def get_regularization_multiplier(self):
        if self.cka_regularization != 'view':
            return self.cka_regularization
        else:
            return 1

    def get_parameters_to_log(self):
        return {
            "n_gm_layers": self.n_gm_layers,
            "num_gaussians": self.num_gaussians,
            "cka_regularization": self.cka_regularization,
            "hidden_size_fe": self.hidden_size_fe,
            "gaussian_dimensions": self.gaussian_dimensions,
        }


class GMNet(DLQuantification):
    """
    GMNet quantifier — class-conditioned Gaussian mixture with bag-level aggregation.
    """

    def __init__(
        self,
        train_epochs,
        test_epochs,
        n_classes,
        start_lr,
        end_lr,
        n_bags,
        bag_size,
        random_seed,
        linear_sizes,
        feature_extraction_module,
        n_gm_layers,
        num_gaussians,
        gaussian_dimensions,
        batch_size: int,
        bag_generator: BaseBagGenerator,
        hidden_size_fe=None,
        dropout_fe=0,
        gradient_accumulation: int = 1,
        val_bag_generator: BaseBagGenerator = None,
        test_bag_generator: BaseBagGenerator = None,
        optimizer_class=torch.optim.AdamW,
        cka_regularization=0,
        dropout: float = 0,
        weight_decay: float = 0,
        lr_factor=0.1,
        val_split=0,
        quant_loss=torch.nn.L1Loss(),
        quant_loss_val=None,
        epsilon=0,
        output_function="softmax",
        metadata_size=None,
        use_labels: bool = False,
        use_labels_epochs=None,
        residual_connection=False,
        batch_size_fe=None,
        device=torch.device("cpu"),
        use_multiple_devices=False,
        patience: int = 20,
        num_workers: int = 0,
        use_fp16: bool = False,
        callback_epoch=None,
        save_model_path=None,
        save_checkpoint_epochs=None,
        verbose=0,
        tensorboard_dir=None,
        use_wandb: bool = False,
        wandb_experiment_name: str = None,
        log_samples=False,
        dataset_name="",
    ):
        torch.manual_seed(random_seed)

        quantmodule = GMNet_Module(
            input_size=feature_extraction_module.output_size,
            bag_size=bag_size,
            device=device,
            num_gaussians=num_gaussians,
            n_gm_layers=n_gm_layers,
            gaussian_dimensions=gaussian_dimensions,
            hidden_size_fe=hidden_size_fe,
            dropout_fe=dropout_fe,
            cka_regularization=cka_regularization,
            n_classes=n_classes,
        )

        super().__init__(
            train_epochs=train_epochs,
            test_epochs=test_epochs,
            n_classes=n_classes,
            start_lr=start_lr,
            end_lr=end_lr,
            n_bags=n_bags,
            bag_size=bag_size,
            random_seed=random_seed,
            batch_size=batch_size,
            quantmodule=quantmodule,
            bag_generator=bag_generator,
            val_bag_generator=val_bag_generator,
            test_bag_generator=test_bag_generator,
            optimizer_class=optimizer_class,
            weight_decay=weight_decay,
            lr_factor=lr_factor,
            val_split=val_split,
            quant_loss=quant_loss,
            quant_loss_val=quant_loss_val,
            batch_size_fe=batch_size_fe,
            gradient_accumulation=gradient_accumulation,
            feature_extraction_module=feature_extraction_module,
            linear_sizes=linear_sizes,
            dropout=dropout,
            epsilon=epsilon,
            output_function=output_function,
            metadata_size=metadata_size,
            use_labels=use_labels,
            use_labels_epochs=use_labels_epochs,
            residual_connection=residual_connection,
            device=device,
            use_multiple_devices=use_multiple_devices,
            patience=patience,
            num_workers=num_workers,
            use_fp16=use_fp16,
            callback_epoch=callback_epoch,
            save_model_path=save_model_path,
            save_checkpoint_epochs=save_checkpoint_epochs,
            verbose=verbose,
            tensorboard_dir=tensorboard_dir,
            use_wandb=use_wandb,
            wandb_experiment_name=wandb_experiment_name,
            log_samples=log_samples,
            dataset_name=dataset_name,
        )