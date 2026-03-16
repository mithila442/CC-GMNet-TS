import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from dlquantification.gmnet import GMNet
from dlquantification.utils.utils import *
from dlquantification.featureextraction.transformer_fe import TransformerFEModule
import os
from dlquantification.utils.lossfunc import MRAE, MAE, NMD
import json
import argparse
from dlquantification.utils.smartfall_dataset import SmartFallDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_lequa(data_path, parameters_path, train_name, network, network_parameters, dataset, standarize,
                feature_extraction="rff", bag_generator="QLibPriorShiftBagGenerator", cuda_device="cuda:0"):
    with open(os.path.join(parameters_path, f"common_parameters_{dataset}.json")) as f:
        common_params = json.load(f)
    with open(network_parameters) as f:
        network_params = json.load(f)

    if dataset == "SMARTFALL":
        n_classes = 2
        n_channels = 3
        bag_size = common_params["bag_size"]                          # 20 segments per bag
        sequence_length = common_params.get("sequence_length", 64)    # 64 timesteps per segment (2s at 32Hz)

        # MAE training loss — stable gradients at all prevalence levels
        loss = MAE()

        ds_full = SmartFallDataset(
            data_directory="./dataset/smartfallMM",
            sequence_length=sequence_length,
        )

        n_total = len(ds_full)
        n_train = int(0.8 * n_total)

        # Report dataset statistics
        fall_count = sum(1 for i in range(n_total) if ds_full.labels[i] == 1)
        adl_count = n_total - fall_count
        print(f"[SMARTFALL] Segments: {n_total} total, {n_train} train, {n_total - n_train} val")
        print(f"[SMARTFALL] ADL: {adl_count}, Fall: {fall_count}, Fall ratio: {fall_count/n_total:.3f}")
        print(f"[SMARTFALL] Sequence length: {sequence_length} ({sequence_length/32:.1f}s), Bag size: {bag_size}")

        X_all, Y_all = zip(*[(x.to(cuda_device), y.to(cuda_device)) for x, y in ds_full])
        X_tr, Y_tr = torch.stack(X_all[:n_train]), torch.stack(Y_all[:n_train])
        X_v, Y_v = torch.stack(X_all[n_train:]), torch.stack(Y_all[n_train:])

        train_baggen = QLibPriorShiftBagGenerator(
            device=cuda_device,
            method="Dirichlet",
            alphas=np.ones(n_classes) * 1.0,
        )
        val_baggen = QLibPriorShiftBagGenerator(
            device=cuda_device,
            method="Dirichlet",
            alphas=np.ones(n_classes) * 1.0,
            seed=5555,
        )

    else:
        raise ValueError(f"Dataset '{dataset}' not recognized")

    fe = TransformerFEModule(
        input_dim=n_channels,
        d_model=128,
        output_size=256,
        num_layers=2,
        nhead=4,
        dim_feedforward=256,
        dropout=0.3,
    )

    parameters = {**common_params, **network_params}

    parameters.update({
        "n_classes":                n_classes,
        "feature_extraction_module": fe,
        "bag_generator":            train_baggen,
        "val_bag_generator":        val_baggen,
        "quant_loss":               loss,
        "dataset_name":             dataset,
        "device":                   cuda_device,
        "start_lr":                 1e-4,
        "end_lr":                   1e-6,
        "weight_decay":             5e-4,
        "optimizer_class":          torch.optim.AdamW,
        # Label warm-up: classification loss for first 150 epochs
        "use_labels":               True,
        "use_labels_epochs":        150,
        "patience":                 50,
    })

    parameters["save_model_path"] = os.path.join("savedmodels", f"{train_name}.pth")

    # Remove keys that GMNet doesn't accept
    parameters.pop("sequence_length", None)

    model = GMNet(**parameters)
    model.fit(
        dataset=TensorDataset(X_tr, Y_tr),
        val_dataset=TensorDataset(X_v, Y_v),
    )

    return model


def test_lequa(model, data_path, train_name, dataset, standarize):
    print("\n" + "="*60)
    print("Testing the model...")
    print("="*60)

    if dataset == "SMARTFALL":
        path = "./dataset/smartfallMM"
        n_classes = 2
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    with open(os.path.join("experiments/parameters", f"common_parameters_{dataset}.json")) as f:
        common_params = json.load(f)

    bag_size = common_params["bag_size"]
    sequence_length = common_params.get("sequence_length", 64)
    cuda_device = next(model.model.parameters()).device

    ds_full = SmartFallDataset(data_directory=path, sequence_length=sequence_length)
    n_total = len(ds_full)
    n_train = int(0.8 * n_total)
    X, Y = zip(*[ds_full[i] for i in range(n_train, n_total)])

    X = torch.stack(X)
    Y = torch.stack(Y)
    N = X.shape[0]
    print(f"[Test] {N} test segments, seq_len={sequence_length}, shape={X.shape}")

    # Generate test bags with PShift (paper §4.1)
    n_test_bags = common_params["n_bags"][2] if isinstance(common_params["n_bags"], list) else 100
    print(f"Generating {n_test_bags} test bags (bag_size={bag_size})...")

    test_baggen = QLibPriorShiftBagGenerator(
        device=cuda_device,
        method="Dirichlet",
        alphas=np.ones(n_classes) * 1.0,
        seed=9999,
    )

    Y_dev = Y.to(cuda_device)
    bag_indices, bag_prevalences = test_baggen.compute_bags(
        n_bags=n_test_bags, bag_size=bag_size, y=Y_dev
    )

    if not torch.is_tensor(bag_indices):
        bag_indices = torch.tensor(bag_indices, dtype=torch.long)
    bag_indices = bag_indices.to('cpu')
    bag_prevalences = bag_prevalences.to('cpu').float()

    # Build: [n_test_bags, bag_size, seq_len, channels]
    X_bags = torch.stack([X[bag_indices[i]] for i in range(n_test_bags)])
    print(f"Test bag tensor shape: {X_bags.shape}")

    # Predict
    model.model.eval()
    preds_list = []
    batch_sz = 10
    with torch.no_grad():
        for start in range(0, n_test_bags, batch_sz):
            end = min(start + batch_sz, n_test_bags)
            X_batch = X_bags[start:end].to(cuda_device)
            raw = model.model(input=X_batch, return_classification=False)
            if raw.dim() == 3:
                B, M, CK = raw.shape
                C = n_classes
                K = CK // C
                raw = raw.view(B, M, C, K).mean(dim=1).mean(dim=2)
                raw = raw / (raw.sum(dim=1, keepdim=True) + 1e-8)
            preds_list.append(raw.cpu())

    preds = torch.cat(preds_list, dim=0)

    results = pd.DataFrame(preds.numpy())
    os.makedirs("results/", exist_ok=True)
    results.to_csv(os.path.join("results/", f"{train_name}.txt"), index_label="id")
    print(f"Saved predictions.")

    p_true = bag_prevalences
    p_pred = preds

    # Metrics
    eps = 1.0 / (2 * bag_size)
    mrae_fn = MRAE(eps=eps, n_classes=n_classes)
    nmd_fn = NMD()

    mrae = mrae_fn(p_true, p_pred).item()
    mae = torch.mean(torch.abs(p_true - p_pred)).item()
    rmse = torch.sqrt(torch.mean((p_true - p_pred) ** 2)).item()
    nmd = nmd_fn(p_true, p_pred).item()

    print(f"\n{'='*60}")
    print(f"[Test] Results for {dataset}")
    print(f"{'='*60}")
    print(f"Test bags: {n_test_bags}, Bag size: {bag_size}, Seq length: {sequence_length}")
    print(f"{'='*60}")
    print(f"[Test] MRAE:  {mrae:.4f}")
    print(f"[Test] MAE:   {mae:.4f}")
    print(f"[Test] RMSE:  {rmse:.4f}")
    print(f"[Test] NMD:   {nmd:.4f}")
    print(f"{'='*60}\n")

    print(f"Per-class analysis:")
    print(f"{'Class':<10} {'True Mean':<15} {'Pred Mean':<15} {'MAE':<15}")
    print(f"{'-'*55}")
    for c in range(n_classes):
        true_m = p_true[:, c].mean().item()
        pred_m = p_pred[:, c].mean().item()
        c_mae = torch.mean(torch.abs(p_true[:, c] - p_pred[:, c])).item()
        print(f"{c:<10} {true_m:<15.4f} {pred_m:<15.4f} {c_mae:<15.4f}")
    print()

    print("First 10 predictions:")
    print(f"{'Bag':<5} {'True ADL':<10} {'Pred ADL':<10} {'True Fall':<10} {'Pred Fall':<10}")
    for i in range(min(10, n_test_bags)):
        print(f"{i:<5} {p_true[i,0]:.4f}    {p_pred[i,0]:.4f}    {p_true[i,1]:.4f}    {p_pred[i,1]:.4f}")

    return mrae, mae, rmse, p_true, p_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_name", required=True)
    parser.add_argument("-n", "--network", required=True)
    parser.add_argument("-p", "--network_parameters", required=True)
    parser.add_argument("-f", "--feature_extraction", default="transformers")
    parser.add_argument("-b", "--bag_generator", default="QLibPriorShiftBagGenerator")
    parser.add_argument("-s", "--standarize", action='store_true')
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-c", "--cuda_device", required=True)
    args = vars(parser.parse_args())

    print("Arguments:", args)
    args["cuda_device"] = torch.device(args["cuda_device"])

    model = train_lequa("data/", "experiments/parameters", **args)
    mrae, mae, rmse, p_true, p_pred = test_lequa(
        model, "data/", args["train_name"], args["dataset"], args["standarize"]
    )

    os.makedirs("results/", exist_ok=True)
    with open(f"results/{args['train_name']}_metrics.json", 'w') as f:
        json.dump({"mrae": mrae, "mae": mae, "rmse": rmse}, f, indent=4)
    print(f"\nSaved metrics to results/{args['train_name']}_metrics.json")