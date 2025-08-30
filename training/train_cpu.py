import os, time, mlflow, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# baked defaults
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.mlflow:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-demo-cpu")
EPOCHS, BATCH, LR, DIM, CLASSES = 3, 256, 1e-3, 128, 5
ARTIFACT_DIR, MODEL_NAME = "artifacts", "torch-cpu"

device = torch.device("cpu")
print("Using device:", device)

# synthetic dataset
N = 20000
X = torch.randn(N, DIM)
y = torch.randint(0, CLASSES, (N,))
dl = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)

# simple model
model = nn.Sequential(
    nn.Linear(DIM, 64),
    nn.ReLU(),
    nn.Linear(64, CLASSES)
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

os.makedirs(ARTIFACT_DIR, exist_ok=True)
t0 = time.time()

with mlflow.start_run():
    mlflow.log_params({"epochs": EPOCHS, "batch": BATCH, "lr": LR, "dim": DIM, "classes": CLASSES})
    for ep in range(EPOCHS):
        total, correct, running = 0, 0, 0.0
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

            running += loss.item() * xb.size(0)
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        mlflow.log_metrics({"loss": running/total, "acc": correct/total}, step=ep)
        print(f"epoch {ep+1}: loss={running/total:.4f} acc={correct/total:.4f}")

    path = f"{ARTIFACT_DIR}/{MODEL_NAME}.pt"
    torch.save(model.state_dict(), path)
    mlflow.log_artifact(path)
    mlflow.log_metric("train_time_sec", time.time()-t0)

print("Done.")
