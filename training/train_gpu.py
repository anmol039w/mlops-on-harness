import mlflow, os, time, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# ---- baked defaults ----
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.mlflow:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-demo-gpu")
EPOCHS, BATCH, LR, DIM, CLASSES = 5, 512, 1e-3, 128, 10
ARTIFACT_DIR, MODEL_NAME = "artifacts", "torch-synth"

assert torch.cuda.is_available(), "CUDA not available"
device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

# synthetic data
N = 60000
X = torch.randn(N, DIM); y = torch.randint(0, CLASSES, (N,))
dl = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)

model = nn.Sequential(nn.Linear(DIM,256), nn.ReLU(), nn.Linear(256,CLASSES)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

os.makedirs(ARTIFACT_DIR, exist_ok=True)
t0 = time.time()
with mlflow.start_run():
    mlflow.log_params({"epochs":EPOCHS,"batch":BATCH,"lr":LR,"dim":DIM,"classes":CLASSES})
    for epoch in range(EPOCHS):
        total=0; correct=0; running=0.0
        for xb,yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb); loss.backward(); opt.step()
            running += loss.item()*xb.size(0)
            pred = out.argmax(1); correct += (pred==yb).sum().item(); total += yb.size(0)
        mlflow.log_metrics({"loss": running/total, "acc": correct/total}, step=epoch)
        print(f"epoch {epoch+1}: loss={running/total:.4f} acc={correct/total:.4f}")
    path = f"{ARTIFACT_DIR}/{MODEL_NAME}.pt"
    torch.save(model.state_dict(), path)
    mlflow.log_artifact(path)
    mlflow.log_metric("train_time_sec", time.time()-t0)
print("Done.")
