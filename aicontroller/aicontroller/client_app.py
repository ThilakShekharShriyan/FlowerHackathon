"""biometrichackathon: Flower ClientApp for SOCOFing."""
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from aicontroller.task  import Net, load_data, test as test_fn, train as train_fn

app = ClientApp()

def _cfg(ctx: Context):
    rc = ctx.run_config
    data_root = rc["data-root"]
    print(f"Using data root: {data_root}")
    label_mode = rc.get("label-mode", "binary")
    num_partitions = int(rc["num-partitions"])
    partition_id = int(ctx.node_config.get("partition-id", ctx.node_id % max(1, num_partitions)))
    local_epochs = int(rc["local-epochs"])
    return data_root, label_mode, num_partitions, partition_id, local_epochs

@app.train()
def train(msg: Message, context: Context):
    data_root, label_mode, num_partitions, partition_id, local_epochs = _cfg(context)
    num_classes = 2 if label_mode == "binary" else 4

    model = Net(num_classes=num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainloader, _ = load_data(partition_id, num_partitions, data_root, label_mode)
    train_loss = train_fn(model, trainloader, local_epochs, msg.content["config"]["lr"], device)

    arrays = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    metrics = {"train_loss": train_loss, "num-examples": len(trainloader.dataset)}
    return Message(content=RecordDict({"arrays": ArrayRecord(arrays), "metrics": MetricRecord(metrics)}), reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    data_root, label_mode, num_partitions, partition_id, _ = _cfg(context)
    num_classes = 2 if label_mode == "binary" else 4

    model = Net(num_classes=num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _, valloader = load_data(partition_id, num_partitions, data_root, label_mode)
    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {"eval_loss": eval_loss, "eval_acc": eval_acc, "num-examples": len(valloader.dataset)}
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)
