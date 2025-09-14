import argparse
import csv
import json
import os
import socket
import time
import uuid
import random
from datetime import datetime
from itertools import cycle
from typing import Any, Dict, List

import flwr as fl
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import psutil

from data import collate_fn, load_sst2_datasets
from modeling import load_model_and_tokenizer

from peft import get_peft_model_state_dict, set_peft_model_state_dict

LOG_ROOT = "logs"
os.makedirs(LOG_ROOT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def csv_write(path: str, header: List[str], row: Dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def jsonl_write(path: str, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(tokenizer, train_ds, val_ds, batch_size: int):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(tokenizer, b),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(tokenizer, b),
    )
    return train_loader, val_loader


def train_steps(model, loader, optimizer, scheduler=None, steps: int = 100, use_amp: bool = True):
    proc = psutil.Process(os.getpid())
    cpu_count = psutil.cpu_count(logical=True) or 1
    ct0 = proc.cpu_times()
    wall0 = time.time()

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    total_loss = 0.0
    tokens_total = 0
    samples_total = 0

    scaler = torch.amp.GradScaler(enabled=use_amp and DEVICE == "cuda")
    it = cycle(loader)

    for _ in range(max(1, steps)):
        batch = next(it)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        bs = int(batch["labels"].shape[0])
        samples_total += bs
        tokens_total += int(batch["attention_mask"].sum().item())

        optimizer.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            with torch.amp.autocast(device_type="cuda", enabled=True):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += float(loss.item())

    wall1 = time.time()
    ct1 = proc.cpu_times()

    train_ms = (wall1 - wall0) * 1000.0
    cpu_time = (ct1.user - ct0.user) + (ct1.system - ct0.system)
    cpu_util_pct = (100.0 * cpu_time / max(1e-9, (wall1 - wall0) * cpu_count))
    gpu_peak = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else 0
    tok_per_sec = tokens_total / max(1e-9, (train_ms / 1000.0))
    samples_per_sec = samples_total / max(1e-9, (train_ms / 1000.0))
    avg_loss = total_loss / max(1, steps)

    return avg_loss, train_ms, tokens_total, samples_total, tok_per_sec, samples_per_sec, cpu_util_pct, gpu_peak


@torch.no_grad()
def evaluate(model, loader):
    t0 = time.time()
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        logits = out.logits
        loss = out.loss
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += preds.size(0)
        total_loss += float(loss.item())
    eval_ms = (time.time() - t0) * 1000.0
    acc = correct / max(1, total)
    return total_loss / max(1, len(loader)), acc, eval_ms


def state_to_ndarrays(model) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def ndarrays_to_state(model, nds: List[np.ndarray]):
    keys = list(model.state_dict().keys())
    tensors = [torch.tensor(a) for a in nds]
    sd = {k: t for k, t in zip(keys, tensors)}
    try:
        model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        print(f"[Client][WARN] load_state_dict mismatch -> keep local weights. Detail: {e}")


# ---------- LoRA-only comm helpers ----------
def lora_state_to_arrays(model):
    sd = get_peft_model_state_dict(model)
    keys = sorted(sd.keys())
    arrays = [sd[k].detach().cpu().numpy() for k in keys]
    return keys, arrays


def arrays_to_lora_state(model, arrays, keys):
    import torch
    sd = {k: torch.tensor(arr) for k, arr in zip(keys, arrays)}
    set_peft_model_state_dict(model, sd)


class TorchClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        self.client_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.run_id = None
        self.run_dir = None

        # สถานะโมเดล/LoRA/โหมด comm
        self.current_model_name = args.default_model_name
        self.current_lora_sig = (False, 8, 16, 0.05, "auto")  # (use_lora,r,alpha,dropout,target)
        self.comm_lora_only = False

        self.model, self.tok = load_model_and_tokenizer(
            model_name=self.current_model_name,
            num_labels=2,
            use_lora=self.current_lora_sig[0],
            lora_r=self.current_lora_sig[1],
            lora_alpha=self.current_lora_sig[2],
            lora_dropout=self.current_lora_sig[3],
            lora_target=self.current_lora_sig[4],
        )
        self.model.to(DEVICE)

        # dataset lazy
        self.train_ds = None
        self.val_ds = None
        self.train_loader = None
        self.val_loader = None

        self.client_fit_csv = None
        self.client_eval_csv = None
        self.client_events = None

    def _init_run_logs(self):
        rd = os.path.join(LOG_ROOT, str(self.run_id) if self.run_id else "_pending")
        os.makedirs(rd, exist_ok=True)
        self.run_dir = rd
        self.client_fit_csv = os.path.join(rd, f"client_fit_{self.client_id}.csv")
        self.client_eval_csv = os.path.join(rd, f"client_eval_{self.client_id}.csv")
        self.client_events = os.path.join(rd, f"client_events_{self.client_id}.jsonl")

    def _ensure_dataset(self, max_length: int):
        if self.train_ds is None or self.val_ds is None:
            self.train_ds, self.val_ds = load_sst2_datasets(
                self.tok, max_length=max_length, avoid_cache=True, cache_dir=None
            )

    def _reload_model_and_dataset(self, model_name: str, max_length: int, lora_sig, comm_lora_only: bool):
        use_lora, r, alpha, dropout, target = lora_sig
        self.model, self.tok = load_model_and_tokenizer(
            model_name=model_name,
            num_labels=2,
            use_lora=use_lora,
            lora_r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            lora_target=target,
        )
        self.comm_lora_only = bool(comm_lora_only)
        self.model.to(DEVICE)
        self.train_ds = None
        self.val_ds = None
        self._ensure_dataset(max_length=max_length)
        print(f"[Client] Switched model to {model_name} "
              f"(LoRA={use_lora}, r={r}, alpha={alpha}, drop={dropout}, target={target}, "
              f"comm_lora_only={self.comm_lora_only})")

    def _maybe_reload_model(self, model_name: str, max_length: int, lora_sig, comm_lora_only: bool):
        if (model_name != self.current_model_name) or (lora_sig != self.current_lora_sig) or (bool(comm_lora_only) != self.comm_lora_only):
            self.current_model_name = model_name
            self.current_lora_sig = lora_sig
            self._reload_model_and_dataset(model_name, max_length, lora_sig, comm_lora_only)
        else:
            self._ensure_dataset(max_length=max_length)

    # Flower API
    def get_parameters(self, config: Dict[str, Any]):
        if self.comm_lora_only and self.current_lora_sig[0]:
            _, arrays = lora_state_to_arrays(self.model)
            return arrays
        return state_to_ndarrays(self.model)

    def fit(self, parameters, config):
        if self.run_id is None:
            self.run_id = config.get("run_id", "unknown")
            self._init_run_logs()
            print(f"[Client] Using run_id={self.run_id} (logs in {self.run_dir})")

        server_model_name = config.get("model_name", self.current_model_name)
        max_length = int(config.get("max_length", self.args.max_length))
        lora_sig = (
            bool(int(config.get("use_lora", 0))),
            int(config.get("lora_r", 8)),
            int(config.get("lora_alpha", 16)),
            float(config.get("lora_dropout", 0.05)),
            str(config.get("lora_target", "auto")),
        )
        comm_lora_only = int(config.get("comm_lora_only", 0))
        self._maybe_reload_model(server_model_name, max_length, lora_sig, comm_lora_only)

        round_start = time.time()

        recv_bytes = int(sum(arr.nbytes for arr in parameters)) if isinstance(parameters, list) else 0

        # ---- apply received params ----
        t0 = time.time()
        if self.comm_lora_only and self.current_lora_sig[0]:
            keys, _ = lora_state_to_arrays(self.model)  # order (sorted keys)
            arrays_to_lora_state(self.model, parameters, keys)
        else:
            ndarrays_to_state(self.model, parameters)
        deserialize_ms = (time.time() - t0) * 1000.0

        lr = float(config.get("lr", self.args.lr))
        batch_size = int(config.get("batch_size", self.args.batch_size))
        no_amp = bool(int(config.get("no_amp", int(self.args.no_amp))))
        iters_per_round = int(config.get("iters_per_round", self.args.iters_per_round))
        server_round = int(config.get("server_round", -1))
        notes = str(config.get("notes", ""))

        self.train_loader, self.val_loader = build_loaders(self.tok, self.train_ds, self.val_ds, batch_size)

        t_total = max(1, iters_per_round)
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total
        )

        (
            train_loss, train_ms, tokens_total, samples_total,
            tok_per_sec, samples_per_sec, cpu_util_pct, gpu_mem_peak_bytes
        ) = train_steps(
            self.model, self.train_loader, optimizer, scheduler,
            steps=iters_per_round, use_amp=not no_amp,
        )

        # ---- serialize to send back ----
        t1 = time.time()
        if self.comm_lora_only and self.current_lora_sig[0]:
            _, new_params = lora_state_to_arrays(self.model)
        else:
            new_params = state_to_ndarrays(self.model)
        serialize_ms = (time.time() - t1) * 1000.0
        send_bytes = int(sum(arr.nbytes for arr in new_params))

        total_fit_ms = (time.time() - round_start) * 1000.0
        steps = iters_per_round
        device_str = DEVICE

        csv_write(
            self.client_fit_csv,
            header=[
                "ts_iso", "round", "device", "steps",
                "lr", "batch_size", "use_amp",
                "recv_bytes", "send_bytes",
                "deserialize_ms", "train_ms", "serialize_ms", "total_fit_ms",
                "train_loss", "notes",
                "tokens", "samples", "tok_per_sec", "samples_per_sec",
                "cpu_util_pct", "gpu_mem_peak_bytes",
                "model_name", "use_lora", "lora_r", "lora_alpha", "lora_dropout",
                "lora_target", "comm_lora_only",
            ],
            row={
                "ts_iso": datetime.utcnow().isoformat(),
                "round": server_round,
                "device": device_str,
                "steps": steps,
                "lr": lr,
                "batch_size": batch_size,
                "use_amp": int(not no_amp),
                "recv_bytes": recv_bytes,
                "send_bytes": send_bytes,
                "deserialize_ms": round(deserialize_ms, 3),
                "train_ms": round(train_ms, 3),
                "serialize_ms": round(serialize_ms, 3),
                "total_fit_ms": round(total_fit_ms, 3),
                "train_loss": round(train_loss, 6),
                "notes": notes,
                "tokens": tokens_total,
                "samples": samples_total,
                "tok_per_sec": round(tok_per_sec, 3),
                "samples_per_sec": round(samples_per_sec, 3),
                "cpu_util_pct": round(cpu_util_pct, 3),
                "gpu_mem_peak_bytes": gpu_mem_peak_bytes,
                "model_name": self.current_model_name,
                "use_lora": int(self.current_lora_sig[0]),
                "lora_r": self.current_lora_sig[1],
                "lora_alpha": self.current_lora_sig[2],
                "lora_dropout": self.current_lora_sig[3],
                "lora_target": self.current_lora_sig[4],
                "comm_lora_only": int(self.comm_lora_only),
            },
        )

        jsonl_write(self.client_events, {
            "ts": time.time(),
            "event": "fit_done",
            "round": server_round,
            "device": device_str,
            "lr": lr,
            "batch_size": batch_size,
            "use_amp": not no_amp,
            "steps": steps,
            "recv_bytes": recv_bytes,
            "send_bytes": send_bytes,
            "deserialize_ms": deserialize_ms,
            "train_ms": train_ms,
            "serialize_ms": serialize_ms,
            "total_fit_ms": total_fit_ms,
            "train_loss": train_loss,
            "tokens": tokens_total,
            "samples": samples_total,
            "tok_per_sec": tok_per_sec,
            "samples_per_sec": samples_per_sec,
            "cpu_util_pct": cpu_util_pct,
            "gpu_mem_peak_bytes": gpu_mem_peak_bytes,
            "notes": notes,
            "run_id": self.run_id,
            "client_id": self.client_id,
            "model_name": self.current_model_name,
            "use_lora": self.current_lora_sig[0],
            "lora_r": self.current_lora_sig[1],
            "lora_alpha": self.current_lora_sig[2],
            "lora_dropout": self.current_lora_sig[3],
            "lora_target": self.current_lora_sig[4],
            "comm_lora_only": self.comm_lora_only,
        })

        num_examples = len(self.train_loader.dataset)
        print(
            f"[Client] r={server_round} steps={steps} lr={lr} bs={batch_size} "
            f"tok/s={tok_per_sec:.1f} samp/s={samples_per_sec:.2f} "
            f"CPU%~{cpu_util_pct:.1f} GPUpeak={gpu_mem_peak_bytes/1e6:.1f}MB "
            f"recv/send={recv_bytes/1e6:.3f}/{send_bytes/1e6:.3f}MB "
            f"train_ms={train_ms:.1f} total_ms={total_fit_ms:.1f} dev={device_str} "
            f"model={self.current_model_name} LoRA={self.current_lora_sig} comm_lora_only={self.comm_lora_only} "
            f"run={self.run_id}"
        )

        metrics = {
            "recv_bytes": recv_bytes,
            "send_bytes": send_bytes,
            "deserialize_ms": deserialize_ms,
            "train_ms": train_ms,
            "serialize_ms": serialize_ms,
            "total_fit_ms": total_fit_ms,
            "steps": steps,
            "tokens": tokens_total,
            "samples": samples_total,
            "tok_per_sec": tok_per_sec,
            "samples_per_sec": samples_per_sec,
            "cpu_util_pct": cpu_util_pct,
            "gpu_mem_peak_bytes": gpu_mem_peak_bytes,
        }
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config):
        if self.run_id is None:
            self.run_id = config.get("run_id", "unknown")
            self._init_run_logs()

        server_model_name = config.get("model_name", self.current_model_name)
        max_length = int(config.get("max_length", self.args.max_length))
        lora_sig = (
            bool(int(config.get("use_lora", 0))),
            int(config.get("lora_r", 8)),
            int(config.get("lora_alpha", 16)),
            float(config.get("lora_dropout", 0.05)),
            str(config.get("lora_target", "auto")),
        )
        comm_lora_only = int(config.get("comm_lora_only", 0))
        self._maybe_reload_model(server_model_name, max_length, lora_sig, comm_lora_only)

        if self.comm_lora_only and self.current_lora_sig[0]:
            keys, _ = lora_state_to_arrays(self.model)
            arrays_to_lora_state(self.model, parameters, keys)
        else:
            ndarrays_to_state(self.model, parameters)

        eval_bs = int(config.get("eval_batch_size", self.args.batch_size))
        _, self.val_loader = build_loaders(self.tok, self.train_ds, self.val_ds, eval_bs)
        val_loss, val_acc, eval_ms = evaluate(self.model, self.val_loader)
        num_examples = len(self.val_loader.dataset)

        csv_write(
            self.client_eval_csv,
            header=["ts_iso", "round", "eval_bs", "eval_loss", "eval_acc", "eval_ms",
                    "model_name", "use_lora", "lora_r", "lora_alpha", "lora_dropout", "lora_target", "comm_lora_only"],
            row={
                "ts_iso": datetime.utcnow().isoformat(),
                "round": int(config.get("server_round", -1)),
                "eval_bs": eval_bs,
                "eval_loss": round(val_loss, 6),
                "eval_acc": round(val_acc, 6),
                "eval_ms": round(eval_ms, 3),
                "model_name": self.current_model_name,
                "use_lora": int(self.current_lora_sig[0]),
                "lora_r": self.current_lora_sig[1],
                "lora_alpha": self.current_lora_sig[2],
                "lora_dropout": self.current_lora_sig[3],
                "lora_target": self.current_lora_sig[4],
                "comm_lora_only": int(self.comm_lora_only),
            },
        )
        jsonl_write(self.client_events, {
            "ts": time.time(),
            "event": "evaluate_done",
            "round": int(config.get("server_round", -1)),
            "eval_bs": eval_bs,
            "eval_loss": val_loss,
            "eval_acc": val_acc,
            "eval_ms": eval_ms,
            "run_id": self.run_id,
            "client_id": self.client_id,
            "model_name": self.current_model_name,
            "use_lora": self.current_lora_sig[0],
            "lora_r": self.current_lora_sig[1],
            "lora_alpha": self.current_lora_sig[2],
            "lora_dropout": self.current_lora_sig[3],
            "lora_target": self.current_lora_sig[4],
            "comm_lora_only": self.comm_lora_only,
        })
        return float(val_loss), num_examples, {"acc": float(val_acc)}


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--iters_per_round", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--use_cpu", type=str, default="false")
    parser.add_argument("--default_model_name", type=str, default="facebook/opt-125m")
    args = parser.parse_args()

    use_cpu_flag = args.use_cpu.lower() == "true"
    if use_cpu_flag:
        DEVICE = "cpu"
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Client] Using device: {DEVICE}")

    set_seed(args.seed)

    client = TorchClient(args)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
