import argparse
import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from modeling import load_model_and_tokenizer  # ใช้สร้าง initial params

# ================== Run/Log helpers ==================
LOG_ROOT = "logs"
os.makedirs(LOG_ROOT, exist_ok=True)


def next_run_id(log_root: str = LOG_ROOT) -> str:
    os.makedirs(log_root, exist_ok=True)
    ids = []
    for name in os.listdir(log_root):
        if name.isdigit():
            try:
                ids.append(int(name))
            except ValueError:
                pass
    return str(max(ids) + 1 if ids else 1)


def ensure_run_dir(run_id: str) -> str:
    run_dir = os.path.join(LOG_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def csv_write(path: str, header: List[str], row: Dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ================== Config/Controller ==================
@dataclass
class RoundConfig:
    epochs: int = 1
    lr: float = 2e-5
    batch_size: int = 8
    no_amp: bool = False
    max_length: int = 256
    iters_per_round: int = 100
    model_name: str = "facebook/opt-125m"
    # ---- LoRA ----
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: str = "auto"  # หรือ "query,key,value,dense"
    # เฉพาะสื่อสาร LoRA adapters
    comm_lora_only: bool = False
    # --------------
    notes: str = ""


class Controller:
    def __init__(self, default_cfg: RoundConfig, run_id: str):
        self.default = default_cfg
        self.plan: Dict[int, RoundConfig] = {}
        self.run_id = run_id

    def fit_config(self, server_round: int) -> Dict[str, Scalar]:
        cfg = self.plan.get(server_round, self.default)
        return {
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "no_amp": int(cfg.no_amp),
            "max_length": cfg.max_length,
            "iters_per_round": cfg.iters_per_round,
            "model_name": cfg.model_name,
            # LoRA
            "use_lora": int(cfg.use_lora),
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": float(cfg.lora_dropout),
            "lora_target": cfg.lora_target,
            "comm_lora_only": int(cfg.comm_lora_only),
            # misc
            "server_round": server_round,
            "notes": cfg.notes,
            "run_id": self.run_id,
        }

    def eval_config(self, server_round: int) -> Dict[str, Scalar]:
        cfg = self.plan.get(server_round, self.default)
        return {
            "server_round": server_round,
            "eval_batch_size": 32,
            "run_id": self.run_id,
            "model_name": cfg.model_name,
            "use_lora": int(cfg.use_lora),
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": float(cfg.lora_dropout),
            "lora_target": cfg.lora_target,
            "comm_lora_only": int(cfg.comm_lora_only),
        }


# ================== Strategy (FedAvg + logs) ==================
class MyStrategy(FedAvg):
    def __init__(self, *args, controller: Controller, run_dir: str, **kwargs):
        super().__init__(
            *args,
            on_fit_config_fn=self._fit_cfg_fn(controller),
            on_evaluate_config_fn=self._eval_cfg_fn(controller),
            **kwargs,
        )
        self.run_dir = run_dir
        self.server_csv = os.path.join(self.run_dir, "server_rounds.csv")
        self.controller = controller

        self._initial_parameters: Optional[Parameters] = None
        self._init_params_from_model_name()

    def _init_params_from_model_name(self):
        cfg = self.controller.default
        model, _ = load_model_and_tokenizer(
            model_name=cfg.model_name,
            num_labels=2,
            use_lora=cfg.use_lora,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            lora_target=cfg.lora_target,
        )
        if cfg.comm_lora_only and cfg.use_lora:
            # init เฉพาะ LoRA (ค่าเริ่มต้นของ LoRA เป็นศูนย์/สุ่มเล็ก ๆ ตาม PEFT)
            from peft import get_peft_model_state_dict
            sd = get_peft_model_state_dict(model)
            keys = sorted(sd.keys())
            nds = [sd[k].detach().cpu().numpy() for k in keys]
            self._initial_parameters = ndarrays_to_parameters(nds)
            print(f"[Server] Prepared initial LoRA-only params for '{cfg.model_name}' (keys={len(keys)})")
        else:
            # เต็มโมเดล
            nds = [v.detach().cpu().numpy() for _, v in model.state_dict().items()]
            self._initial_parameters = ndarrays_to_parameters(nds)
            print(f"[Server] Prepared initial FULL params for '{cfg.model_name}' "
                  f"(comm_lora_only={cfg.comm_lora_only}, use_lora={cfg.use_lora})")

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self._initial_parameters

    @staticmethod
    def _fit_cfg_fn(controller: Controller) -> Callable[[int], Dict[str, Scalar]]:
        def fn(server_round: int) -> Dict[str, Scalar]:
            cfg = controller.fit_config(server_round)
            print(f"[Server] Round {server_round} fit_config -> {cfg}")
            return cfg
        return fn

    @staticmethod
    def _eval_cfg_fn(controller: Controller) -> Callable[[int], Dict[str, Scalar]]:
        def fn(server_round: int) -> Dict[str, Scalar]:
            cfg = controller.eval_config(server_round)
            print(f"[Server] Round {server_round} eval_config -> {cfg}")
            return cfg
        return fn

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        t0 = time.time()
        if not results:
            return None, {}

        weights_list, sizes = [], []

        # รวม metrics client
        sum_recv_bytes = 0
        sum_send_bytes = 0
        sum_deser_ms = 0.0
        sum_ser_ms = 0.0
        sum_train_ms = 0.0
        sum_total_fit_ms = 0.0
        sum_steps = 0
        sum_tokens = 0
        sum_samples = 0
        max_gpu_mem_peak = 0
        cpu_util_pct_sum = 0.0
        clients = 0

        for _, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            weights_list.append(nds)
            sizes.append(fit_res.num_examples)
            m = fit_res.metrics or {}
            sum_recv_bytes += int(m.get("recv_bytes", 0))
            sum_send_bytes += int(m.get("send_bytes", 0))
            sum_deser_ms += float(m.get("deserialize_ms", 0.0))
            sum_ser_ms += float(m.get("serialize_ms", 0.0))
            sum_train_ms += float(m.get("train_ms", 0.0))
            sum_total_fit_ms += float(m.get("total_fit_ms", 0.0))
            sum_steps += int(m.get("steps", 0))
            sum_tokens += int(m.get("tokens", 0))
            sum_samples += int(m.get("samples", 0))
            max_gpu_mem_peak = max(max_gpu_mem_peak, int(m.get("gpu_mem_peak_bytes", 0)))
            cpu_util_pct_sum += float(m.get("cpu_util_pct", 0.0))
            clients += 1

        # FedAvg weighted by num_examples
        n_layers = len(weights_list[0])
        assert all(len(w) == n_layers for w in weights_list), "Layer count mismatch"
        total_examples = float(sum(sizes))

        avg_weights = []
        for li in range(n_layers):
            stacked = np.stack([w[li] * sizes[i] for i, w in enumerate(weights_list)], axis=0)
            avg = stacked.sum(axis=0) / total_examples
            avg_weights.append(avg.astype(weights_list[0][li].dtype, copy=False))

        aggregated_params = ndarrays_to_parameters(avg_weights)

        agg_ms = (time.time() - t0) * 1000.0
        eff_tok_per_sec = (sum_tokens / (sum_train_ms / 1000.0)) if sum_train_ms > 0 else 0.0
        eff_samples_per_sec = (sum_samples / (sum_train_ms / 1000.0)) if sum_train_ms > 0 else 0.0
        cpu_util_pct_avg = (cpu_util_pct_sum / clients) if clients > 0 else 0.0

        csv_write(
            self.server_csv,
            header=[
                "ts_iso", "round", "num_clients", "total_examples",
                "steps_sum", "tokens_sum", "samples_sum",
                "eff_tok_per_sec", "eff_samples_per_sec", "cpu_util_pct_avg",
                "gpu_mem_peak_max_bytes", "recv_bytes_sum", "send_bytes_sum",
                "deserialize_ms_sum", "serialize_ms_sum", "train_ms_sum",
                "total_fit_ms_sum", "aggregate_ms",
            ],
            row={
                "ts_iso": datetime.utcnow().isoformat(),
                "round": server_round,
                "num_clients": clients,
                "total_examples": int(total_examples),
                "steps_sum": sum_steps,
                "tokens_sum": sum_tokens,
                "samples_sum": sum_samples,
                "eff_tok_per_sec": round(eff_tok_per_sec, 3),
                "eff_samples_per_sec": round(eff_samples_per_sec, 3),
                "cpu_util_pct_avg": round(cpu_util_pct_avg, 3),
                "gpu_mem_peak_max_bytes": max_gpu_mem_peak,
                "recv_bytes_sum": sum_recv_bytes,
                "send_bytes_sum": sum_send_bytes,
                "deserialize_ms_sum": round(sum_deser_ms, 3),
                "serialize_ms_sum": round(sum_ser_ms, 3),
                "train_ms_sum": round(sum_train_ms, 3),
                "total_fit_ms_sum": round(sum_total_fit_ms, 3),
                "aggregate_ms": round(agg_ms, 3),
            },
        )
        print(
            f"[Server] R{server_round} agg: clients={clients} ex={int(total_examples)} "
            f"steps={sum_steps} tok/s={eff_tok_per_sec:.1f} samp/s={eff_samples_per_sec:.2f} "
            f"CPU%~{cpu_util_pct_avg:.1f} GPUpeak={max_gpu_mem_peak/1e6:.1f}MB "
            f"recv/send={sum_recv_bytes/1e6:.3f}/{sum_send_bytes/1e6:.3f}MB "
            f"train_ms_sum={sum_train_ms:.1f} agg_ms={agg_ms:.1f}"
        )
        return aggregated_params, {"num_clients": clients, "total_examples": int(total_examples), "round": server_round}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        t0 = time.time()
        if not results:
            return None, {}
        total_examples, loss_sum, acc_sum = 0, 0.0, 0.0
        for _, ev in results:
            n = ev.num_examples
            total_examples += n
            loss_sum += float(ev.loss) * n
            if "acc" in ev.metrics:
                acc_sum += float(ev.metrics["acc"]) * n
        agg_loss = loss_sum / max(1, total_examples)
        agg_acc = acc_sum / max(1, total_examples)
        eval_ms = (time.time() - t0) * 1000.0
        print(f"[Server] R{server_round} eval: loss={agg_loss:.4f} acc={agg_acc:.4f} ({eval_ms:.1f} ms)")
        return agg_loss, {"acc": float(agg_acc), "round": server_round}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--iters_per_round", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_amp", type=str, default="false")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    # LoRA options
    parser.add_argument("--use_lora", type=str, default="false")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target", type=str, default="auto")
    parser.add_argument("--comm_lora_only", type=str, default="false")
    args = parser.parse_args()

    run_id = next_run_id()
    run_dir = ensure_run_dir(run_id)
    print(f"[Server] Starting run_id={run_id} (logs in {run_dir})")

    default_cfg = RoundConfig(
        epochs=1,
        lr=args.lr,
        batch_size=args.batch_size,
        no_amp=args.no_amp.lower() == "true",
        max_length=args.max_length,
        iters_per_round=args.iters_per_round,
        model_name=args.model_name,
        use_lora=(args.use_lora.lower() == "true"),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
        comm_lora_only=(args.comm_lora_only.lower() == "true"),
        notes=f"default: iters_per_round={args.iters_per_round}",
    )
    controller = Controller(default_cfg=default_cfg, run_id=run_id)

    strategy = MyStrategy(
        controller=controller,
        run_dir=run_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )


if __name__ == "__main__":
    main()
