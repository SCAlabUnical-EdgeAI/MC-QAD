import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, List, Optional, Tuple
import math
import copy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_cf_pairs, evaluate_model, load_task_data

MODEL_MAP = {
    ("imdb", "bert"):    "textattack/bert-base-uncased-imdb",
    ("imdb", "roberta"): "textattack/roberta-base-imdb",
    ("yelp", "bert"):    "textattack/bert-base-uncased-yelp-polarity",
    ("yelp", "roberta"): "VictorSanh/roberta-base-finetuned-yelp-polarity",
}

class MC_QAD:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.fp_logits = None
        self.quant_size = None


    def prepare(self, model, tokenizer, model_name, cf_train,  device, bitwidth_choices = [4, 8, 16, 32]):
        self.tokenizer = tokenizer
        self.model = copy.deepcopy(model)
        self.quant_size = self.get_quantizable_size()
        self.replace_linear_mixed_precision(self.model, bitwidth_choices)
        self.model.to(device)
        self.fp_logits = self.compute_fp_logits(cf_train, device)
        return self  

    @staticmethod
    def replace_linear_mixed_precision(module, bitwidth_choices) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, MC_QAD.MixedPrecisionLinear(child, bitwidth_choices))
            else:
                MC_QAD.replace_linear_mixed_precision(child, bitwidth_choices)
            
    def compute_fp_logits(self, cf_train, device):
    
        logits_dict: Dict[str, torch.Tensor] = {}
    
        with torch.no_grad():
            for s1, _ in cf_train:
                if s1 not in logits_dict:
                    t   = self.tokenize([s1])
                    t   = {k: v.to(device) for k, v in t.items()}
                    out = self.model(**t).logits.cpu()
                    logits_dict[s1] = out
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return logits_dict
    
    @staticmethod
    def tokenize(texts: List[str]) -> dict:
        return tokenizer(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        )
    
    class CounterfactualFairnessLoss(nn.Module):
        def forward(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(logits_a, logits_b)
    
    
    class QuantConsistencyLoss(nn.Module):    
        def __init__(self, temperature: float = 2.0):
            super().__init__()
            self.T = temperature
    
        def forward(self, fp_logits: torch.Tensor, qat_logits: torch.Tensor) -> torch.Tensor:
            p_fp  = F.softmax(fp_logits  / self.T, dim=-1)
            p_qat = F.log_softmax(qat_logits / self.T, dim=-1)
    
            return F.kl_div(p_qat, p_fp, reduction="batchmean") * (self.T ** 2)


    def compute_expected_memory(self, device) -> torch.Tensor:
        total = torch.tensor(0.0, device=device)
        for module in self.model.modules():
            if isinstance(module, MC_QAD.MixedPrecisionLinear):
                ebw = module.expected_bitwidth
                if ebw is not None:
                    total = total + ebw * module.num_params
        return (1.0 / 8388608) * total

    def get_quantizable_size(self):
        total_bytes = 0
    
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                for param in module.parameters(recurse=False):
                    total_bytes += param.numel() * param.element_size()
    
        total_mb = total_bytes / (1024 ** 2)
        return total_mb


    @torch.no_grad()
    def check_kkt( self, lambda_dual, eps, budget, device):
        
        memory_cost = self.compute_expected_memory(device)
        violation   = (memory_cost - budget) / budget
        lam         = lambda_dual.item()
    
        cond_primal = violation <= 0
        cond_dual   = lam >= 0.0
        cond_cs     = abs(lam * violation) <= eps
    
        converged = cond_primal and cond_dual and cond_cs
    
        info = {
            "violation":  violation,
            "lambda":     lam,
            "cs_residual": abs(lam * violation),
            "cond_primal": cond_primal,
            "cond_cs":     cond_cs,
            "converged":   converged,
        }
        return converged, info

    def _set_reuse_sample(self, flag: bool) -> None:
        for module in self.model.modules():
            if isinstance(module, MC_QAD.MixedPrecisionLinear):
                if flag:
                    orig_forward = MC_QAD.MixedPrecisionLinear.forward.__get__(module, MC_QAD.MixedPrecisionLinear)
                    module.forward = lambda x, _m=module: MC_QAD.MixedPrecisionLinear.forward(_m, x, reuse_sample=True)
                else:
                    if hasattr(module, "forward") and not isinstance(module.__dict__.get("forward"), type(None)):
                        try:
                            del module.forward
                        except AttributeError:
                            pass
    
    def train( self, model_name, cf_train, device, lr_model=1.0e-6, lr_alloc= 1.0e-3, lr_lambda= 1.0e-2, max_epochs = 100,
              cf_batch_size= 16, reduction_perc= 0.7, beta=0.1, tolerance= 1.0e-2):
        if self.model is None:
            raise ValueError("Call prepare() first")

        budget = self.quant_size * (1 - reduction_perc)

        self.model.train()
    
        fairness    = MC_QAD.CounterfactualFairnessLoss()
        consistency = MC_QAD.QuantConsistencyLoss()
    
        alloc_params  = [m.alloc_logits for m in self.model.modules() if isinstance(m, MC_QAD.MixedPrecisionLinear)]
        weight_params = [p for p in self.model.parameters()
                         if not any(p is ap for ap in alloc_params)]
    
        optimizer = torch.optim.AdamW([
            {"params": weight_params, "lr": lr_model},
            {"params": alloc_params,  "lr": lr_alloc},
        ])
    
        lambda_dual = torch.tensor(0.0, device=device)
        epoch = 0
        while epoch < max_epochs:
            random.shuffle(cf_train)
            epoch_loss   = 0.0
            epoch_quant  = 0.0
            epoch_fair   = 0.0
    
            for i in range(0, len(cf_train), cf_batch_size):
                batch   = cf_train[i : i + cf_batch_size]
                s1_list = [s for s, _ in batch]
                s2_list = [s for _, s in batch]
    
                t1 = {k: v.to(device) for k, v in self.tokenize(s1_list).items()}
                t2 = {k: v.to(device) for k, v in self.tokenize(s2_list).items()}
    
                self._set_reuse_sample(flag = False)
                out1    = self.model(**t1)
                logits1 = out1.logits
    
                self._set_reuse_sample(flag =True)
                out2    = self.model(**t2)
                logits2 = out2.logits
                self._set_reuse_sample(flag = False)
    
                teacher_logits = torch.cat([self.fp_logits[s] for s in s1_list], dim=0).to(device)
    
                quant_loss  = consistency(teacher_logits, logits1)
                fair_loss   = fairness(logits1, logits2)
                memory_cost = self.compute_expected_memory(device)
    
                violation  = (memory_cost - budget) / budget
                lagrangian = fair_loss + beta * quant_loss + lambda_dual.detach() * violation
    
                optimizer.zero_grad()
                lagrangian.backward()
                torch.nn.utils.clip_grad_norm_(alloc_params, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(weight_params, max_norm=1.0)
                optimizer.step()
    
                epoch_loss  += lagrangian.item()
                epoch_quant += quant_loss.item()
                epoch_fair  += fair_loss.item()
    
            with torch.no_grad():
                memory_cost  = self.compute_expected_memory(device)
                violation    = (memory_cost - budget) / budget
                lambda_dual  = (lambda_dual + lr_lambda * violation).clamp_(min=0.0)
            
            converged_kkt, kkt_info = self.check_kkt( lambda_dual, tolerance, budget, device)
            
            n_batches = math.ceil(len(cf_train) / cf_batch_size)
            print_string = (
                f"Epoch {epoch+1:>3}"
                f"  │  quant={epoch_quant/n_batches:.4f}  fair={epoch_fair/n_batches:.4f}  loss={epoch_loss:.4f}"
                f"  │  mem={kkt_info['violation']*100:+.2f}%  λ={kkt_info['lambda']:.4f}"
                f"  │  cs={kkt_info['cs_residual']:.2e}"
                f"  │  KKT={'✓' if converged_kkt else '✗'}"
                f"  [P={'✓' if kkt_info['cond_primal'] else '✗'}"
                f"   CS={'✓' if kkt_info['cond_cs'] else '✗'}]"
            )
            print(print_string)
            
            if converged_kkt:
                print(f"[stop] KKT conditions satisfied at epoch {epoch+1}.")
                break
    
            epoch += 1
            
        return self.model

        
    
    class MixedPrecisionLinear(nn.Module):

        def __init__(self, linear: nn.Linear, bitwidth_choices):
            super().__init__()
            self.bitwidth_choices = bitwidth_choices
            self.weight = nn.Parameter(linear.weight.detach().clone())
            self.bias   = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None
            self.num_params = self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
    
            self.alloc_logits = nn.Parameter(torch.empty(len(self.bitwidth_choices)))
            mean = 0.0
            std = 0.02
            nn.init.normal_(self.alloc_logits, mean, std)
            with torch.no_grad():
                self.alloc_logits[-1] += 2*std
    
            self._cached_soft_weights: Optional[torch.Tensor] = None
    
    
        def sample_bitwidth(self) -> Tuple[torch.Tensor, torch.Tensor]:
            y_soft = F.softmax((self.alloc_logits), dim=0)  # [K]
    
            idx    = y_soft.detach().argmax()
            y_hard = torch.zeros_like(y_soft).scatter_(0, idx.unsqueeze(0), 1.0)
    
            y = y_soft + (y_hard - y_soft).detach()
    
            self._cached_soft_weights = y
    
            bit_tensor  = torch.tensor(self.bitwidth_choices, device=y.device, dtype=torch.float32)
            expected_bw = (y * bit_tensor).sum()
    
            return y, expected_bw
    
    
        def forward(self, x: torch.Tensor, reuse_sample: bool = False) -> torch.Tensor:
            
            if reuse_sample:
                assert self._cached_soft_weights is not None, \
                    "reuse_sample=True but sample_bitwidth() was never called this step"
                y = self._cached_soft_weights
            else:
                y, _ = self.sample_bitwidth()
    
            w_quant = sum(
                y[k] * self.ste_fake_quant(self.weight, self.bitwidth_choices[k], per_channel=True)
                for k in range(len(self.bitwidth_choices))
            )
    
            if self.bias is not None:
                b_quant = sum(
                    y[k] * self.ste_fake_quant(self.bias, self.bitwidth_choices[k], per_channel=False)
                    for k in range(len(self.bitwidth_choices))
                )
            else:
                b_quant = None
    
            return F.linear(x, w_quant, b_quant)
    
        @property
        def expected_bitwidth(self) -> Optional[torch.Tensor]:
            if self._cached_soft_weights is None:
                return None
            bit_tensor = torch.tensor(
                self.bitwidth_choices,
                device=self._cached_soft_weights.device,
                dtype=torch.float32,
            )
            return (self._cached_soft_weights * bit_tensor).sum()
    
        @property
        def chosen_bitwidth(self) -> int:
            idx = self.alloc_logits.detach().argmax().item()
            return self.bitwidth_choices[idx]

        @staticmethod
        def ste_fake_quant(v: torch.Tensor, bitwidth: int, per_channel: bool = False) -> torch.Tensor:
            if bitwidth >= 32:
                return v
        
            qmin = -(2 ** (bitwidth - 1))
            qmax =  (2 ** (bitwidth - 1)) - 1
        
            if per_channel and v.dim() >= 2:
                v_flat  = v.detach().view(v.shape[0], -1)
                vmin    = v_flat.min(dim=1).values
                vmax    = v_flat.max(dim=1).values
                scale   = (vmax - vmin) / (qmax - qmin + 1e-8)
                zp      = torch.round(qmin - vmin / (scale + 1e-8))
        
                shape   = [-1] + [1] * (v.dim() - 1)
                scale   = scale.view(shape)
                zp      = zp.view(shape)
            else:
                vmin    = v.detach().min()
                vmax    = v.detach().max()
                scale   = (vmax - vmin) / (qmax - qmin + 1e-8)
                zp      = torch.round(qmin - vmin / (scale + 1e-8))
        
            v_q     = torch.clamp(torch.round(v / (scale + 1e-8) + zp), qmin, qmax)
            v_hat   = (v_q - zp) * scale
        
            return v + (v_hat - v).detach()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MC-QAD Training and Evaluation")

    parser.add_argument("--dataset",  required=True, choices=["imdb", "yelp"])
    parser.add_argument("--model",    required=True, choices=["bert", "roberta"])
    parser.add_argument("--bias_category", required=True, 
                        choices=["age", "disability", "ethnicity", "gender", "religion", "sexual_orientation", "all"])
    
    parser.add_argument("--bitwidth_choices", nargs="+", type=int, default=None)
    parser.add_argument("--lr_model",       type=float, default=1.0e-6)
    parser.add_argument("--lr_alloc",       type=float, default=1.0e-3)
    parser.add_argument("--lr_lambda",      type=float, default=1.0e-2)
    parser.add_argument("--max_epochs",     type=int,   default=100)
    parser.add_argument("--cf_batch_size",  type=int,   default=16)
    parser.add_argument("--reduction_perc", type=float, default=0.7)
    parser.add_argument("--beta",           type=float, default=0.1)
    parser.add_argument("--tolerance",      type=float, default=1.0e-2)

    args = parser.parse_args()

    model_name = MODEL_MAP[(args.dataset, args.model)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = f"{args.dataset}"
    cf_train   = load_cf_pairs(f"{base_path}/calibration_set.json")
    test_data, test_labels = load_task_data(data_name=args.dataset, pct=0.5, subset="test")
    if args.bias_category == "all":
        test_pairs = load_cf_pairs(f"{base_path}/test_set.json")
    else:
        test_pairs = load_cf_pairs(f"{base_path}/{args.bias_category}_test_set.json")

    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    mc_qad = MC_QAD()

    prepare_kwargs = {}
    if args.bitwidth_choices is not None:
        prepare_kwargs["bitwidth_choices"] = args.bitwidth_choices
    mc_qad_model = mc_qad.prepare(model, tokenizer, model_name, cf_train, device, **prepare_kwargs)

    train_kwargs = {
        "lr_model":       args.lr_model,
        "lr_alloc":       args.lr_alloc,
        "lr_lambda":      args.lr_lambda,
        "max_epochs":     args.max_epochs,
        "cf_batch_size":  args.cf_batch_size,
        "reduction_perc": args.reduction_perc,
        "beta":           args.beta,
        "tolerance":      args.tolerance,
    }

    mc_qad_model = mc_qad.train(model_name, cf_train, device, **train_kwargs)

    results = evaluate_model(mc_qad_model, tokenizer, test_data, test_labels, test_pairs, cache_path=f"test_tokens_cache_{args.dataset}_{args.model}.pkl")
    
    print(f"\nResults [{args.dataset.upper()} | {args.model.upper()}]")
    print(f"  Accuracy: {results['accuracy']:.4f}  |  F1: {results['f1']:.4f}")
    print(f"  CFS: {results['cfs']:.4f}  |  Flip: {results['flip']:.4f}  |  W-CFS: {results['w_cfs']:.4f}  |  W-Flip: {results['w_flip']:.4f}\n")
