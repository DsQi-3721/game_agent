"""PPO for the legal-action letter policy with a detached, training-only value head."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import torch

from arena.s1.scorer import _chat_ids_batch, _label_ids, label_logits
from arena.s1.prompts import render


def distribution(logits, sizes, temperature=1.0):
    sizes = torch.as_tensor(sizes, device=logits.device)
    legal = torch.arange(logits.shape[-1], device=logits.device)[None, :] < sizes[:, None]
    return torch.distributions.Categorical(logits=(logits.float()/temperature).masked_fill(~legal, -torch.inf))


def advantages(rewards, values, terminated, bootstrap=0.0, gamma=1.0, lam=.97):
    """One contiguous episode or truncated segment; terminal transitions never bootstrap."""
    result = [0.0] * len(rewards)
    carry, next_value = 0.0, bootstrap
    for i in reversed(range(len(rewards))):
        continuation = float(not terminated[i])
        delta = rewards[i] + gamma * next_value * continuation - values[i]
        carry = delta + gamma * lam * continuation * carry
        result[i] = carry
        next_value = values[i]
    return result, [a+v for a,v in zip(result,values)]


def clipped_objective(logp, old_logp, advantage, epsilon):
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    objective = torch.minimum(ratio*advantage, ratio.clamp(1-epsilon,1+epsilon)*advantage)
    return -objective, (ratio-1-log_ratio).detach(), ((ratio-1).abs() > epsilon).float().detach()


class ActionPolicy(torch.nn.Module):
    def __init__(self, model, tokenizer, max_tokens, temperature=1.0, batch_size=2, padding_multiple=1024):
        super().__init__()
        self.actor = model
        self.tokenizer = tokenizer
        self.label_ids = _label_ids(tokenizer)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.max_tokens, self.temperature = max_tokens, temperature
        self.batch_size, self.padding_multiple = batch_size, padding_multiple
        head = model.get_output_embeddings()
        self.value = torch.nn.Linear(head.weight.shape[1], 1, device=head.weight.device, dtype=torch.float32)
        torch.nn.init.zeros_(self.value.weight); torch.nn.init.zeros_(self.value.bias)

    @classmethod
    def load(cls, base, adapter, device, max_tokens=16384, trainable=True, temperature=1.0,
             batch_size=2, padding_multiple=1024):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        dtype = torch.bfloat16 if str(device).startswith('cuda') else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
        model.config.use_cache = False
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
        if trainable:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.enable_input_require_grads()
        result = cls(model, AutoTokenizer.from_pretrained(base), max_tokens, temperature,batch_size,padding_multiple)
        state = Path(adapter) / 'value_head.pt'
        if state.exists(): result.value.load_state_dict(torch.load(state, map_location=device, weights_only=True))
        result.eval()
        return result

    def encode(self, examples):
        prompts = [render(e['state'], e['question']) for e in examples]
        ids = _chat_ids_batch(self.tokenizer, [p[0] for p in prompts])
        for ex, (_, keys), seq in zip(examples, prompts, ids):
            if keys != ex['keys'] or not keys:
                raise ValueError('Recorded action keys differ from the rendered options')
            if len(seq) > self.max_tokens:
                raise ValueError(f'Complete policy input requires {len(seq)} tokens; limit={self.max_tokens}')
        return ids

    def forward(self, ids, sizes):
        if len(ids) > self.batch_size or len({self.bucket(s) for s in ids}) != 1:
            raise ValueError('PPO microbatches must share the rollout batch shape and length bucket')
        logits, hidden = label_logits(self.actor, ids, self.label_ids, self.pad_id, return_hidden=True,
            pad_to_length=self.bucket(ids[0]),pad_to_batch=self.batch_size)
        return distribution(logits, sizes, self.temperature), self.value(hidden.detach()).squeeze(-1)

    def bucket(self, ids):
        return math.ceil(len(ids)/self.padding_multiple)*self.padding_multiple

    @torch.no_grad()
    def predict(self, examples):
        self.eval()
        ids = self.encode(examples)
        result = [None]*len(examples)
        for width in sorted({self.bucket(s) for s in ids}):
            indices = [i for i,s in enumerate(ids) if self.bucket(s)==width]
            for start in range(0,len(indices),self.batch_size):
                positions = indices[start:start+self.batch_size]
                dist, values = self([ids[i] for i in positions], [len(examples[i]['keys']) for i in positions])
                for i,p,v in zip(positions,dist.probs.cpu(),values.cpu()):
                    result[i] = {'ids':ids[i],'probs':p[:len(examples[i]['keys'])].tolist(),'value':v.item()}
        return result

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        self.actor.save_pretrained(directory)
        torch.save(self.value.state_dict(), directory / 'value_head.pt')


class PPOTrainer:
    def __init__(self, policy, config, seed):
        self.policy, self.cfg, self.rng = policy, config, random.Random(seed)
        self.actor_params = [p for p in policy.actor.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW([
            {'params': self.actor_params, 'lr': config['actor_lr']},
            {'params': policy.value.parameters(), 'lr': config['value_lr']}], weight_decay=0)
        self.step = 0

    def update(self, episodes):
        rows = []
        for episode in episodes:
            if not episode['completed'] or episode.get('invalid'):
                raise ValueError('PPO updates require completed, accepted-action trajectories')
            steps = episode['steps']
            rewards = [0.0] * len(steps); rewards[-1] = episode['reward']
            terminal = [False] * len(steps); terminal[-1] = True
            adv, returns = advantages(rewards, [s['value'] for s in steps], terminal,
                                      gamma=self.cfg['gamma'], lam=self.cfg['gae_lambda'])
            rows.extend({**s, 'advantage': a, 'return': ret} for s,a,ret in zip(steps,adv,returns))
        active = [r['advantage'] for r in rows if len(r['keys']) > 1]
        norm = torch.tensor(active); mean, std = float(norm.mean()), float(norm.std(unbiased=False))
        for r in rows:
            r['advantage'] = (r['advantage']-mean)/max(std,1e-8) if len(r['keys']) > 1 else 0.0
        self.policy.train()
        metrics, count, stopped, stop_kl = {}, 0, False, 0.0
        dev = next(self.policy.parameters()).device
        for _ in range(self.cfg['update_epochs']):
            self.rng.shuffle(rows)
            for start in range(0,len(rows),self.cfg['minibatch_size']):
                batch = rows[start:start+self.cfg['minibatch_size']]
                actor_n = sum(len(r['keys']) > 1 for r in batch)
                self.optimizer.zero_grad(set_to_none=True)
                sums = dict(policy_loss=0., value_loss=0., entropy=0., kl=0., clip_fraction=0.)
                chunks = []
                for width in sorted({self.policy.bucket(r['ids']) for r in batch}):
                    bucket = [r for r in batch if self.policy.bucket(r['ids'])==width]
                    chunks.extend(bucket[i:i+self.cfg['microbatch_size']] for i in range(0,len(bucket),self.cfg['microbatch_size']))
                for chunk in chunks:
                    dist, values = self.policy([r['ids'] for r in chunk], [len(r['keys']) for r in chunk])
                    def tensor(key): return torch.tensor([r[key] for r in chunk], device=dev)
                    mask = torch.tensor([len(r['keys']) > 1 for r in chunk], device=dev)
                    loss, kl, clipped = clipped_objective(dist.log_prob(tensor('action')), tensor('logp'),
                        tensor('advantage'), self.cfg['clip_ratio'])
                    entropy = dist.entropy()
                    policy_loss = (loss[mask].sum()-self.cfg['entropy_coef']*entropy[mask].sum()) / max(actor_n,1)
                    value_loss = .5*(values-tensor('return')).square().sum()/len(batch)
                    (policy_loss+value_loss).backward()
                    sums['policy_loss'] += float(loss[mask].detach().sum())/max(actor_n,1)
                    sums['value_loss'] += float(value_loss.detach())
                    sums['entropy'] += float(entropy[mask].detach().sum())/max(actor_n,1)
                    sums['kl'] += float(kl[mask].sum())/max(actor_n,1)
                    sums['clip_fraction'] += float(clipped[mask].sum())/max(actor_n,1)
                if sums['kl'] > self.cfg['target_kl']:
                    self.optimizer.zero_grad(set_to_none=True); stopped = True; stop_kl = sums['kl']; break
                # Separate clipping keeps critic errors from shrinking the actor step.
                actor_norm = torch.nn.utils.clip_grad_norm_(self.actor_params, self.cfg['max_grad_norm'], error_if_nonfinite=True)
                value_norm = torch.nn.utils.clip_grad_norm_(self.policy.value.parameters(), self.cfg['max_grad_norm'], error_if_nonfinite=True)
                self.optimizer.step(); self.step += 1; count += 1
                sums['actor_grad_norm'] = float(actor_norm)
                sums['value_grad_norm'] = float(value_norm)
                for k,v in sums.items(): metrics[k] = metrics.get(k,0.0)+v
            if stopped: break
        self.policy.eval()
        returns = torch.tensor([r['return'] for r in rows])
        values = torch.tensor([r['value'] for r in rows])
        variance = returns.var(unbiased=False)
        explained = float(1-(returns-values).var(unbiased=False)/variance) if variance > 1e-8 else 0.0
        return {**{k:v/max(count,1) for k,v in metrics.items()}, 'updates': count,
                'kl_stopped': stopped, 'stop_kl': stop_kl, 'value_explained_variance': explained,
                'return_mean':float(returns.mean()),'step': self.step, 'decisions': len(rows),
                'trainable_decisions':len(active),
                'actor_lr': self.optimizer.param_groups[0]['lr']}

    def save(self, path, state):
        self.policy.save(path)
        torch.save({'optimizer': self.optimizer.state_dict(), 'step': self.step,
                    'python_rng': self.rng.getstate(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'state': state}, Path(path)/'ppo_state.pt')

    def resume(self, path):
        state = torch.load(Path(path)/'ppo_state.pt', map_location='cpu', weights_only=False)
        self.optimizer.load_state_dict(state['optimizer']); self.step = state['step']
        self.rng.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng']: torch.cuda.set_rng_state_all(state['cuda_rng'])
        return state['state']


@dataclass
class BattleStop:
    learning_rate: float
    min_learning_rate: float
    min_rounds: int = 10
    patience: int = 6
    min_gain: float = .03
    best_score: float = -1.0
    reference_score: float = -1.0
    best_round: int = 0
    bad_checks: int = 0
    floor_checks: int = 0

    def observe(self, score, round_id):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Development score must be the completed-game result in [0,1]')
        best = score > self.best_score
        if best: self.best_score, self.best_round = score, round_id
        if score >= self.reference_score + self.min_gain:
            self.reference_score = score; self.bad_checks = self.floor_checks = 0
        else:
            self.bad_checks += 1
            if self.learning_rate <= self.min_learning_rate: self.floor_checks += 1
            elif self.bad_checks >= 2:
                self.learning_rate = max(self.min_learning_rate, self.learning_rate*.5)
                self.bad_checks = 0
        return {'new_best': best, 'plateau': round_id >= self.min_rounds and self.floor_checks >= self.patience,
                **asdict(self)}
