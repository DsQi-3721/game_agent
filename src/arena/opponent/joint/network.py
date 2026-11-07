"""Shared causal encoder with predictive, recurrent strategy uncertainty."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class NetworkConfig:
    vocabulary: int
    layers: int = 8
    width: int = 256
    heads: int = 8
    latent: int = 32
    prototypes: int = 16
    memory: int = 32


class MemoryLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.norm = nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.output = nn.Linear(c.width, c.width)
        self.ffnorm = nn.LayerNorm(c.width)
        self.ff = nn.Sequential(nn.Linear(c.width, 4 * c.width), nn.GELU(), nn.Linear(4 * c.width, c.width))

    def forward(self, x, memory=None, mask=None):
        n = x.shape[1]
        joined = x if memory is None else torch.cat([memory, x], 1)
        m = joined.shape[1] - n
        q, k, v = self.qkv(self.norm(joined)).chunk(3, -1)
        def split(t):
            return t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)
        q, k, v = map(split, (q[:, m:], k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        x = x + self.output(y.transpose(1, 2).flatten(2))
        return x + self.ff(self.ffnorm(x))


class JointNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        self.embedding = nn.Embedding(c.vocabulary, c.width, padding_idx=0)
        self.event_numeric = nn.Linear(8, c.width)
        self.state_numeric = nn.Linear(32, c.width)
        self.position = nn.Embedding(256, c.width)
        self.channel = nn.Embedding(3, c.width)
        self.field_position = nn.Parameter(torch.zeros(4, c.width))
        self.config_summary = nn.Sequential(nn.Linear(21*c.width, c.width), nn.GELU(), nn.Linear(c.width, c.width))
        self.event_gate = nn.Linear(c.width, 1)
        self.layers = nn.ModuleList(MemoryLayer(c) for _ in range(c.layers))
        self.norm = nn.LayerNorm(c.width)
        self.strategy = nn.Linear(c.width, 2*c.latent + 3 + c.prototypes)
        self.prototype = nn.Parameter(torch.randn(c.prototypes, c.latent) * .05)
        self.transition = nn.Sequential(nn.Linear(c.width+c.latent, 128), nn.GELU(), nn.Linear(128, c.latent))
        self.action_features = nn.Sequential(nn.Linear(32, c.width), nn.GELU(), nn.Linear(c.width, c.width))
        self.policy = nn.Linear(c.width+c.latent, c.width)
        self.outcome_task = nn.Parameter(torch.zeros(c.width))
        self.configuration = nn.Linear(c.width, c.vocabulary, bias=False)
        self.proposal = nn.Linear(c.width, c.vocabulary, bias=False)
        self.context_projection = nn.Linear(c.width, c.width)
        self.fact_query = nn.Sequential(nn.Linear(c.width, c.width), nn.GELU(), nn.LayerNorm(c.width))
        self.action_temperature = 1.
        self.behavior_probes = {}
        self.behavior_prior = None
        with torch.no_grad():
            self.strategy.bias[c.latent:2*c.latent].fill_(-5.)
            self.strategy.bias[2*c.latent:2*c.latent+3].copy_(torch.tensor([3.,-2.,0.]))
            self.transition[-1].bias.fill_(2.)

    def encode(self, events, numeric, state, configuration=None, memory=None,
               channel=0, position=None, segments=None):
        valid = (events != 0).any(-1)
        embedded = self.embedding(events)
        embedded = embedded * (1 + self.field_position)
        e = embedded.sum(-2) + self.event_numeric(numeric)
        order = torch.arange(e.shape[-2], device=e.device)
        # Position is part of the event representation, not only a pooling bias.
        e = e + self.position(order % 256)
        weights = self.event_gate(e).squeeze(-1).masked_fill(~valid, -1e4).softmax(-1)
        x = (e * weights[..., None] * valid[..., None]).sum(-2)
        x = x + self.state_numeric(state) + self.channel.weight[channel]
        if configuration is not None:
            sets = self.embedding(configuration).reshape(*configuration.shape[:-1], 6, 21*self.config.width)
            x = x + self.config_summary(sets).mean(-2)
        b, length = x.shape[:2]
        if position is None:
            position = torch.arange(length, device=x.device)
        x = x + self.position(position % 256)
        if segments is None:
            segments = torch.zeros((b, length), dtype=torch.long, device=x.device)
        previous_segments = memory[-1] if memory else segments[:, :0]
        all_segments = torch.cat([previous_segments, segments], 1)
        m = previous_segments.shape[1]
        key = torch.arange(m+length, device=x.device)[None, :]
        query = torch.arange(length, device=x.device)[:, None] + m
        local = (key <= query) & (key >= query-self.config.memory)
        mask = local[None, None] & (segments[:, None, :, None] == all_segments[:, None, None, :])
        memories = []
        for i, layer in enumerate(self.layers):
            previous = memory[i] if memory else None
            cached = torch.cat([previous, x], 1) if previous is not None else x
            memories.append(cached[:, -self.config.memory:].detach())
            x = layer(x, previous, mask)
        memories.append(all_segments[:, -self.config.memory:])
        return self.norm(x), memories

    def strategy_distribution(self, h, previous=None, components=False):
        c = self.config
        behavior_previous=None if previous is None or previous.shape[-1]==c.latent else previous[...,c.latent:]
        previous=None if previous is None else previous[...,:c.latent]
        mean, logvar, change, prototype = torch.split(self.strategy(h), [c.latent, c.latent, 3, c.prototypes], -1)
        mean = mean + prototype.softmax(-1) @ self.prototype
        if self.behavior_prior is not None:
            probability=change.softmax(-1)*.98
            probability[...,1]+=.02
            change=probability.log()
        variance = logvar.clamp(-8, 0).exp()
        if components:
            mean=mean.unsqueeze(-2).expand(*mean.shape[:-1],3,c.latent)
            variance=variance.unsqueeze(-2).expand_as(mean)
        if previous is not None:
            modes = change.softmax(-1)
            retention = self.transition(torch.cat([h, previous], -1)).sigmoid()
            if components:
                gate=retention.unsqueeze(-2)*torch.tensor([1.,0.,.5],device=h.device)[:,None]
                previous=previous.unsqueeze(-2)
            else:gate=retention*(modes[...,:1]+.5*modes[...,2:3])
            mean = gate*previous + (1-gate)*mean
            variance = (1-gate).square()*variance
        variance=variance.clamp_min(1e-6)
        if self.behavior_prior is not None:
            center=self.behavior_prior['mean'].to(h.device).expand(*h.shape[:-1],-1)
            spread=self.behavior_prior['scale'].to(h.device).square().expand_as(center)
            if behavior_previous is None:
                means=center.unsqueeze(-2).expand(*center.shape[:-1],3,center.shape[-1]);variances=spread.unsqueeze(-2).expand_as(means)
            else:
                means=torch.stack([behavior_previous,center,behavior_previous],-2)
                variances=torch.stack([spread*1e-8,spread,spread*.05**2],-2)
            if not components:
                mixture=change.softmax(-1)[...,None];average=(mixture*means).sum(-2)
                variances=(mixture*(variances+(means-average.unsqueeze(-2)).square())).sum(-2);means=average
            mean=torch.cat([mean,means],-1);variance=torch.cat([variance,variances],-1)
        return mean, variance.clamp_min(1e-12).log(), change, prototype

    def sample_strategy(self,h,previous=None,*,generator=None,noise=None,eligible=None):
        """Sample a keep/reset/drift transition and its conditional Gaussian."""
        means,variances,change,prototype=self.strategy_distribution(h,previous,components=True)
        dimensions=means.shape[-1]
        if noise is None:
            noise=torch.randn((*h.shape[:-1],dimensions+1),device=h.device,generator=generator)
        uniform=.5*(1+torch.erf(noise[...,dimensions]/math.sqrt(2)))
        mode=(uniform[...,None]>change.float().softmax(-1).cumsum(-1)).sum(-1).clamp_max(2)
        if eligible is not None:mode=torch.where(eligible,mode,torch.zeros_like(mode))
        if previous is None:mode=torch.zeros_like(mode)
        index=mode[...,None,None].expand(*mode.shape,1,dimensions)
        mean=means.gather(-2,index).squeeze(-2);lv=variances.gather(-2,index).squeeze(-2)
        return mean+noise[...,:dimensions]*(.5*lv).exp(),noise,mode,change,prototype

    def strategy_paths(self, h, samples=4, game_start=None, history='full', previous_games=None):
        """Reparameterized recursive paths; the previous state is a model sample."""
        k=max(1,samples);b=h.shape[0];previous=None;paths=[];changes=[]
        for t in range(h.shape[1]):
            current=h[:,t].expand(k,-1,-1)
            if previous is not None and previous_games is not None:
                previous=torch.where(game_start[:,t][None,:,None],previous_games,previous)
            mean,lv,change,_=self.strategy_distribution(current,previous)
            if previous is not None and history=='none':
                fresh,fresh_lv,_,_=self.strategy_distribution(current)
                reset=game_start[:,t][None,:,None]
                mean=torch.where(reset,fresh,mean);lv=torch.where(reset,fresh_lv,lv)
            previous=mean+torch.randn_like(mean)*(.5*lv).exp() if samples else mean
            paths.append(previous);changes.append(change[0])
        return torch.stack(paths,2),torch.stack(changes,1)

    def policy_logits(self, h, features, z=None, action_tokens=None):
        if z is None:
            z = self.strategy_distribution(h)[0]
        query = self.policy(torch.cat([h, z[...,:self.config.latent]], -1))
        encoded = self.action_features(features)
        if action_tokens is not None:
            encoded = encoded + self.embedding(action_tokens).sum(-2)
        logits=(encoded*query.unsqueeze(-2)).sum(-1)/math.sqrt(self.config.width)
        if z.shape[-1]>self.config.latent:
            from .strategy import behavior_features
            logits+=(behavior_features(features)*z[...,None,self.config.latent:]).sum(-1)
        return logits/self.action_temperature

    def outcome_keys(self, move_tokens, preview, kind_tokens):
        # Visible outcome classes: move x observed Tera, six switches/leads, unknown.
        moves = self.embedding(move_tokens) + self.embedding(kind_tokens[0])
        tera = self.embedding(kind_tokens[3])
        moves = torch.stack([moves, moves+tera], 1).flatten(0, 1)
        teams = self.embedding(preview) + self.embedding(kind_tokens[1])
        missing = self.embedding(kind_tokens[2]).expand(*preview.shape[:-1], 1, -1)
        return moves, torch.cat([teams, missing], -2)

    def outcome_logits(self, h, z, keys):
        query = self.policy(torch.cat([h, z[...,:self.config.latent]], -1)) + self.outcome_task
        moves, rest = keys
        move_logits = query @ moves.T
        switch_logits = (query.unsqueeze(-2)*rest).sum(-1)
        return torch.cat([move_logits, switch_logits], -1) / math.sqrt(self.config.width) / self.action_temperature

    def config_logits(self, context, tokens, proposal=False):
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device) % 256)
        x = x + self.context_projection(context)[:, None]
        for layer in self.layers:
            x = layer(x)
        return (self.proposal if proposal else self.configuration)(self.norm(x))

    def config_prefill(self, projected_context, tokens):
        """Configuration decoder states and keys/values for subsequent edits."""
        x=self.embedding(tokens)+self.position(torch.arange(tokens.shape[1],device=tokens.device))+projected_context[:,None]
        cache=[]
        for layer in self.layers:
            q,k,v=layer.qkv(layer.norm(x)).chunk(3,-1)
            q,k,v=[a.reshape(len(x),x.shape[1],layer.heads,-1).transpose(1,2) for a in (q,k,v)]
            keys=k.new_zeros((len(x),layer.heads,126,k.shape[-1]));values=torch.zeros_like(keys)
            keys[:,:,:tokens.shape[1]]=k;values[:,:,:tokens.shape[1]]=v;cache.append((keys,values))
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            x=x+layer.output(y.transpose(1,2).flatten(2));x=x+layer.ff(layer.ffnorm(x))
        return self.norm(x),cache

    def config_step(self, projected_context, token, positions, cache):
        """Advance one position per row; rows may edit different roster slots."""
        x=self.embedding(token)[:,None]+self.position(positions)[:,None]+projected_context[:,None]
        rows=torch.arange(len(x),device=x.device)
        mask=(torch.arange(126,device=x.device)[None,:]<=positions[:,None])[:,None,None,:]
        for layer,(keys,values) in zip(self.layers,cache):
            q,k,v=layer.qkv(layer.norm(x)).chunk(3,-1)
            q,k,v=[a.reshape(len(x),1,layer.heads,-1).transpose(1,2) for a in (q,k,v)]
            keys[rows,:,positions]=k[:,:,0];values[rows,:,positions]=v[:,:,0]
            y=F.scaled_dot_product_attention(q,keys,values,attn_mask=mask)
            x=x+layer.output(y.transpose(1,2).flatten(2));x=x+layer.ff(layer.ffnorm(x))
        return self.norm(x)[:,0]

    def attribute_logits(self, context, queries):
        # A partial-label auxiliary marginal; missing AR prefixes are never fabricated.
        x = self.fact_query(self.context_projection(context) + self.embedding(queries).sum(-2))
        return self.configuration(x)

    def forward(self, batch, samples=4, history='full'):
        h, _ = self.encode(batch['events'], batch['numeric'], batch['state'],
                           configuration=batch.get('configuration'), channel=batch['channel'],
                           position=batch['position'], segments=batch['segments'])
        z,change=self.strategy_paths(h,samples,batch['game_start'],history)
        return {'h': h, 'z': z, 'change': change}

    def warm_start(self, checkpoint):
        old = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
        current = self.state_dict()
        copied = []
        for key, value in old.items():
            if key.startswith(('strategy.', 'transition.', 'prototype', 'latent_posterior.', 'memory_readout.')):
                continue
            if key in current and current[key].shape == value.shape:
                current[key].copy_(value); copied.append(key)
            elif key in ('embedding.weight', 'configuration.weight', 'proposal.weight'):
                current[key][:len(value)].copy_(value); copied.append(key)
        self.load_state_dict(current)
        with torch.no_grad():
            self.policy.weight[:, self.config.width:].zero_()
        return copied

    def save(self, path, **metadata):
        torch.save({'config': asdict(self.config), 'model': self.state_dict(),
                    'temperature': self.action_temperature, 'behavior_probes':self.behavior_probes,'behavior_prior':self.behavior_prior,
                    'training_semantics': 'predictive_recurrent_z', **metadata}, path)

    def behavior_profile(self,z,weights):
        from .strategy import behavior_profile
        return behavior_profile(self,z,weights)

    @classmethod
    def load(cls, path, device='cpu'):
        d = torch.load(path, map_location='cpu', weights_only=True)
        m = cls(NetworkConfig(**d['config']))
        m.load_state_dict(d['model'])
        m.action_temperature = d.get('temperature', 1.)
        m.behavior_probes=d.get('behavior_probes',{})
        m.behavior_prior=d.get('behavior_prior')
        if m.behavior_prior is not None:m.behavior_prior={k:v.to(device) for k,v in m.behavior_prior.items()}
        m.metadata = {k: d[k] for k in ('steps', 'epoch', 'training_semantics') if k in d}
        return m.to(device).eval()
