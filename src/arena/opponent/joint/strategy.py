"""Particle change histories and standardized behavioral predictions."""
import torch


def change_eligible(state,mask):
    return (state[...,1:3].sum(-1)==0)&(mask.sum(-1)>1)


def advance_changes(history,mode,eligible):
    shifted=torch.cat([history[...,1:],mode[...,None].to(history.dtype)],-1)
    return torch.where(eligible[...,None],shifted,history)


def change_marginals(history,weights):
    """The two change types may both occur within the same five-decision window."""
    weights=torch.as_tensor(weights,device=history.device,dtype=torch.float64)
    weights=weights/weights.sum()
    indicators=torch.stack([(history==0).all(-1),(history==1).any(-1),(history==2).any(-1)],-1)
    return (weights[...,None]*indicators).sum(0)


BEHAVIORS={
    'switch_when_threatened':('受到属性威胁且可以自主换人时的换人概率',2),
    'setup_with_high_hp':('高血量且无明显属性威胁、可以强化时的强化概率',12),
    'recover_with_low_hp':('血量不超过一半且可以回复时的回复概率',11),
    'tera_when_available':('可以太晶化时的太晶化概率',3),
}


def behavior_features(features):
    selected=features[...,[0,2,3,11,12,21,22,17,13,15,27]]
    accuracy=(features[...,6]-1)*features[...,0]
    return torch.cat([selected,accuracy[...,None]],-1)


def behavior_opportunities(features,phase):
    if phase!=1:return {}
    f=features
    return dict(switch_when_threatened=bool((f[:,2]>0).any() and (f[:,0]+f[:,1]>0).any() and f[0,23]>=1/3),
        setup_with_high_hp=bool((f[:,12]>0).any() and (f[:,0]>0).any() and f[0,9]>=.75 and f[0,23]==0),
        recover_with_low_hp=bool((f[:,11]>0).any() and (f[:,11]==0).any() and f[0,9]<=.5),
        tera_when_available=bool((f[:,3]>0).any() and (f[:,3]==0).any()))


@torch.inference_mode()
def behavior_profile(model,z,weights):
    if not model.behavior_probes:return {}
    weights=torch.as_tensor(weights,device=z.device,dtype=torch.float32);result={}
    for name,probes in model.behavior_probes.items():
        values=[]
        for probe in probes:
            h=probe['h'].to(z.device);f=probe['features'].to(z.device);tokens=probe['actions'].to(z.device)
            keys=model.action_features(f)+model.embedding(tokens).sum(-2)
            query=model.policy(torch.cat([h.expand(len(z),-1),z[:,:model.config.latent]],-1))
            logits=query@keys.T/(model.config.width**.5)
            if z.shape[-1]>model.config.latent:logits+=z[:,model.config.latent:]@behavior_features(f).T
            p=(logits/model.action_temperature).softmax(-1)
            values.append((p*f[:,BEHAVIORS[name][1]]).sum(-1))
        values=torch.stack(values).mean(0);ordered,indices=values.sort();mass=weights[indices].cumsum(0)
        bounds=ordered[torch.searchsorted(mass,torch.tensor([.05,.95],device=z.device)).clamp_max(len(z)-1)]
        result[name]=dict(description=BEHAVIORS[name][0],probability=float(weights@values),
            posterior_interval_90=bounds.cpu().tolist(),reference_situations=len(probes))
    return result
