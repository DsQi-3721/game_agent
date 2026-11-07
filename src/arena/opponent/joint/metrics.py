"""Proper probability scores and calibration, not battle win rate."""
import numpy as np

class Scores:
    def __init__(self):self.n=0;self.totals=np.zeros(6);self.bins=np.zeros((15,3));self.game_losses={}
    def add(self,p,y,game=None):
        p=np.asarray(p,float);p=p/p.sum();order=np.argsort(-p);loss=-np.log(max(p[y],1e-12));correct=int(order[0]==y)
        self.n+=1;self.totals+=np.array([loss,np.square(p).sum()-2*p[y]+1,correct,int(y in order[:3]),-np.sum(p*np.log(np.maximum(p,1e-12))),p[y]])
        self.bins[min(14,int(p.max()*15))]+=np.array([1,p.max(),correct])
        if game is not None:self.game_losses.setdefault(game,[]).append(loss)
        return loss
    def result(self):
        if not self.n:return {'count':0}
        values=self.totals/self.n;result=dict(zip(('nll','brier','top1','top3','entropy','true_action_probability'),map(float,values)))
        result.update(count=self.n,ece=float(np.abs(self.bins[:,1]-self.bins[:,2]).sum()/self.n));return result
    def add_many(self,p,y,group=None):
        p=np.asarray(p,float);y=np.asarray(y);p=p/p.sum(-1,keepdims=True);rows=np.arange(len(y));chosen=p[rows,y];loss=-np.log(np.maximum(chosen,1e-12));correct=p.argmax(-1)==y
        top3=(np.argpartition(p,-min(3,p.shape[-1]),axis=-1)[:,-min(3,p.shape[-1]):]==y[:,None]).any(-1);confidence=p.max(-1);bins=np.minimum(14,(confidence*15).astype(int))
        self.n+=len(y);self.totals+=np.array([loss.sum(),(np.square(p).sum(-1)-2*chosen+1).sum(),correct.sum(),top3.sum(),-(p*np.log(np.maximum(p,1e-12))).sum(),chosen.sum()])
        self.bins[:,0]+=np.bincount(bins,minlength=15);self.bins[:,1]+=np.bincount(bins,weights=confidence,minlength=15);self.bins[:,2]+=np.bincount(bins,weights=correct,minlength=15)
        if group is not None:self.game_losses.setdefault(group,[]).extend(loss.tolist())
        return loss

def paired_interval(a,b,seed=7):
    keys=sorted(set(a)&set(b));d=np.array([np.mean(a[k])-np.mean(b[k]) for k in keys]);rng=np.random.default_rng(seed)
    samples=np.array([rng.choice(d,len(d)).mean() for _ in range(1000)])
    return {'mean_nll_difference':float(d.mean()),'95pct_interval':np.quantile(samples,[.025,.975]).tolist(),'independent_groups':len(d),'sign':'negative favors first method'}

class Traditional:
    def __init__(self,parameters):
        self.weights=np.asarray(parameters['weights']);self.temperature=parameters.get('temperature',1.);self.forgetting=parameters.get('forgetting',.98);self.posterior={}
    def predict(self,row,player):
        scores=row['features']@self.weights.T/self.temperature;scores-=scores.max(0);p=np.exp(scores);p/=p.sum(0)
        weights=self.posterior.get(player,np.ones(len(self.weights))/len(self.weights));mixture=p@weights
        y=row['target'];updated=weights*p[y];updated/=updated.sum();self.posterior[player]=self.forgetting*updated+(1-self.forgetting)/len(weights)
        return mixture,p.mean(-1)
