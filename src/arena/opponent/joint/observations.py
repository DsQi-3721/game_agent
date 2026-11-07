"""Public-history encoding for partial-label external/configuration diagnostics."""
import numpy as np
import torch
from .tokens import ident

@torch.inference_mode()
def configuration_context(model,vocab,public,side):
    chunks=[];start=0
    for i,line in enumerate(public):
        if line.startswith('|start') and i>start:chunks.append((public[start:i],0));start=i
        if line.startswith('|turn|'):
            chunks.append((public[start:i+1],int(line.split('|')[2])));start=i+1
    if start<len(public):chunks.append((public[start:],chunks[-1][1] if chunks else 0))
    if not chunks:chunks=[(public,0)]
    encoded=[vocab.events(lines,side) for lines,_ in chunks];length=max(1,max(len(e) for e,n in encoded));device=next(model.parameters()).device
    e=np.zeros((1,len(chunks),length,4),np.int64);n=np.zeros((1,len(chunks),length,8),np.float32);s=np.zeros((1,len(chunks),32),np.float32)
    for i,((events,numeric),(_,turn)) in enumerate(zip(encoded,chunks)):e[0,i,:len(events)]=events;n[0,i,:len(numeric)]=numeric;s[0,i,0]=min(turn,200)/200;s[0,i,1]=float(turn==0)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):h,_=model.encode(torch.tensor(e,device=device),torch.tensor(n,device=device),torch.tensor(s,device=device),channel=1)
    return h[0,-1:].float()

def reveals(public,side,preview):
    facts=[{} for _ in preview];active=None
    for line in public:
        p=line.split('|')
        if len(p)<3:continue
        source=next((s.split('item:',1)[1] for s in p[3:] if s.startswith('[from] item:')),None)
        holder=next((s[5:] for s in p[3:] if s.startswith('[of] ')),p[2])
        if source and holder.startswith(side) and active is not None:facts[active]['item']=ident(source)
        ability=next((s.split('ability:',1)[1] for s in p[3:] if s.startswith('[from] ability:')),None)
        if ability and holder.startswith(side) and active is not None:facts[active]['ability']=ident(ability)
        if not p[2].startswith(side):continue
        if p[1] in ('switch','drag','replace'):active=next((i for i,s in enumerate(preview) if ident(s)==ident(p[3].split(',')[0])),None)
        if active is None:continue
        if p[1]=='move' and ident(p[3])!='struggle' and not any(s.startswith('[from]') for s in p[4:]):facts[active].setdefault('moves',[]).append(ident(p[3]))
        if p[1]=='-terastallize':facts[active]['tera']=ident(p[3])
        if p[1] in ('-item','-enditem') and not any(s.startswith('[from] move:') for s in p[4:]):facts[active]['item']=ident(p[3])
        if p[1]=='-ability' and not any(s.startswith('[from]') for s in p[4:]):facts[active]['ability']=ident(p[3])
    return facts
