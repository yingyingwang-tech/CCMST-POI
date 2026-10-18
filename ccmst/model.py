import math
import torch
from torch import nn
import torch.nn.functional as F

class RelationEncoderLayer(nn.Module):
    """Per-head additive temporal/geographic/semantic bias in causal MHA."""
    def __init__(self,d=64,heads=4,dropout=.2):
        super().__init__();self.heads=heads
        self.attn=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.bias=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,heads))
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*d,d))
        self.drop=nn.Dropout(dropout)
    def forward(self,x,valid,relations,use_relation=True):
        batch,length,_=x.shape
        # Causal masking combined with padding. -1e4 keeps padded query positions finite.
        causal=torch.triu(torch.ones(length,length,dtype=torch.bool,device=x.device),1)
        invalid=causal[None,:,:] | (~valid[:,None,:])
        if use_relation:
            bias=self.bias(relations).permute(0,3,1,2) # B,H,L,L
            bias=bias.reshape(batch*self.heads,length,length)
        else:bias=x.new_zeros(batch*self.heads,length,length)
        additive=invalid[:,None].expand(batch,self.heads,length,length).reshape(batch*self.heads,length,length)
        bias=bias.masked_fill(additive,-1e4)
        attended,_=self.attn(x,x,x,attn_mask=bias,need_weights=False)
        x=self.norm1(x+self.drop(attended));x=self.norm2(x+self.drop(self.ff(x)))
        return x*valid.unsqueeze(-1)

class ScaleBranch(nn.Module):
    def __init__(self,d=64,heads=4,layers=2,dropout=.2):
        super().__init__()
        self.layers=nn.ModuleList([RelationEncoderLayer(d,heads,dropout) for _ in range(layers)])
        self.pool=nn.Linear(d,1)
    def forward(self,x,mask,relations,use_relation=True):
        for layer in self.layers:x=layer(x,mask,relations,use_relation)
        p=self.pool(x).squeeze(-1).masked_fill(~mask,-1e4)
        return (x*torch.softmax(p,dim=-1).unsqueeze(-1)).sum(1)

class CCMSTPOI(nn.Module):
    def __init__(self,np_,nc,d=64,heads=4,layers=2,dropout=.2,windows=(10,30,100),fusion='reliability',relation=True):
        super().__init__();self.windows=windows;self.fusion=fusion;self.relation=relation;self.npoi=np_
        self.poi=nn.Embedding(np_,d,padding_idx=0);self.cat=nn.Embedding(nc,d,padding_idx=0)
        self.hour=nn.Embedding(25,d,padding_idx=0);self.dow=nn.Embedding(8,d,padding_idx=0)
        self.pos=nn.Embedding(max(windows),d);self.geo=nn.Linear(2,d)
        self.project=nn.Sequential(nn.Linear(d*6,d),nn.LayerNorm(d),nn.Dropout(dropout))
        self.branches=nn.ModuleList([ScaleBranch(d,heads,layers,dropout) for _ in windows])
        self.branch_heads=nn.ModuleList([nn.Linear(d,np_) for _ in windows])
        self.rel_est=nn.Sequential(nn.Linear(d+5,d),nn.ReLU(),nn.Linear(d,1))
        self.att_est=nn.Linear(d,1);self.gate_est=nn.Linear(d,1)
        self.out=nn.Linear(d,np_)
    def features(self,b):
        mask=b['poi']!=0;pos=torch.arange(mask.shape[1],device=mask.device)[None].expand_as(mask)
        x=torch.cat([self.poi(b['poi']),self.cat(b['cat']),self.hour(b['hour']),self.dow(b['dow']),
             self.geo(b['coords']),self.pos(pos)],dim=-1)
        return self.project(x)*mask.unsqueeze(-1),mask
    def relation_features(self,b):
        """Log time-in-days, log distance-in-km, same-category indicator."""
        t=b['ts'];coords=b['coords'];cats=b['cat']
        dt=torch.log1p((t[:,:,None]-t[:,None,:]).abs())
        lat=coords[:,:,0]*math.pi/2;lon=coords[:,:,1]*math.pi
        dlat=lat[:,:,None]-lat[:,None,:];dlon=lon[:,:,None]-lon[:,None,:]
        hav=(torch.sin(dlat/2)**2+torch.cos(lat[:,:,None])*torch.cos(lat[:,None,:])*torch.sin(dlon/2)**2).clamp(0,1-1e-7)
        dist=torch.log1p(6371*2*torch.asin(torch.sqrt(hav)))
        same=(cats[:,:,None]==cats[:,None,:]).to(dt.dtype)
        return torch.stack([dt,dist,same],-1)

    def quality(self,b,mask,prob,window):
        """All quality descriptors derived from observed prefix/model outputs (never label)."""
        n=mask.sum(-1).float();availability=n/window
        ts=b['ts'];latlon=b['coords'];cat=b['cat']
        pair=mask[:,1:]&mask[:,:-1]
        dt=(ts[:,1:]-ts[:,:-1]).abs()
        # Haversine distance on original normalized coordinates.
        lat1=latlon[:,:-1,0]*math.pi/2;lat2=latlon[:,1:,0]*math.pi/2
        dl=(lat2-lat1);dlo=(latlon[:,1:,1]-latlon[:,:-1,1])*math.pi
        hav=(torch.sin(dl/2)**2+torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlo/2)**2).clamp(0,1)
        dist=6371*2*torch.asin(torch.sqrt(hav.clamp(max=1-1e-7)))
        def regularity(v):
            count=pair.sum(1).float();mean=(v*pair).sum(1)/count.clamp_min(1)
            var=(((v-mean[:,None])**2)*pair).sum(1)/count.clamp_min(1)
            return torch.where(count>1,1/(1+torch.sqrt(var+1e-8)/(mean+1e-5)),torch.zeros_like(mean))
        temporal=regularity(dt);geo=regularity(dist)
        semantic=(((cat[:,1:]==cat[:,:-1])&pair).float().sum(1)/pair.sum(1).clamp_min(1))
        entropy=-(prob.clamp_min(1e-12)*prob.clamp_min(1e-12).log()).sum(-1)/math.log(self.npoi)
        return torch.stack([availability,temporal,geo,semantic,entropy],-1)
    def forward(self,b):
        x,mask=self.features(b);states=[];aux=[];qualities=[]
        for k,w in enumerate(self.windows):
            # Select most recent valid observations for each batch item; right-pad to a fixed window.
            indices=(mask.cumsum(1)*mask).long()
            length=mask.sum(1);start=(length-w).clamp_min(0)
            select=start[:,None]+torch.arange(w,device=x.device)[None,:]
            select=select.clamp(max=x.shape[1]-1)
            xx=x.gather(1,select.unsqueeze(-1).expand(-1,-1,x.shape[-1]));mm=mask.gather(1,select)
            bb={key:val.gather(1,select.unsqueeze(-1).expand(-1,-1,val.shape[-1])) if key=='coords' else val.gather(1,select)  if key in ('poi','cat','hour','dow','coords','ts') else val for key,val in b.items()}
            relations=self.relation_features(bb)
            h=self.branches[k](xx,mm,relations,self.relation);logits=self.branch_heads[k](h)
            p=F.softmax(logits,dim=-1)
            q=self.quality(bb,mm,p,w);states.append(h);aux.append(logits);qualities.append(q)
        hs=torch.stack(states,1);qs=torch.stack(qualities,1)
        if self.fusion=='uniform':alpha=torch.ones(hs.shape[:2],device=hs.device)/len(self.windows)
        elif self.fusion=='attention':alpha=F.softmax(self.att_est(hs).squeeze(-1),dim=-1)
        elif self.fusion=='gating':alpha=F.softmax(self.gate_est(hs).squeeze(-1),dim=-1)
        elif self.fusion=='quality':alpha=F.softmax(self.rel_est(torch.cat([hs,qs],-1)).squeeze(-1),dim=-1)
        else:alpha=F.softmax(self.rel_est(torch.cat([hs,qs],-1)).squeeze(-1),dim=-1)
        fused=(alpha.unsqueeze(-1)*hs).sum(1)
        return {'logits':self.out(fused),'aux_logits':torch.stack(aux,1),'alpha':alpha,'quality':qs}

def objective(out,labels,aux_weight=.2,rel_weight=.1,target_temperature=1.):
    logits=out['logits'];branch=out['aux_logits'];b,k,v=branch.shape
    main=F.cross_entropy(logits,labels)
    losses=F.cross_entropy(branch.reshape(b*k,v),labels[:,None].expand(-1,k).reshape(-1),reduction='none').reshape(b,k)
    auxiliary=losses.mean()
    target=F.softmax(-losses.detach()/target_temperature,dim=-1)
    kl=F.kl_div(out['alpha'].clamp_min(1e-12).log(),target,reduction='batchmean')
    return main+aux_weight*auxiliary+rel_weight*kl,{'main':main.detach(),'aux':auxiliary.detach(),'reliability':kl.detach()}
