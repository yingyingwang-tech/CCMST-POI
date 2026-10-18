import torch,math

def perturb(batch,mode='random',ratio=.2,seed=42):
    """Batch already tensorized and right-padded; labels never touched.
    Deterministic by explicit seed. For coherent repeated tests use batch_size=1.
    """
    b={k:(v.clone() if torch.is_tensor(v) else v) for k,v in batch.items()}
    if mode=='none' or ratio==0:return b
    g=torch.Generator(device='cpu').manual_seed(seed)
    keys=['poi','cat','hour','dow','coords','ts'];n=b['poi'].shape[0]
    for i in range(n):
        length=int((b['poi'][i]!=0).sum())
        if length<2:continue
        if mode in ('random','block','recent'):
            removal=min(length-1,max(1,int(round(length*ratio))))
            if mode=='random':inds=torch.randperm(length,generator=g)[:removal].tolist()
            elif mode=='recent':inds=list(range(length-removal,length))
            else:
                start=int(torch.randint(0,length-removal+1,(1,),generator=g));inds=list(range(start,start+removal))
            keep=[j for j in range(length) if j not in inds]
            for key in keys:
                val=b[key][i].clone();b[key][i].zero_();b[key][i,:len(keep)]=val[keep]
        elif mode=='time':
            noise=torch.randn(length,generator=g)*ratio/1440 # ratio in minutes
            b['ts'][i,:length]+=noise
            # Keep original event order, clamp to past of target (timestamp not provided).
            hours=((b['ts'][i,:length].remainder(1)*24).long()%24)+1
            b['hour'][i,:length]=hours
        elif mode=='geo':
            # ratio in meters; coordinates stored as latitude/90,longitude/180
            lat=b['coords'][i,:length,0]*90
            latnoise=torch.randn(length,generator=g)*ratio/111320
            lonnoise=torch.randn(length,generator=g)*ratio/(111320*torch.cos(lat*math.pi/180).abs().clamp_min(.1))
            b['coords'][i,:length,0]+=(latnoise/90)
            b['coords'][i,:length,1]+=(lonnoise/180)
        else:raise ValueError(mode)
    return b
