import numpy as np, soundfile as sf, onnxruntime as ort, sys
from scipy.signal import resample
M = "X:/pi-code/spartacvc/models/ozada"
CPU = ["CPUExecutionProvider"]
voice_path = sys.argv[1] if len(sys.argv)>1 else f"{M}/beyonce.onnx"
hubert_path = sys.argv[2] if len(sys.argv)>2 else f"{M}/vec-256-layer-9.onnx"
rmvpe_path = f"{M}/rmvpe.onnx"
inp = sys.argv[3] if len(sys.argv)>3 else "C:/temp/test_speech.wav"
outp = sys.argv[4] if len(sys.argv)>4 else "C:/temp/spartacvc_fem.wav"

hub = ort.InferenceSession(hubert_path, providers=CPU)
rmvpe = ort.InferenceSession(rmvpe_path, providers=CPU)
voice = ort.InferenceSession(voice_path, providers=CPU)
print(f"voice={voice_path.split('/')[-1]}  hubert={hubert_path.split('/')[-1]}")

wav, sr = sf.read(inp); wav = wav.astype(np.float32)
wav16 = resample(wav, int(len(wav)*16000/sr)).astype(np.float32)
dur = len(wav16)/16000
TARGET_FPS = 80.0
T = int(round(dur * TARGET_FPS))

# HuBERT -> resample to 80fps
embed = hub.run(None, {hub.get_inputs()[0].name: wav16.reshape(1,1,-1)})[0]  # [1,T50,FEDIM]
T50 = embed.shape[1]
e = embed[0]
if T != T50:
    idx = np.linspace(0, T50-1, T)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo+1, T50-1); fr = (idx-lo)[:,None]
    e = e[lo]*(1-fr) + e[hi]*fr
phone = e.astype(np.float32)[None]

# mel + RMVPE -> f0 -> 80fps
def hz2mel(h): return 2595*np.log10(1+h/700)
def mel2hz(m): return 700*(10**(m/2595)-1)
n_fft, hop, n_mels, fmin, fmax = 1024, 160, 128, 30, 8000
pad=n_fft//2; xp=np.pad(wav16,(pad,pad),mode='reflect')
nf=1+(len(xp)-n_fft)//hop
idx2=np.arange(n_fft)[None,:]+hop*np.arange(nf)[:,None]
S=np.abs(np.fft.rfft(xp[idx2]*np.hanning(n_fft),axis=1))
nfbin=n_fft//2+1; low,high=hz2mel(fmin),hz2mel(fmax)
mp=np.linspace(low,high,n_mels+2); hp=mel2hz(mp); bp=np.floor((n_fft/2+1)*hp/(16000/2)).astype(int)
fb=np.zeros((n_mels,nfbin),np.float32)
for i in range(n_mels):
    l,c,r=bp[i],bp[i+1],bp[i+2]
    if c>l: fb[i,l:c]=(np.arange(l,c)-l)/(c-l)
    if r>c: fb[i,c:r]=(r-np.arange(c,r))/(r-c)
ms=np.log(np.clip(S@fb.T,1e-5,None)).astype(np.float32)
nff=ms.shape[0]; npad=32*((nff-1)//32+1)-nff
if npad>0: ms=np.pad(ms,((0,npad),(0,0)))
sal=rmvpe.run(None,{rmvpe.get_inputs()[0].name:ms.T.reshape(1,n_mels,-1)})[0][0]
cm=np.pad(20*np.arange(360)+1997.3794084376191,(4,4))
center=np.argmax(sal,axis=1); sp=np.pad(sal,((0,0),(4,4))); center+=4
st,en=center-4,center+5
ts=np.stack([sp[i,st[i]:en[i]] for i in range(len(center))],0)
tc=np.stack([cm[st[i]:en[i]] for i in range(len(center))],0)
cents=(ts*tc).sum(1)/ts.sum(1); cents[sal.max(1)<=0.05]=0
f0=10*(2**(cents/1200)); f0[f0==10]=0
f0 = np.interp(np.linspace(0,1,T), np.linspace(0,1,len(f0)), f0)

pf = f0.astype(np.float32).reshape(1,-1)
f0min,f0max=50,1100; fmm=1127*np.log(1+f0min/700); fmx=1127*np.log(1+f0max/700)
fm=1127*np.log(1+pf/700); fm[fm>0]=(fm[fm>0]-fmm)*254/(fmx-fmm)+1
fm[fm<=1]=1; fm[fm>255]=255
pitch=np.rint(fm).astype(np.int64)
out=voice.run(None,{voice.get_inputs()[0].name:phone,voice.get_inputs()[1].name:np.array([T],np.int64),
  voice.get_inputs()[2].name:pitch,voice.get_inputs()[3].name:pf,voice.get_inputs()[4].name:np.array([0],np.int64),
  voice.get_inputs()[5].name:np.random.randn(1,192,T).astype(np.float32)})[0]
a=out.reshape(-1); outdur=len(a)/32000
print(f"IN {dur:.2f}s OUT {outdur:.2f}s ratio={outdur/dur:.3f} peak={float(abs(a).max()):.3f} rms={float(np.sqrt(np.mean(a**2))):.4f}")
def flat(x):
    s=np.abs(np.fft.rfft(x*np.hanning(len(x))))+1e-9
    return np.exp(np.mean(np.log(s)))/np.mean(s)
print("flatness:", round(flat(a),4), "(speech~0.02-0.1)")
sf.write(outp,(a*32767).astype(np.int16),32000)
print("WROTE", outp)
