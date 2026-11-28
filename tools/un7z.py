import sys, lzma, struct, zlib, os, bz2
def rnum(b,p):
    f=b[p];p+=1;m=0x80;v=0
    for i in range(8):
        if f&m==0:
            hi=f&(m-1); v|=hi<<(8*i); return v,p
        v|=b[p]<<(8*i);p+=1;m>>=1
    return v,p
class R:
    def __init__(s,b):s.b=b;s.p=0
    def u8(s):v=s.b[s.p];s.p+=1;return v
    def num(s):v,s.p=rnum(s.b,s.p);return v
    def bytes(s,n):v=s.b[s.p:s.p+n];s.p+=n;return v
def bitvec(r,n):
    out=[];m=0;cur=0
    for i in range(n):
        if m==0:cur=r.u8();m=0x80
        out.append(bool(cur&m));m>>=1
    return out
def optbits(r,n):
    alld=r.u8()
    return [True]*n if alld else bitvec(r,n)
def read_streams(r):
    info={}
    while True:
        t=r.u8()
        if t==0:break
        if t==6: # PackInfo
            info['packpos']=r.num();n=r.num();info['packsizes']=[]
            while True:
                t2=r.u8()
                if t2==0:break
                if t2==9: info['packsizes']=[r.num() for _ in range(n)]
                elif t2==10:
                    d=optbits(r,n)
                    for x in d:
                        if x:r.bytes(4)
        elif t==7: # UnpackInfo
            assert r.u8()==11
            nf=r.num();assert r.u8()==0
            folders=[]
            for _ in range(nf):
                nc=r.num();coders=[];nin=0;nout=0
                for _ in range(nc):
                    fl=r.u8();idl=fl&15;cid=r.bytes(idl)
                    if fl&0x10: ni=r.num();no=r.num()
                    else: ni=no=1
                    props=r.bytes(r.num()) if fl&0x20 else b''
                    coders.append((cid,ni,no,props));nin+=ni;nout+=no
                npk=nout-1 # bindpairs = nout-1
                bp=[(r.num(),r.num()) for _ in range(nout-1)]
                npacked=nin-len(bp)
                if npacked>1: [r.num() for _ in range(npacked)]
                folders.append(dict(coders=coders,bp=bp,nout=nout))
            assert r.u8()==12
            for f in folders: f['unsizes']=[r.num() for _ in range(f['nout'])]
            while True:
                t2=r.u8()
                if t2==0:break
                if t2==10:
                    d=optbits(r,nf)
                    for x in d:
                        if x:r.bytes(4)
            info['folders']=folders
        elif t==8: # substreams
            info['sub']=True
            folders=info['folders'];nums=[1]*len(folders)
            t2=r.u8()
            if t2==13: nums=[r.num() for _ in folders];t2=r.u8()
            sizes=[]
            if t2==9:
                for f,n in zip(folders,nums):
                    tot=f['unsizes'][-1] if False else f['unsizes'][f['mainout']] if 'mainout' in f else f['unsizes'][-1]
                    s=[r.num() for _ in range(n-1)] if n>0 else []
                    s.append(tot-sum(s)) if n>0 else None
                    sizes.append(s)
                t2=r.u8()
            else:
                for f,n in zip(folders,nums):
                    sizes.append([f['unsizes'][-1]] if n==1 else [])
            while t2!=0:
                if t2==10:
                    cnt=sum(n for n in nums)  # approx: digests for those w/o crc
                    d=optbits(r,cnt)
                    for x in d:
                        if x:r.bytes(4)
                t2=r.u8()
            info['substream_sizes']=sizes
    return info
def decode_folder(f,data):
    # assume simple chain: single packed stream through coders in order
    cur=data
    for (cid,ni,no,props) in f['coders']:
        if cid==b'\x21': # LZMA2
            d=props[0];filt=[{'id':lzma.FILTER_LZMA2,'dict_size':(2|(d&1))<<(d//2+11) if d<40 else 0xFFFFFFFF}]
            cur=lzma.LZMADecompressor(lzma.FORMAT_RAW,filters=filt).decompress(cur)
        elif cid==b'\x03\x01\x01':
            lc=props[0]; 
            ds=struct.unpack('<I',props[1:5])[0]
            pb=lc//45;lp=(lc%45)//9;lcc=lc%9
            filt=[{'id':lzma.FILTER_LZMA1,'dict_size':ds,'lc':lcc,'lp':lp,'pb':pb}]
            cur=lzma.LZMADecompressor(lzma.FORMAT_RAW,filters=filt).decompress(cur)
        elif cid==b'\x00': pass
        elif cid==b'\x04\x01\x08': cur=zlib.decompress(cur,-15)
        elif cid==b'\x04\x02\x02': cur=bz2.decompress(cur)
        else: raise Exception('coder %r'%cid)
    return cur
def extract(path,out):
    b=open(path,'rb').read()
    assert b[:6]==b"7z\xbc\xaf\x27\x1c"
    off,size,_=struct.unpack('<QQI',b[12:32])
    hdr=b[32+off:32+off+size]
    while hdr[0]==0x17:
        r=R(hdr);r.u8();info=read_streams(r)
        pos=32+info['packpos'];ps=info['packsizes'][0]
        hdr=decode_folder(info['folders'][0],b[pos:pos+ps])
    r=R(hdr);assert r.u8()==1
    info={}
    names=[];sizes=[];emptystream=[];emptyfile=[]
    while True:
        t=r.u8()
        if t==0:break
        if t==2: # archive props
            while True:
                tt=r.u8()
                if tt==0:break
                r.bytes(r.num())
        elif t==3: r.u8() # additional streams - unsupported
        elif t==4:
            info=read_streams(r)
        elif t==5:
            nfiles=r.num()
            while True:
                t2=r.u8()
                if t2==0:break
                sz=r.num();end=r.p+sz
                if t2==14: emptystream=bitvec(r,nfiles)
                elif t2==15: emptyfile=bitvec(r,sum(emptystream))
                elif t2==17:
                    ext=r.u8();raw=r.bytes(end-r.p)
                    s=raw.decode('utf-16-le');names=s.split('\x00')[:nfiles]
                r.p=end
    if not emptystream: emptystream=[False]*nfiles
    # fix 'mainout'
    pos=32+info['packpos'];pi=0
    files=[n for n,e in zip(names,emptystream) if not e]
    ssz=info.get('substream_sizes') or [[f['unsizes'][-1]] for f in info['folders']]
    fi=0;idx=0
    # unpack
    streams=[]
    for fo,sz in zip(info['folders'],ssz):
        ps=info['packsizes'][pi];pi+=1
        raw=decode_folder(fo,b[pos:pos+ps]);pos+=ps
        o=0
        for s in sz: streams.append(raw[o:o+s]);o+=s
    si=0;ei=0
    for i,n in enumerate(names):
        p=os.path.join(out,n)
        if emptystream[i]:
            isdir = not (emptyfile and emptyfile[ei]); ei+=1
            if isdir: os.makedirs(p,exist_ok=True)
            else:
                os.makedirs(os.path.dirname(p),exist_ok=True);open(p,'wb').close()
        else:
            os.makedirs(os.path.dirname(p) or '.',exist_ok=True);open(p,'wb').write(streams[si]);si+=1
    print(len(names),'entries')
if __name__=='__main__': extract(sys.argv[1],sys.argv[2])
