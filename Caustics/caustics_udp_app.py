#!/usr/bin/env python3
from __future__ import annotations
import argparse, array, socket, struct, time, threading
from collections import defaultdict
import tkinter as tk
from tkinter import ttk
import moderngl, numpy as np, pygame

DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720
VERT = """
#version 330
in vec2 in_pos;
void main(){ gl_Position=vec4(in_pos,0.0,1.0); }
"""
PREFIX = """
#version 330
uniform vec3 iResolution; uniform float iTime; uniform float iTimeDelta; uniform int iFrame; uniform vec4 iMouse;
uniform sampler2D iChannel0; uniform sampler2D iChannel1; uniform sampler2D iChannel2; uniform sampler2D iChannel3;
out vec4 fragColor;
void mainImage(out vec4 fragColor, in vec2 fragCoord);
"""
SUFFIX = """
void main(){ vec4 col=vec4(0.0); mainImage(col, gl_FragCoord.xy); fragColor=col; }
"""
def shader(src):
    lines=[]
    for line in src.splitlines():
        s=line.strip()
        if s.startswith('#version') or s.startswith('precision '): continue
        lines.append(line)
    return PREFIX+'\n'+'\n'.join(lines)+'\n'+SUFFIX

BUFFER_A = r'''
#define MAX_OBJECTS __MAX_OBJECTS__
vec2 R;
uniform int uObjectCount;
uniform int uUseMouse;
uniform int uCenterEmitter;
uniform float uDissipation;
uniform float uObjectRadius;
uniform float uObjectStrength;
uniform float uMouseRadius;
uniform float uMouseStrength;
uniform float uVelocityMultiplier;
float ln(vec2 p, vec2 a, vec2 b){ vec2 ab=b-a; float d=max(dot(ab,ab),1e-6); return length(p-a-ab*clamp(dot(p-a,ab)/d,0.,1.)); }
vec4 A(vec2 U){return texture(iChannel0,U/R);} vec4 B(vec2 U){return texture(iChannel1,U/R);} vec4 T(vec2 U){return A(B(U).xy);}
void injectSegment(inout vec4 C, vec2 U, vec2 now, vec2 prev, float radius, float strength){
    vec2 delta=(now-prev)*uVelocityMultiplier; float l=ln(U,now,prev);
    if(l<radius){ float falloff=(radius-l)/radius; C += vec4(strength*falloff*delta/R.y,0.,0.); }
}
void mainImage(out vec4 C, in vec2 U){
   R=iResolution.xy; C=T(U);
   vec4 n=T(U+vec2(0,1)), e=T(U+vec2(1,0)), s=T(U-vec2(0,1)), w=T(U-vec2(1,0));
   C.x -= 0.25*(e.z-w.z+(n.w*C.w-s.w*C.w));
   C.y -= 0.25*(n.z-s.z+(e.w*C.w-w.w*C.w));
   C.z  = 0.25*((s.y-n.y+w.x-e.x)+(n.z+e.z+s.z+w.z));
   C.w  = 0.25*((s.x-n.x+w.y-e.y)-(n.w+e.w+s.w+w.w));
   if(uCenterEmitter==1) C.xy += exp(-length(U.xy-0.5*R))*(0.9*vec2(sin(0.2*iTime),cos(0.2*iTime))-C.xy);
   C.xy *= uDissipation; C.zw *= uDissipation;
   if(U.x<1.||R.x-U.x<1.) C.xy*=0.; if(U.y<1.||R.y-U.y<1.) C.xy*=0.; if(iFrame<1) C=vec4(0);
   for(int i=0;i<MAX_OBJECTS;i++){
       if(i>=uObjectCount) break;
       vec4 obj=texelFetch(iChannel3,ivec2(i,0),0);
       if(obj.x>=0.&&obj.y>=0.&&obj.z>=0.&&obj.w>=0.) injectSegment(C,U,obj.xy,obj.zw,uObjectRadius,uObjectStrength);
   }
   if(uUseMouse==1 && iMouse.z>0.0) injectSegment(C,U,iMouse.xy,iMouse.zw,uMouseRadius,uMouseStrength);
}
'''
BUFFER_B = r'''
vec2 R; vec4 A(vec2 U){return texture(iChannel0,U/R);} vec4 B(vec2 U){return texture(iChannel1,U/R);}
void mainImage(out vec4 C, in vec2 U){
    R=iResolution.xy;
    float n=A(U+vec2(0,1)).z,e=A(U+vec2(1,0)).z,s=A(U-vec2(0,1)).z,w=A(U-vec2(1,0)).z;
    #define N 2.
    for(float i=0.;i<N;i++) U-=A(U).xy/N;
    C.xy=U; C.zw=vec2(e-w,n-s);
}
'''
BUFFER_C = r'''
vec2 R;
#define D 5
uniform float uCausticPersistence;
float ln(vec3 p, vec3 a, vec3 b){return length(p-a-(b-a)*dot(p-a,b-a)/max(dot(b-a,b-a),1e-6));}
vec4 A(vec2 U){return texture(iChannel0,U/R);} vec4 B(vec2 U){return texture(iChannel1,U/R);} vec4 C(vec2 U){return texture(iChannel2,U/R);}
float dI(vec2 U, vec3 me, vec3 light, float mu){ vec3 r=vec3(U,100); vec3 n=normalize(vec3(B(r.xy).zw,mu)); vec3 li=reflect((r-light),n); float len=ln(me,r,li); return 2.5*exp(-1.7*len); }
float I(vec2 U, vec3 me, vec3 light, float mu){ float intensity=0.; for(int x=-D;x<=D;x++) for(int y=-D;y<=D;y++) intensity+=dI(U+vec2(x,y),me,light,0.1*mu); return intensity; }
vec3 S(vec2 U, vec3 me, vec3 light, float mu){ return I(U,me,light,mu)*vec3(exp(-(mu-0.5)*(mu-0.5)),exp(-(mu-1.0)*(mu-1.0)),exp(-(mu-1.4)*(mu-1.4))); }
void mainImage(out vec4 Q, in vec2 U){ R=iResolution.xy; vec3 light=vec3(0.5*R,1e5), me=vec3(U,0); vec3 c=vec3(0); for(float mu=.4;mu<=1.6;mu+=.4) c+=S(U,me,light,mu); Q=vec4(0.03*c,1); if(R.x>=800.) Q=mix(Q,C(U),uCausticPersistence); }
'''
IMAGE = r'''
vec2 R;
uniform vec2 uSimResolution;
uniform int uUpsampleMode;
uniform int uDebugObjects;
uniform int uObjectCount;
vec2 toSim(vec2 U){ return U * uSimResolution / R; }
float sinc(float x){ x=abs(x); if(x<1e-5) return 1.0; float pix=3.141592653589793*x; return sin(pix)/pix; }
float lanczos2(float x){ x=abs(x); if(x>=2.0) return 0.0; return sinc(x)*sinc(x/2.0); }
float catmull(float x){ x=abs(x); float x2=x*x; float x3=x2*x; if(x<1.0) return 1.5*x3-2.5*x2+1.0; if(x<2.0) return -0.5*x3+2.5*x2-4.0*x+2.0; return 0.0; }
vec4 sampleLinear(sampler2D tex, vec2 P){ return texture(tex,P/uSimResolution); }
vec4 sampleFiltered(sampler2D tex, vec2 P, int mode){
    if(mode==0) return sampleLinear(tex,P);
    vec2 base=floor(P-0.5)+0.5; vec4 sum=vec4(0.0); float wsum=0.0;
    if(mode==1){
        for(int j=-1;j<=2;j++) for(int i=-1;i<=2;i++){
            vec2 sampleP=base+vec2(i,j); float w=catmull(P.x-sampleP.x)*catmull(P.y-sampleP.y);
            sampleP=clamp(sampleP,vec2(0.5),uSimResolution-vec2(0.5)); sum+=texture(tex,sampleP/uSimResolution)*w; wsum+=w;
        }
    } else {
        for(int j=-1;j<=2;j++) for(int i=-1;i<=2;i++){
            vec2 sampleP=base+vec2(i,j); float w=lanczos2(P.x-sampleP.x)*lanczos2(P.y-sampleP.y);
            sampleP=clamp(sampleP,vec2(0.5),uSimResolution-vec2(0.5)); sum+=texture(tex,sampleP/uSimResolution)*w; wsum+=w;
        }
    }
    return sum/max(wsum,1e-6);
}
vec4 A(vec2 U){return sampleFiltered(iChannel0,toSim(U),uUpsampleMode);} vec4 B(vec2 U){return sampleFiltered(iChannel1,toSim(U),uUpsampleMode);} vec4 C(vec2 U){return sampleFiltered(iChannel2,toSim(U),uUpsampleMode);}
float circle(vec2 p, vec2 c, float r){ return smoothstep(r,r-2.0,length(p-c)); }
void mainImage(out vec4 Q, in vec2 U){
    R=iResolution.xy; vec2 M=iMouse.z>0.?iMouse.xy:0.5*R; vec2 r=2.*(U-M)/R.y; r=r/sqrt(max(length(r),1e-6));
    Q=vec4(0); for(float i=1.;i<10.;i++){ vec4 c=C(U-i*r); Q+=c*c*exp(-.2*i); } Q=mix(C(U),.8*Q*exp(-1.5*length(r)),.5);
    if(uDebugObjects==1){
        for(int i=0;i<__MAX_OBJECTS__;i++){
            if(i>=uObjectCount) break; vec4 obj=texelFetch(iChannel3,ivec2(i,0),0);
            if(obj.x>=0.0&&obj.y>=0.0){ vec2 p=obj.xy*R/uSimResolution; float m=circle(U,p,6.0); Q.rgb=mix(Q.rgb,vec3(1.0,0.15,0.05),m); }
        }
    }
}
'''
class UDPObjectReceiver:
    def __init__(self, port, max_objects, timeout_seconds=.5, recv_buffer_bytes=4*1024*1024):
        self.max_objects=max_objects
        self.timeout_seconds=timeout_seconds
        self.tracks={}
        self.lock=threading.Lock()
        self.running=True

        self.stats_packets=0
        self.stats_short=0
        self.stats_updates=defaultdict(int)
        self.stats_latest={}
        self.stats_last_print=time.perf_counter()

        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(recv_buffer_bytes))
        except OSError:
            pass
        self.sock.bind(('0.0.0.0',port))
        self.sock.settimeout(0.25)

        self.thread=threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

        print(f'Listening for UDP objects on 0.0.0.0:{port} in dedicated receiver thread')
        try:
            actual_buf = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print(f'UDP receive buffer: {actual_buf} bytes')
        except OSError:
            pass

    def close(self):
        self.running=False
        try:
            self.sock.close()
        except OSError:
            pass

    def _recv_loop(self):
        recsize=struct.calcsize('<Hfff')
        while self.running:
            try:
                packet,_=self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            now=time.perf_counter()

            if len(packet)<1:
                continue

            count=packet[0]
            expected=1+count*recsize

            with self.lock:
                self.stats_packets += 1

                if len(packet)<expected:
                    self.stats_short += 1
                    continue

                off=1
                for _ in range(count):
                    oid,x,y,rot=struct.unpack_from('<Hfff',packet,off)
                    off+=recsize

                    x=min(max(float(x),0.),1.)
                    y=min(max(float(y),0.),1.)

                    self.stats_updates[oid] += 1
                    self.stats_latest[oid] = (x,y,float(rot),now)

                    p=self.tracks.get(oid)
                    if p is None:
                        self.tracks[oid]={
                            'x':x,'y':y,
                            'px':x,'py':y,
                            'accum_dx':0.0,'accum_dy':0.0,
                            'rot':float(rot),
                            'last':now
                        }
                    else:
                        dx=x-p['x']
                        dy=y-p['y']
                        p['accum_dx']=p.get('accum_dx',0.0)+dx
                        p['accum_dy']=p.get('accum_dy',0.0)+dy
                        p.update(px=p['x'],py=p['y'],x=x,y=y,rot=float(rot),last=now)

                stale=[oid for oid,t in self.tracks.items() if now-t['last']>self.timeout_seconds]
                for oid in stale:
                    del self.tracks[oid]

    def poll(self):
        # Compatibility no-op. The receiver thread is doing the work.
        pass

    def debug_print(self, interval=1.0):
        now=time.perf_counter()
        with self.lock:
            if now-self.stats_last_print < interval:
                return
            dt=max(now-self.stats_last_print,1e-6)
            active=sorted(self.tracks.keys())
            print("="*72)
            print(f"UDP packets/sec: {self.stats_packets/dt:7.1f} | objects: {len(active)} {active[:24]}")
            print(f"short packets: {self.stats_short}")
            for oid in active[:24]:
                hz=self.stats_updates.get(oid,0)/dt
                x,y,rot,t=self.stats_latest.get(oid,(0,0,0,0))
                age=now-t
                tr=self.tracks.get(oid,{})
                adx=tr.get('accum_dx',0.0)
                ady=tr.get('accum_dy',0.0)
                print(f"id {oid:5d} | {hz:6.1f} Hz | age {age:5.3f}s | x {x: .4f} y {y: .4f} | accum dx {adx:+.5f} dy {ady:+.5f} | rot {rot:+.3f}")
            if len(active)>24:
                print(f"... {len(active)-24} more ids")

            self.stats_packets=0
            self.stats_short=0
            self.stats_updates.clear()
            self.stats_last_print=now

    def make_texture_data(self,w,h):
        arr=np.full((1,self.max_objects,4),-1.,dtype=np.float32)

        with self.lock:
            active=sorted(self.tracks.items(),key=lambda kv:kv[0])[:self.max_objects]

            for i,(_,o) in enumerate(active):
                px=o['x']-o.get('accum_dx',0.0)
                py=o['y']-o.get('accum_dy',0.0)
                arr[0,i,:]=(o['x']*w,(1-o['y'])*h,px*w,(1-py)*h)

                o['accum_dx']=0.0
                o['accum_dy']=0.0

            count=len(active)

        return arr,count

class ControlPanel:
    def __init__(self, args, on_reset):
        self.args = args
        self.on_reset = on_reset
        self.root = tk.Tk()
        self.root.title("Caustics Controls")
        self.root.geometry("420x650")
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self.visible = True
        self._building = True
        self._vars = {}
        self._make_slider("sim_scale", "Sim scale", 0.25, 2.0, args.sim_scale, 0.01, True)
        self._make_slider("speed", "Speed / solver steps", 0.05, 5.0, args.speed, 0.01, False)
        self._make_slider("velocity_multiplier", "Velocity multiplier", 0.0, 10.0, args.velocity_multiplier, 0.01, False)
        self._make_slider("dissipation", "Dissipation", 0.90, 1.01, args.dissipation, 0.0005, False)
        self._make_slider("caustic_persistence", "Caustic persistence", 0.0, 1.0, args.caustic_persistence, 0.005, False)
        self._make_slider("object_radius", "Object radius", 0.5, 60.0, args.object_radius, 0.1, False)
        self._make_slider("object_strength", "Object strength", 0.0, 100.0, args.object_strength, 0.1, False)
        self._make_slider("mouse_radius", "Mouse radius", 0.5, 60.0, args.mouse_radius, 0.1, False)
        self._make_slider("mouse_strength", "Mouse strength", 0.0, 100.0, args.mouse_strength, 0.1, False)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10, pady=10)

        self.center_var = tk.BooleanVar(value=not args.no_center_emitter)
        self.mouse_var = tk.BooleanVar(value=not args.no_mouse)
        self.debug_var = tk.BooleanVar(value=args.debug_objects)
        ttk.Checkbutton(self.root, text="Center emitter", variable=self.center_var, command=self._sync_bools).pack(anchor="w", padx=12)
        ttk.Checkbutton(self.root, text="Mouse input", variable=self.mouse_var, command=self._sync_bools).pack(anchor="w", padx=12)
        ttk.Checkbutton(self.root, text="Debug objects", variable=self.debug_var, command=self._sync_bools).pack(anchor="w", padx=12)

        ttk.Label(self.root, text="Upsample").pack(anchor="w", padx=12, pady=(12,0))
        self.upsample_var = tk.StringVar(value=args.upsample)
        for mode in ("linear", "catmull", "lanczos"):
            ttk.Radiobutton(self.root, text=mode, value=mode, variable=self.upsample_var, command=self._sync_upsample).pack(anchor="w", padx=24)

        ttk.Label(self.root, text="F1 hides/shows controls. R resets simulation.").pack(anchor="w", padx=12, pady=(16,0))
        self._building = False

    def _make_slider(self, attr, label, lo, hi, value, resolution, reset):
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", padx=10, pady=4)
        val_label = ttk.Label(frame)
        val_label.pack(anchor="w")
        var = tk.DoubleVar(value=float(value))
        self._vars[attr] = var

        def update_label():
            val_label.config(text=f"{label}: {var.get():.4g}")

        def on_change(_):
            setattr(self.args, attr, float(var.get()))
            update_label()
            if reset and not self._building:
                self.on_reset()

        update_label()
        scale = tk.Scale(frame, from_=lo, to=hi, orient="horizontal",
                         resolution=resolution, variable=var, showvalue=False,
                         command=on_change, length=360)
        scale.pack(fill="x")

    def _sync_bools(self):
        self.args.no_center_emitter = not bool(self.center_var.get())
        self.args.no_mouse = not bool(self.mouse_var.get())
        self.args.debug_objects = bool(self.debug_var.get())

    def _sync_upsample(self):
        self.args.upsample = self.upsample_var.get()

    def value(self, name):
        return float(self._vars[name].get())

    def flag_center_emitter(self):
        return bool(self.center_var.get())

    def flag_mouse(self):
        return bool(self.mouse_var.get())

    def flag_debug_objects(self):
        return bool(self.debug_var.get())

    def upsample(self):
        return self.upsample_var.get()

    def poll(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.visible = False

    def hide(self):
        if self.visible:
            self.root.withdraw()
            self.visible = False

    def toggle(self):
        if self.visible:
            self.hide()
        else:
            self.root.deiconify()
            self.visible = True

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--port',type=int,default=5005); ap.add_argument('--max-objects',type=int,default=256); ap.add_argument('--udp-recv-buffer',type=int,default=4*1024*1024); ap.add_argument('--no-mouse',action='store_true')
    ap.add_argument('--sim-scale',type=float,default=1.0); ap.add_argument('--speed',type=float,default=1.0); ap.add_argument('--dissipation',type=float,default=0.995); ap.add_argument('--caustic-persistence',type=float,default=0.5)
    ap.add_argument('--upsample',choices=['linear','catmull','lanczos'],default='catmull'); ap.add_argument('--no-center-emitter',action='store_true'); ap.add_argument('--object-radius',type=float,default=8.0); ap.add_argument('--object-strength',type=float,default=18.0); ap.add_argument('--mouse-radius',type=float,default=4.0); ap.add_argument('--mouse-strength',type=float,default=12.0); ap.add_argument('--debug-objects',action='store_true'); ap.add_argument('--velocity-multiplier',type=float,default=1.0); ap.add_argument('--udp-debug',action='store_true'); ap.add_argument('--udp-debug-interval',type=float,default=1.0)
    args=ap.parse_args()
    if args.max_objects<1: raise SystemExit('--max-objects must be >= 1')
    if args.sim_scale<=0: raise SystemExit('--sim-scale must be > 0')
    if not (0<=args.caustic_persistence<=1): raise SystemExit('--caustic-persistence must be between 0 and 1')
    pygame.init(); pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION,3); pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION,3); pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,pygame.GL_CONTEXT_PROFILE_CORE)
    w,h=DEFAULT_WIDTH,DEFAULT_HEIGHT; pygame.display.set_mode((w,h),pygame.OPENGL|pygame.DOUBLEBUF|pygame.RESIZABLE); pygame.display.set_caption('WdBGDm caustics - UDP object input v3.0 threaded UDP')
    ctx=moderngl.create_context(); ctx.enable_only(moderngl.NOTHING); recv=UDPObjectReceiver(args.port,args.max_objects, recv_buffer_bytes=args.udp_recv_buffer)
    vbo=ctx.buffer(array.array('f',[-1,-1,1,-1,-1,1,1,1]).tobytes())
    programs={'A':ctx.program(vertex_shader=VERT,fragment_shader=shader(BUFFER_A.replace('__MAX_OBJECTS__',str(args.max_objects)))),'B':ctx.program(vertex_shader=VERT,fragment_shader=shader(BUFFER_B)),'C':ctx.program(vertex_shader=VERT,fragment_shader=shader(BUFFER_C)),'Image':ctx.program(vertex_shader=VERT,fragment_shader=shader(IMAGE.replace('__MAX_OBJECTS__',str(args.max_objects))))}
    vaos={k:vao(ctx,p,vbo) for k,p in programs.items()}; empty=ctx.texture((1,1),4,dtype='f4'); empty.filter=(moderngl.NEAREST,moderngl.NEAREST)
    objtex=ctx.texture((args.max_objects,1),4,dtype='f4'); objtex.filter=(moderngl.NEAREST,moderngl.NEAREST); objtex.repeat_x=False; objtex.repeat_y=False
    def sim_size(w,h): return max(1,int(round(w*args.sim_scale))),max(1,int(round(h*args.sim_scale)))
    def makebufs(sw,sh): return {k:PingPong(ctx,(sw,sh)) for k in ['A','B','C']}
    sw,sh=sim_size(w,h); bufs=makebufs(sw,sh); reset_requested=False
    def request_reset():
        nonlocal reset_requested
        reset_requested=True
    panel=ControlPanel(args, request_reset)
    start=last=time.perf_counter(); frame=0; sim_accum=0.0; paused=False; ps=None; pt=0.; md=False; mxy=(0.,0.); mprev=(0.,0.); mclick=(0.,0.); run=True
    while run:
        now=time.perf_counter()
        for e in pygame.event.get():
            if e.type==pygame.QUIT: run=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE: run=False
                elif e.key==pygame.K_F1: panel.toggle()
                elif e.key==pygame.K_SPACE:
                    paused=not paused
                    if paused: ps=time.perf_counter()
                    else: pt+=time.perf_counter()-ps; ps=None
                elif e.key==pygame.K_r:
                    [b.clear() for b in bufs.values()]; start=last=time.perf_counter(); pt=0.; frame=0; sim_accum=0.0
            elif e.type==pygame.VIDEORESIZE:
                w,h=max(1,e.w),max(1,e.h); pygame.display.set_mode((w,h),pygame.OPENGL|pygame.DOUBLEBUF|pygame.RESIZABLE); [b.release() for b in bufs.values()]; sw,sh=sim_size(w,h); bufs=makebufs(sw,sh); frame=0
            elif e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
                md=True; x,y=e.pos; mxy=(float(x),float(h-y)); mprev=mxy; mclick=mxy
            elif e.type==pygame.MOUSEBUTTONUP and e.button==1:
                md=False; x,y=e.pos; mxy=(float(x),float(h-y))
            elif e.type==pygame.MOUSEMOTION:
                x,y=e.pos; mprev=mxy; mxy=(float(x),float(h-y))
        panel.poll()
        if reset_requested:
            [b.release() for b in bufs.values()]
            sw,sh=sim_size(w,h)
            bufs=makebufs(sw,sh)
            start=last=time.perf_counter()
            pt=0.; frame=0; sim_accum=0.0
            reset_requested=False
        recv.poll();
        if args.udp_debug: recv.debug_print(args.udp_debug_interval)
        data,count=recv.make_texture_data(sw,sh); data=apply_velocity_multiplier_to_object_data(data,panel.value('velocity_multiplier')); objtex.write(data.tobytes())
        if paused: raw_t=(ps or now)-start-pt; raw_dt=0.
        else: raw_t=now-start-pt; raw_dt=now-last; frame+=1
        last=now; speed=panel.value('speed'); t=raw_t*speed; dt=raw_dt*speed
        mouse_sim=(mxy[0]*sw/w,mxy[1]*sh/h,(mxy[0]*sw/w)-((mxy[0]-mprev[0])*sw/w)*panel.value('velocity_multiplier'),(mxy[1]*sh/h)-((mxy[1]-mprev[1])*sh/h)*panel.value('velocity_multiplier')) if md else (mxy[0]*sw/w,mxy[1]*sh/h,-abs(mclick[0]*sw/w),-abs(mclick[1]*sh/h))
        mouse_screen=(mxy[0],mxy[1],mprev[0],mprev[1]) if md else (mxy[0],mxy[1],-abs(mclick[0]),-abs(mclick[1]))
        sim_accum += 0.0 if paused else max(0.0, panel.value('speed'))
        sim_steps = int(sim_accum)
        sim_accum -= sim_steps
        sim_steps = min(sim_steps, 8)
        for _sim_step in range(sim_steps):
            ctx.viewport=(0,0,sw,sh)
            p=programs['B']; bufs['B'].dst_fbo.use(); ctx.viewport=(0,0,sw,sh); uniforms(p,sw,sh,t,dt,frame,mouse_sim); bind(p,'iChannel0',bufs['A'].src,0); bind(p,'iChannel1',bufs['B'].src,1); bind(p,'iChannel2',empty,2); bind(p,'iChannel3',empty,3); vaos['B'].render(moderngl.TRIANGLE_STRIP); bufs['B'].swap()
            p=programs['A']; bufs['A'].dst_fbo.use(); ctx.viewport=(0,0,sw,sh); uniforms(p,sw,sh,t,dt,frame,mouse_sim); p['uObjectCount'].value=int(count); p['uUseMouse'].value=1 if panel.flag_mouse() else 0; p['uCenterEmitter'].value=1 if panel.flag_center_emitter() else 0; p['uDissipation'].value=panel.value('dissipation'); p['uObjectRadius'].value=panel.value('object_radius'); p['uObjectStrength'].value=panel.value('object_strength'); p['uMouseRadius'].value=panel.value('mouse_radius'); p['uMouseStrength'].value=panel.value('mouse_strength'); p['uVelocityMultiplier'].value=1.0; bind(p,'iChannel0',bufs['A'].src,0); bind(p,'iChannel1',bufs['B'].src,1); bind(p,'iChannel2',empty,2); bind(p,'iChannel3',objtex,3); vaos['A'].render(moderngl.TRIANGLE_STRIP); bufs['A'].swap()
            p=programs['C']; bufs['C'].dst_fbo.use(); ctx.viewport=(0,0,sw,sh); uniforms(p,sw,sh,t,dt,frame,mouse_sim); p['uCausticPersistence'].value=panel.value('caustic_persistence'); bind(p,'iChannel0',bufs['A'].src,0); bind(p,'iChannel1',bufs['B'].src,1); bind(p,'iChannel2',bufs['C'].src,2); bind(p,'iChannel3',empty,3); vaos['C'].render(moderngl.TRIANGLE_STRIP); bufs['C'].swap()
        ctx.screen.use(); ctx.viewport=(0,0,w,h); ctx.clear(0,0,0,1); p=programs['Image']; uniforms(p,w,h,t,dt,frame,mouse_screen); p['uSimResolution'].value=(float(sw),float(sh)); p['uUpsampleMode'].value=upsample_mode(panel.upsample()); p['uDebugObjects'].value=1 if panel.flag_debug_objects() else 0; p['uObjectCount'].value=int(count); bind(p,'iChannel0',bufs['A'].src,0); bind(p,'iChannel1',bufs['B'].src,1); bind(p,'iChannel2',bufs['C'].src,2); bind(p,'iChannel3',objtex,3); vaos['Image'].render(moderngl.TRIANGLE_STRIP); pygame.display.flip()
    recv.close()
    pygame.quit()
if __name__=='__main__': main()


DISPLAY = r"""
vec2 R;
void mainImage(out vec4 Q, in vec2 U){
    R = iResolution.xy;
    Q = texture(iChannel0, U/R);
}
"""

