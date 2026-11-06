/* Experimental DOD3 scene-feedback copy optimization. Opt in explicitly.
 * A snapshot is complete at the first copy in every submit. Subsequent
 * copies update the union of ALL writes since that copy, not the next
 * draw's sampled UV range (which can contain distortion offsets).
 * Unknown vertex programs and writes retain the full/scissor bounds.
 * No snapshot state survives a submit or depends on object lifetimes.
 */
typedef struct { double lo, hi; } EngInterval;
static u8 s_snap_skipped[ENG_MAX_RECORDS];
static int eng_snap_alpha_reuse_enabled(void)
{
    static int enabled=-1;
    if(enabled<0) {const char* e=getenv("RSX_SNAPSHOT_ALPHA_REUSE");enabled=e && !strcmp(e,"1");}
    return enabled;
}
static int eng_snap_reader_cull_enabled(void)
{
    static int enabled=-1;
    if(enabled<0) {const char* e=getenv("RSX_SNAPSHOT_READER_CULL");enabled=e && !strcmp(e,"1");}
    return enabled;
}
static int eng_snap_any_reuse_enabled(void)
{ return eng_snap_alpha_reuse_enabled() || eng_snap_reader_cull_enabled(); }

static const void* eng_snap_data(u32 offset, u32 bytes)
{
    const EngStage* st=eng_stage_at(offset);
    if(!bytes || !st || !st->mapped || offset<st->base ||
       (u64)(offset-st->base)+bytes>st->cap) return NULL;
    return (st->cpu_copy ? st->cpu_copy : st->mapped)+offset-st->base;
}
static EngInterval snap_iv(double x) { EngInterval r = {x,x}; return r; }
static EngInterval snap_round(double lo, double hi)
{
    if (!isfinite(lo) || !isfinite(hi) || lo < -0x1.fffffep127 || hi > 0x1.fffffep127) {
        EngInterval r={-INFINITY,INFINITY};return r;
    }
    /* Float32 operations, including contraction and flush-to-zero. Wider
     * than one float ULP; each intermediate is enclosed independently. */
    double e = fmax(fabs(lo), fabs(hi)) * 0x1p-21 + 0x1p-125;
    EngInterval r = {lo-e,hi+e}; return r;
}
static EngInterval snap_add(EngInterval a, EngInterval b)
{ return snap_round(a.lo+b.lo, a.hi+b.hi); }
static EngInterval snap_mul(EngInterval a, EngInterval b)
{
    if ((a.lo == 0 && a.hi == 0) || (b.lo == 0 && b.hi == 0)) return snap_iv(0);
    double p[4] = {a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
    return snap_round(fmin(fmin(p[0],p[1]),fmin(p[2],p[3])),
                      fmax(fmax(p[0],p[1]),fmax(p[2],p[3])));
}
static int eng_snap_draw_bounds_mode(const EngRecord* r, const EngObj* o, VkRect2D* rect, int box)
{
    VkViewport vp; eng_vp_sc(r, eng_obj_scaled(r->rt[0]), o->w,o->h,&vp,rect);
    if (!r->pipeline || r->pipeline > s_pipe_count || r->vbuf ||
        r->topology != RSX_TOPOLOGY_TRIANGLES || r->stride != 64 ||
        !s_pipe[r->pipeline-1].snapshot_linear_position || r->vs_cb_bytes < 8224 ||
        r->vertex_count > 4096) return 0;
    const float* c = eng_snap_data(r->vs_cb_off,r->vs_cb_bytes);
    const unsigned char* v = eng_snap_data(r->vb_off,r->vertex_count*r->stride);
    if (!c || !v) return 0;
    const int rows[]={0,1,2,3,200,201,202,203,467,512,513};
    for (u32 k=0;k<sizeof rows/sizeof rows[0];k++)
        for(int j=0;j<4;j++) if(!isfinite(c[rows[k]*4+j])) return 0;
    EngInterval object_box[4];
    if (box) {
        for(int j=0;j<4;j++) {object_box[j].lo=INFINITY;object_box[j].hi=-INFINITY;}
        for(u32 i=0;i<r->vertex_count;i++) {
            float a[4];memcpy(a,v+i*r->stride,sizeof a);
            for(int j=0;j<4;j++) {
                if(!isfinite(a[j])) return 0;
                object_box[j].lo=fmin(object_box[j].lo,a[j]);
                object_box[j].hi=fmax(object_box[j].hi,a[j]);
            }
        }
    }
    double x0=INFINITY,y0=INFINITY,x1=-INFINITY,y1=-INFINITY;
    for (u32 i=0;i<(box?1:r->vertex_count);i++) {
        EngInterval a[4];
        if(box) memcpy(a,object_box,sizeof a);
        else {
            float point[4];memcpy(point,v+i*r->stride,sizeof point);
            for(int j=0;j<4;j++) {if(!isfinite(point[j])) return 0;a[j]=snap_iv(point[j]);}
        }
        EngInterval world[4],clip[4],pos[4];
        for (int j=0;j<4;j++) {
            world[j]=snap_mul(a[1],snap_iv(c[201*4+j]));
            world[j]=snap_add(snap_mul(a[0],snap_iv(c[200*4+j])),world[j]);
            world[j]=snap_add(snap_mul(a[2],snap_iv(c[202*4+j])),world[j]);
            world[j]=snap_add(snap_mul(a[3],snap_iv(c[203*4+j])),world[j]);
        }
        for (int j=0;j<4;j++) {
            if(j==2) continue; /* depth clipping only reduces XY coverage */
            clip[j]=snap_mul(world[1],snap_iv(c[4+j]));
            clip[j]=snap_add(snap_mul(world[0],snap_iv(c[j])),clip[j]);
            clip[j]=snap_add(snap_mul(world[2],snap_iv(c[8+j])),clip[j]);
            clip[j]=snap_add(snap_mul(world[3],snap_iv(c[12+j])),clip[j]);
        }
        /* o0.xy = clip.xy + clip.w*c467.z*c467.xy; o0.zw=clip.zw.
         * Out.pos = o0*vp_posscale + o0.w*vp_posoffset (xyz only). */
        pos[3]=clip[3];
        for (int j=0;j<2;j++) {
            EngInterval q=clip[j];
            if (j<2) q=snap_add(snap_mul(snap_mul(clip[3],snap_iv(c[467*4+2])),snap_iv(c[467*4+j])),q);
            pos[j]=snap_add(snap_mul(q,snap_iv(c[512*4+j])),snap_mul(clip[3],snap_iv(c[513*4+j])));
        }
        if (!(pos[3].lo>1e-4) || !isfinite(pos[3].hi)) return 0;
        for (int j=0;j<2;j++) {
            double a0=pos[j].lo/pos[3].lo,a1=pos[j].lo/pos[3].hi;
            double a2=pos[j].hi/pos[3].lo,a3=pos[j].hi/pos[3].hi;
            double lo=fmin(fmin(a0,a1),fmin(a2,a3)),hi=fmax(fmax(a0,a1),fmax(a2,a3));
            if (!isfinite(lo) || !isfinite(hi)) return 0;
            double origin=j?vp.y:vp.x,extent=j?vp.height:vp.width;
            double l=origin+(lo+1)*extent*.5,h=origin+(hi+1)*extent*.5;
            if(j) { y0=fmin(y0,fmin(l,h)); y1=fmax(y1,fmax(l,h)); }
            else { x0=fmin(x0,fmin(l,h)); x1=fmax(x1,fmax(l,h)); }
        }
    }
    /* Include raster subpixel rounding and viewport conversion error. */
    double l=fmax(rect->offset.x,floor(x0)-2),t=fmax(rect->offset.y,floor(y0)-2);
    double rr=fmin(rect->offset.x+rect->extent.width,ceil(x1)+2);
    double bb=fmin(rect->offset.y+rect->extent.height,ceil(y1)+2);
    if (rr<=l || bb<=t) { rect->extent.width=rect->extent.height=0; return 1; }
    rect->offset.x=(int32_t)l;rect->offset.y=(int32_t)t;
    rect->extent.width=(u32)(rr-l);rect->extent.height=(u32)(bb-t);
    return 1;
}
/* One interval transform of the object-space vertex box encloses every
 * transformed vertex. It may lose correlations, so refine with the original
 * per-vertex calculation for broad or uncertain boxes. Never use an estimate
 * that can exclude a write. The arithmetic enclosure is unchanged. */
static int eng_snap_draw_bounds(const EngRecord* r, const EngObj* o, VkRect2D* rect)
{
    static int box=-1;
    if(box<0) {const char* e=getenv("RSX_SNAPSHOT_BOUNDS_BOX");box=e && !strcmp(e,"1");}
    if(box && eng_snap_draw_bounds_mode(r,o,rect,1) &&
       (u64)rect->extent.width*rect->extent.height <= (u64)o->w*o->h/64u) return 1;
    return eng_snap_draw_bounds_mode(r,o,rect,0);
}

static int eng_snap_copy_rect(u32 index, EngObj* src, VkRect2D* rect)
{
    static int enabled=-1;
    if(enabled<0) { const char* e=getenv("RSX_SNAPSHOT_DIRTY_RECTS"); enabled=e && !strcmp(e,"1"); }
    if(!enabled || s_ms_n>1 || s_hdr_on) return 0;
    const EngRecord* copy=&s_rec[index];
    int previous=-1;
    for(int k=(int)index-1;k>=0;k--) {
        const EngRecord* r=&s_rec[k];
        if((!eng_snap_any_reuse_enabled() || !s_snap_skipped[k]) && r->kind==ENG_REC_COLOR_COPY && eng_owner(r->depth)==src && r->resolve_dst==copy->resolve_dst) {previous=k;break;}
    }
    if(previous<0) return 0;
    int have=0; u32 x0=0,y0=0,x1=0,y1=0;
    for(u32 k=(u32)previous+1;k<index;k++) {
        const EngRecord* r=&s_rec[k];
        if(r->kind==ENG_REC_DRAW) {
            int writes=0;for(u32 a=0;a<r->nrt;a++) if(eng_owner(r->rt[a])==src) writes=1;
            if(!writes) continue;
            VkRect2D b;eng_snap_draw_bounds(r,src,&b);
            if(!b.extent.width || !b.extent.height) continue;
            u32 l=b.offset.x,t=b.offset.y,rr=l+b.extent.width,bb=t+b.extent.height;
            if(!have) {x0=l;y0=t;x1=rr;y1=bb;have=1;}
            else {if(l<x0)x0=l;if(t<y0)y0=t;if(rr>x1)x1=rr;if(bb>y1)y1=bb;}
        } else if ((r->kind==ENG_REC_CLEAR_COLOR && eng_owner(r->rt[0])==src) ||
                   (r->kind==ENG_REC_UPLOAD && eng_owner(r->depth)==src) ||
                   (r->kind==ENG_REC_COLOR_COPY && eng_owner(r->resolve_dst)==src) ||
                   (r->kind==ENG_REC_DEPTH_RESOLVE && eng_owner(r->resolve_dst)==src)) return 0;
    }
    rect->offset.x=x0;rect->offset.y=y0;
    rect->extent.width=have?x1-x0:0;rect->extent.height=have?y1-y0:0;
    return 1;
}
