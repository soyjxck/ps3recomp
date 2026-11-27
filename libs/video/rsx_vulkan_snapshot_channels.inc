/* Opt-in reuse for two reviewed soft-particle fragment programs. They read
 * only logical alpha from unit 2. Keep the first snapshot of a submit, and
 * refresh before any consumer or write we cannot prove alpha-independent.
 * RGB can be stale only while an alpha-only consumer is being drawn: the
 * next actual refresh includes ALL writes since the last executed copy. */
static u32 s_ch_generation[ENG_MAX_OBJECTS],s_ch_source[ENG_MAX_OBJECTS],s_ch_at_copy[ENG_MAX_OBJECTS];
static u8 s_ch_valid[ENG_MAX_OBJECTS];
static u32 s_ch_skipped,s_ch_copies;

static void eng_snap_channels_prepare(void)
{
    if(!eng_snap_any_reuse_enabled()) return;
    memset(s_snap_skipped,0,sizeof s_snap_skipped);
    memset(s_ch_generation,0,sizeof s_ch_generation);
    memset(s_ch_valid,0,sizeof s_ch_valid);
    s_ch_skipped=s_ch_copies=0;
}
static int eng_snap_particle_zero_alpha(const EngRecord* r,const EngPipeline* p)
{
    if(!p->snapshot_alpha_kind) return 0;
    const u32 index=p->snapshot_alpha_kind==1 ? 25u : p->snapshot_alpha_kind==2 ? 26u : 7u;
    if(r->ps_cb_bytes<(index+1)*16u) return 0;
    const float* c=eng_snap_data(r->ps_cb_off,r->ps_cb_bytes);
    /* The reviewed programs' last assignment to target0 alpha is this
     * constant. Positive zero avoids any 0*Inf/NaN source term in blending. */
    return c && c[index*4]==0.0f && !signbit(c[index*4]);
}
static int eng_snap_draw_preserves_particle_alpha(const EngRecord* r,const EngPipeline* p,u32 target)
{
    if(!(p->rs.color_mask&0xFF000000u)) return 1;
    /* Finite alpha is unchanged. Nonfinite canonicalizations are equivalent
     * only for the bounded, point-sampled particle readers checked below. */
    return target==0 && p->rs.blend_enable && p->rs.sf_a==0 && p->rs.df_a==1 &&
           p->rs.eq_a==0x8006 && eng_snap_particle_zero_alpha(r,p);
}
static void eng_snap_channels_bump(EngObj* o)
{
    if(!o) return;
    const u32 id=(u32)(o-s_obj);
    s_ch_generation[id]++;
    s_ch_valid[id]=0;
}
static void eng_snap_channels_note_write(const EngRecord* r)
{
    if(!eng_snap_any_reuse_enabled()) return;
    if(r->kind==ENG_REC_DRAW) {
        const EngPipeline* p=r->pipeline && r->pipeline<=s_pipe_count ? &s_pipe[r->pipeline-1] : NULL;
        for(u32 k=0;k<r->nrt;k++) {
            EngObj* o=eng_owner(r->rt[k]);if(!o) continue;
            /* A snapshot used as an attachment must be refreshed in full. */
            s_ch_valid[(u32)(o-s_obj)]=0;
            if(!p || !eng_snap_draw_preserves_particle_alpha(r,p,k)) s_ch_generation[(u32)(o-s_obj)]++;
        }
    } else if(r->kind==ENG_REC_CLEAR_COLOR) eng_snap_channels_bump(eng_owner(r->rt[0]));
    else if(r->kind==ENG_REC_UPLOAD) eng_snap_channels_bump(eng_owner(r->depth));
    else if(r->kind==ENG_REC_DEPTH_RESOLVE) eng_snap_channels_bump(eng_owner(r->resolve_dst));
    /* Colour copies are noted only if executed, not if reused. */
}
/* The reviewed linear VS exports the same positive clip W to tc5.w and
 * position.w. Bound all vertices (and hence perspective interpolation) away
 * from zero. This also keeps the depth fade zero for nonfinite background
 * depth values and their FP16 blend canonicalizations. */
static int eng_snap_particle_w_class(const EngRecord* r,const EngPipeline* p)
{
    if(!p->snapshot_linear_position || r->vbuf || r->stride!=64 || !r->vertex_count || r->vertex_count>4096 || r->vs_cb_bytes<8224) return 0;
    const float* c=eng_snap_data(r->vs_cb_off,r->vs_cb_bytes);
    const u8* v=eng_snap_data(r->vb_off,r->vertex_count*r->stride);
    if(!c || !v) return 0;
    EngInterval a[4];for(int j=0;j<4;j++) {a[j].lo=INFINITY;a[j].hi=-INFINITY;}
    for(u32 n=0;n<r->vertex_count;n++) {
        float x[4];memcpy(x,v+n*r->stride,sizeof x);
        for(int j=0;j<4;j++) {if(!isfinite(x[j])) return 0;a[j].lo=fmin(a[j].lo,x[j]);a[j].hi=fmax(a[j].hi,x[j]);}
    }
    EngInterval world[4];
    for(int j=0;j<4;j++) {
        for(int k=0;k<4;k++) if(!isfinite(c[(200+k)*4+j])) return 0;
        world[j]=snap_mul(a[1],snap_iv(c[201*4+j]));
        world[j]=snap_add(snap_mul(a[0],snap_iv(c[200*4+j])),world[j]);
        world[j]=snap_add(snap_mul(a[2],snap_iv(c[202*4+j])),world[j]);
        world[j]=snap_add(snap_mul(a[3],snap_iv(c[203*4+j])),world[j]);
    }
    for(int k=0;k<4;k++) if(!isfinite(c[k*4+3])) return 0;
    EngInterval w=snap_mul(world[1],snap_iv(c[7]));
    w=snap_add(snap_mul(world[0],snap_iv(c[3])),w);
    w=snap_add(snap_mul(world[2],snap_iv(c[11])),w);
    w=snap_add(snap_mul(world[3],snap_iv(c[15])),w);
    if(!isfinite(w.lo) || !isfinite(w.hi)) return 0;
    if(w.hi<0.0) return -1;
    return w.lo>=1.0 ? 1 : 0;
}
/* Do not leave a stale snapshot for an unreviewed reader or a later
 * submit. Require another explicit copy before any subsequent use; the
 * terminal copy in every chain is therefore always executed. */
static int eng_snap_single_reader(u32 i,EngObj* dst)
{
    for(u32 j=i+2;j<s_rec_count;j++) {
        const EngRecord* r=&s_rec[j];
        if(r->kind==ENG_REC_COLOR_COPY && eng_owner(r->resolve_dst)==dst)
            return eng_owner(r->depth)!=dst;
        if(r->kind==ENG_REC_DRAW) {
            for(u32 u=0;u<16;u++) if(eng_owner(r->tex[u])==dst) return 0;
            for(u32 u=0;u<4;u++) if(eng_owner(r->vtex[u])==dst) return 0;
            for(u32 k=0;k<r->nrt;k++) if(eng_owner(r->rt[k])==dst) return 0;
            if(eng_owner(r->depth)==dst) return 0;
        } else if(eng_owner(r->depth)==dst || eng_owner(r->resolve_dst)==dst ||
                  (r->kind==ENG_REC_CLEAR_COLOR && eng_owner(r->rt[0])==dst)) return 0;
    }
    return 0;
}
static int eng_snap_channels_can_skip(u32 i,EngObj* src,EngObj* dst)
{
    if(!eng_snap_any_reuse_enabled() || s_ms_n>1 || s_hdr_on || src==dst ||
       src->kind!=OBJ_COLOR || dst->kind!=OBJ_SNAPSHOT || src->fmt!=VK_FORMAT_R16G16B16A16_SFLOAT ||
       dst->fmt!=src->fmt || src->w!=dst->w || src->h!=dst->h || dst->mips!=1 || dst->faces!=1 || i+1>=s_rec_count) return 0;
    const u32 sid=(u32)(src-s_obj),did=(u32)(dst-s_obj);
    if(!s_ch_valid[did] || s_ch_source[did]!=sid || !eng_snap_single_reader(i,dst)) return 0;
    const EngRecord* d=&s_rec[i+1];
    if(d->kind!=ENG_REC_DRAW || !d->pipeline || d->pipeline>s_pipe_count || d->nrt!=1 || eng_owner(d->rt[0])!=src) return 0;
    const EngPipeline* p=&s_pipe[d->pipeline-1];
    for(u32 u=0;u<4;u++) if(eng_owner(d->vtex[u])==dst) return 0;
    const int wclass=eng_snap_particle_w_class(d,p);
    /* Every vertex has negative clip W: no fragment can consume the copy.
     * The draw and its query remain; a later real consumer still refreshes
     * every write since the last executed copy. */
    if(eng_snap_reader_cull_enabled() && wclass<0) return 1;
    if(!eng_snap_alpha_reuse_enabled() || wclass!=1 || s_ch_at_copy[did]!=s_ch_generation[sid] ||
       (!p->snapshot_alpha_kind || p->snapshot_alpha_kind>2) || !eng_snap_particle_zero_alpha(d,p) || d->ps_cb_bytes<80) return 0;
    const float* c=eng_snap_data(d->ps_cb_off,d->ps_cb_bytes);
    if(!c || !isfinite(c[17]) || c[17]==0.0f || fabs(c[17])>4.0f) return 0;
    const float bias2=c[p->snapshot_alpha_kind==1 ? 17*4+1 : 19*4];
    const float scale=c[p->snapshot_alpha_kind==1 ? 22*4+3 : 23*4];
    const float lod=c[p->snapshot_alpha_kind==1 ? 2*4 : 3*4];
    if(!isfinite(bias2) || fabs(bias2)>4.0f || !isfinite(c[20*4]) || fabs(c[20*4])>1024.0f ||
       !isfinite(scale) || scale<=0 || !isfinite(lod)) return 0;
    /* Point sampling avoids mixtures of positive/negative infinite texels.
     * With |bias|<=8, |near|<=1024 and W>=1, infinite background depth and
     * FP16 blend canonicalizations all yield zero particle depth fade. */
    if(d->samp[2]>=0 && (s_samp[d->samp[2]].key&3u)) return 0;
    int found=0;
    for(u32 u=0;u<16;u++) if(eng_owner(d->tex[u])==dst) {
        const EngObj* v=eng_obj(d->tex[u]);
        if(u!=2 || !v || (v->sample_alpha!=VK_COMPONENT_SWIZZLE_IDENTITY && v->sample_alpha!=VK_COMPONENT_SWIZZLE_A)) return 0;
        found=1;
    }
    return found;
}
static void eng_snap_channels_mark_copy(EngObj* src,EngObj* dst,int physical)
{
    if(!eng_snap_any_reuse_enabled()) return;
    const u32 sid=(u32)(src-s_obj),did=(u32)(dst-s_obj);
    const u32 generation=s_ch_generation[sid];
    eng_snap_channels_bump(dst);
    s_ch_source[did]=sid;s_ch_at_copy[did]=generation;s_ch_valid[did]=1;
    if(physical) s_ch_copies++;
}
static void eng_snap_channels_finish(void)
{
    static int stats=-1;
    if(stats<0) {const char* e=getenv("RSX_SNAPSHOT_CHANNEL_STATS");stats=e && !strcmp(e,"1");}
    if(stats && eng_snap_any_reuse_enabled() && (s_ch_copies || s_ch_skipped))
        fprintf(stderr,"[snapshot-channels] copies=%u reused=%u\n",s_ch_copies,s_ch_skipped);
}
