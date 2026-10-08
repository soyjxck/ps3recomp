/* ps3recomp - Vulkan backend, second half: encoding, submit, present, device.
 * Included by rsx_vulkan_engine.c (one translation unit). */

/* ---- barriers ---------------------------------------------------------------------- */

static VkImageMemoryBarrier s_bar[64]; static u32 s_nbar; static int s_bar_global;

static void bar_flush(void)
{
    if (!s_nbar && !s_bar_global) return;
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(s_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         s_bar_global ? 1 : 0, &mb, 0, NULL, s_nbar, s_bar);
    s_nbar = 0; s_bar_global = 0;
}

/* Queue a layout change for an image (whole image). */
static void img_layout(VkImage img, VkImageLayout* cur, VkImageAspectFlags aspect, VkImageLayout to)
{
    if (*cur == to) return;
    if (s_nbar == 64) bar_flush();
    VkImageMemoryBarrier* b = &s_bar[s_nbar++];
    memset(b, 0, sizeof *b);
    b->sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b->srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b->dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b->oldLayout = *cur; b->newLayout = to;
    b->srcQueueFamilyIndex = b->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b->image = img;
    b->subresourceRange.aspectMask = aspect;
    b->subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    b->subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    *cur = to;
}
static void obj_layout(EngObj* o, VkImageLayout to) { if (o) img_layout(o->img, &o->layout, obj_aspect(o), to); }
/* Does the object need a change to be in `to`? */
static int obj_needs(const EngObj* o, VkImageLayout to) { return o && o->layout != to; }

/* ---- rendering instances -------------------------------------------------------------- */

static int s_in_render;
static u32 s_cur_rt[RSX_BE_MAX_COLOR_TARGETS], s_cur_nrt, s_cur_depth, s_cur_tw, s_cur_th;
static VkPipeline s_cur_pso;
static VkDescriptorSet s_cur_set;
static u64 s_cur_set_key;
static VkPrimitiveTopology s_cur_topo;

static void render_end(void)
{
    if (!s_in_render) return;
    vkCmdEndRendering(s_cmd);
    s_in_render = 0;
    s_bar_global = 1;     /* the next pass may read or write what this one wrote */
}

/* Begin rendering into the given (already transitioned) attachments. */
static void render_begin(const u32* rt, u32 nrt, u32 depth, u32 tw, u32 th)
{
    bar_flush();
    VkRenderingAttachmentInfo ca[RSX_BE_MAX_COLOR_TARGETS];
    for (u32 k = 0; k < nrt; k++) {
        EngObj* o = eng_owner(rt[k]);
        memset(&ca[k], 0, sizeof ca[k]);
        ca[k].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        ca[k].imageView = o->att ? o->att : o->view;
        ca[k].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ca[k].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; ca[k].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    VkRenderingAttachmentInfo da = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    EngObj* z = eng_owner(depth);
    if (z) {
        da.imageView = z->att; da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea.extent.width = tw; ri.renderArea.extent.height = th;
    ri.layerCount = 1;
    ri.colorAttachmentCount = nrt; ri.pColorAttachments = ca;
    ri.pDepthAttachment = z ? &da : NULL; ri.pStencilAttachment = z ? &da : NULL;
    vkCmdBeginRendering(s_cmd, &ri);
    s_in_render = 1;
    for (u32 k = 0; k < RSX_BE_MAX_COLOR_TARGETS; k++) s_cur_rt[k] = k < nrt ? rt[k] : 0;
    s_cur_nrt = nrt; s_cur_depth = depth; s_cur_tw = tw; s_cur_th = th;
}

/* ---- descriptor sets ----------------------------------------------------------------- */

typedef struct {
    VkImageView fs_view[16]; VkSampler fs_smp[16];
    VkImageView vs_view[4];  VkSampler vs_smp[4];
    VkBuffer vs_buf, fs_buf;
} EngSetDesc;

static VkDescriptorSet eng_set(const EngSetDesc* d)
{
    const u64 key = fnv1a64(d, sizeof *d, 1469598103934665603ull);
    if (s_cur_set && key == s_cur_set_key) return s_cur_set;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = s_dpool[s_slot]; ai.descriptorSetCount = 1; ai.pSetLayouts = &s_dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(s_dev, &ai, &set) != VK_SUCCESS) {
        static int n = 0;
        if (n++ < 4) fprintf(stderr, "[rsx engine/vulkan] descriptor pool exhausted\n");
        return VK_NULL_HANDLE;
    }
    VkDescriptorBufferInfo bvs = { d->vs_buf, 0, ENG_UBO_RANGE }, bfs = { d->fs_buf, 0, ENG_UBO_RANGE };
    VkDescriptorImageInfo fsi[16], fsi1[15], vsi[4], fss[16], vss[4];
    for (u32 u = 0; u < 16; u++) {
        fsi[u].sampler = VK_NULL_HANDLE; fsi[u].imageView = d->fs_view[u]; fsi[u].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        fss[u].sampler = d->fs_smp[u]; fss[u].imageView = VK_NULL_HANDLE; fss[u].imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (u) fsi1[u - 1] = fsi[u];
    }
    for (u32 u = 0; u < 4; u++) {
        vsi[u].sampler = VK_NULL_HANDLE; vsi[u].imageView = d->vs_view[u]; vsi[u].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vss[u].sampler = d->vs_smp[u]; vss[u].imageView = VK_NULL_HANDLE; vss[u].imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }
    VkWriteDescriptorSet w[2 + 1 + 15 + 4 + 1 + 4];
    u32 n = 0;
#define WR(bind, elem, cnt, type, img, buf) do { VkWriteDescriptorSet* x = &w[n++]; memset(x, 0, sizeof *x); \
        x->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; x->dstSet = set; x->dstBinding = (bind); \
        x->dstArrayElement = (elem); x->descriptorCount = (cnt); x->descriptorType = (type); \
        x->pImageInfo = (img); x->pBufferInfo = (buf); } while (0)
    WR(RSX_SPV_BIND_VS_CB, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, NULL, &bvs);
    WR(RSX_SPV_BIND_FS_CB, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, NULL, &bfs);
    WR(RSX_SPV_BIND_FS_TEX, 0, 16, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, fsi, NULL);
    for (u32 u = 1; u < 16; u++) WR(RSX_SPV_BIND_FS_TEX + u, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &fsi1[u - 1], NULL);
    for (u32 u = 0; u < 4; u++) WR(RSX_SPV_BIND_VS_TEX + u, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &vsi[u], NULL);
    WR(RSX_SPV_BIND_FS_SAMP, 0, 16, VK_DESCRIPTOR_TYPE_SAMPLER, fss, NULL);
    for (u32 u = 0; u < 4; u++) WR(RSX_SPV_BIND_VS_SAMP + u, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &vss[u], NULL);
#undef WR
    vkUpdateDescriptorSets(s_dev, n, w, 0, NULL);
    s_cur_set = set; s_cur_set_key = key;
    return set;
}

/* ---- encoding ------------------------------------------------------------------------- */

static void eng_vp_sc(const EngRecord* r, int scaled, u32 tw, u32 th, VkViewport* vp, VkRect2D* sc)
{
    const float f = scaled ? s_scale : 1.0f;
    float x = r->vp[0] * f, y = r->vp[1] * f, w = r->vp[2] * f, h = r->vp[3] * f;
    if (w <= 0.0f || h <= 0.0f) { x = 0.0f; y = 0.0f; w = (float)tw; h = (float)th; }
    /* D3D's clip space: y up, flipped by a negative height. */
    vp->x = x; vp->y = y + h; vp->width = w; vp->height = -h; vp->minDepth = 0.0f; vp->maxDepth = 1.0f;
    u32 sx = r->sc[0], sy = r->sc[1], sw = r->sc[2], sh = r->sc[3];
    if (sw && sh && scaled) sc_rect(&sx, &sy, &sw, &sh);
    if (sw && sh) { sc->offset.x = (int32_t)sx; sc->offset.y = (int32_t)sy; sc->extent.width = sw; sc->extent.height = sh; }
    else { sc->offset.x = 0; sc->offset.y = 0; sc->extent.width = tw; sc->extent.height = th; }
    /* Vulkan wants the scissor inside the render area. */
    if ((u32)sc->offset.x > tw) sc->offset.x = (int32_t)tw;
    if ((u32)sc->offset.y > th) sc->offset.y = (int32_t)th;
    if (sc->offset.x + sc->extent.width > tw) sc->extent.width = tw - (u32)sc->offset.x;
    if (sc->offset.y + sc->extent.height > th) sc->extent.height = th - (u32)sc->offset.y;
}

static VkImageView null_view(int cube) { EngObj* o = eng_obj(cube ? s_null_cube : s_null_tex); return o ? o->view : VK_NULL_HANDLE; }

static void eng_encode_draw(const EngRecord* r, EngStage* stage)
{
    /* Attachments: target A's size rules, as the D3D12 backend. */
    u32 nrt = 0, tw = 0, th = 0;
    for (u32 k = 0; k < r->nrt && k < RSX_BE_MAX_COLOR_TARGETS; k++) {
        EngObj* o = eng_owner(r->rt[k]);
        if (!o || o->kind == OBJ_DEPTH) break;
        if (nrt && (o->w != tw || o->h != th)) break;
        if (!nrt) { tw = o->w; th = o->h; }
        nrt++;
    }
    if (nrt != r->nrt) return;
    u32 depth = r->depth;
    EngObj* z = eng_owner(depth);
    if (z && z->kind != OBJ_DEPTH) { z = NULL; depth = 0; }
    if (!nrt && !z) return;
    if (!z) { depth = eng_fallback_depth(tw, th); z = eng_owner(depth); if (!z) return; }
    if (!nrt) { tw = z->w; th = z->h; }
    else { if (z->w < tw) tw = z->w; if (z->h < th) th = z->h; }

    const int scaled = eng_obj_scaled(nrt ? r->rt[0] : depth);
    const VkPipeline pso = eng_pso_for(r->pipeline, topo_class(r->topology), !scaled);
    if (!pso) return;
    const EngPipeline* pp = &s_pipe[r->pipeline - 1];

    /* Textures; a unit naming one of this draw's targets reads the null. */
    EngSetDesc d; memset(&d, 0, sizeof d);
    EngObj* tex_obj[20] = {0};
    for (u32 u = 0; u < 20; u++) {
        const u32 h = u < 16 ? r->tex[u] : r->vtex[u - 16];
        EngObj* v = eng_obj(h);
        EngObj* o = eng_owner(h);
        int hazard = 0;
        if (o) {
            for (u32 k = 0; k < nrt; k++) if (eng_owner(r->rt[k]) == o) hazard = 1;
            if (z == o) hazard = 1;
        }
        const int cube = u < 16 && ((pp->cube_mask >> u) & 1u);
        VkImageView view = (!o || hazard || !v || !v->view) ? null_view(cube) : v->view;
        if (o && !hazard && v && v->view) tex_obj[u] = o;
        if (u < 16) { d.fs_view[u] = view; }
        else d.vs_view[u - 16] = view;
    }
    for (u32 u = 0; u < 16; u++) d.fs_smp[u] = r->samp[u] >= 0 ? s_samp[r->samp[u]].smp : s_point_sampler;
    for (u32 u = 0; u < 4; u++) d.vs_smp[u] = r->vsamp[u] >= 0 ? s_samp[r->vsamp[u]].smp : s_point_sampler;
    const VkBuffer sbuf = stage ? stage->buf : VK_NULL_HANDLE;
    d.vs_buf = (r->vs_cb_bytes && sbuf) ? sbuf : s_zero_buf;
    d.fs_buf = (r->ps_cb_bytes && sbuf) ? sbuf : s_zero_buf;

    /* Layouts: anything to change ends the pass. */
    int same = s_in_render && s_cur_nrt == nrt && s_cur_depth == depth && s_cur_tw == tw && s_cur_th == th;
    for (u32 k = 0; same && k < nrt; k++) same = s_cur_rt[k] == r->rt[k];
    int change = 0;
    for (u32 k = 0; k < nrt; k++) change |= obj_needs(eng_owner(r->rt[k]), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    change |= obj_needs(z, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    for (u32 u = 0; u < 20; u++) change |= obj_needs(tex_obj[u], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!same || change) {
        render_end();
        for (u32 k = 0; k < nrt; k++) obj_layout(eng_owner(r->rt[k]), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        obj_layout(z, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        for (u32 u = 0; u < 20; u++) obj_layout(tex_obj[u], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        u32 rts[RSX_BE_MAX_COLOR_TARGETS];
        for (u32 k = 0; k < nrt; k++) rts[k] = r->rt[k];
        render_begin(rts, nrt, depth, tw, th);
        s_cur_pso = VK_NULL_HANDLE;
    }

    if (pso != s_cur_pso) { vkCmdBindPipeline(s_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pso); s_cur_pso = pso; s_cur_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM; }
    const VkPrimitiveTopology topo = topo_vk(r->topology);
    if (topo != s_cur_topo) { vkCmdSetPrimitiveTopology(s_cmd, topo); s_cur_topo = topo; }
    VkViewport vp; VkRect2D sc;
    eng_vp_sc(r, scaled, tw, th, &vp, &sc);
    vkCmdSetViewport(s_cmd, 0, 1, &vp);
    vkCmdSetScissor(s_cmd, 0, 1, &sc);
    vkCmdSetStencilReference(s_cmd, VK_STENCIL_FACE_FRONT_AND_BACK, r->stencil_ref);

    const VkDescriptorSet set = eng_set(&d);
    if (!set) return;
    const u32 dyn[2] = { d.vs_buf == sbuf ? r->vs_cb_off : 0u, d.fs_buf == sbuf ? r->ps_cb_off : 0u };
    vkCmdBindDescriptorSets(s_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pl, 0, 1, &set, 2, dyn);

    const VkBuffer vbuf = r->vbuf ? s_buf[r->vbuf - 1] : sbuf;
    if (!vbuf) return;
    const VkDeviceSize voff = r->vb_off;
    vkCmdBindVertexBuffers(s_cmd, 0, 1, &vbuf, &voff);
    const int query = r->vis && s_q_used < ENG_QUERIES_PER_SLOT;
    if (query) vkCmdBeginQuery(s_cmd, s_qpool[s_slot], s_q_used, s_query_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
    if (r->index_count) {
        vkCmdBindIndexBuffer(s_cmd, vbuf, r->ib_off, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(s_cmd, r->index_count, 1, 0, 0, 0);
    } else {
        vkCmdDraw(s_cmd, r->vertex_count, 1, 0, 0);
    }
    if (query) { vkCmdEndQuery(s_cmd, s_qpool[s_slot], s_q_used); s_q_vis[s_q_used++] = (r->vis - 1) | (scaled ? ENG_Q_SCALED : 0u); }
}

/* The depth resolve passes: the depth plane drawn into a colour snapshot. */
static void eng_encode_depth_resolve(const EngRecord* r)
{
    EngObj* src = eng_owner(r->depth); EngObj* dst = eng_owner(r->resolve_dst);
    const VkPipeline pso = r->resolve_packed ? s_depth_pack_pso : s_depth_pso;
    if (!src || !dst || !pso) return;
    render_end();
    obj_layout(src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    obj_layout(dst, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    const u32 rt[1] = { r->resolve_dst };
    render_begin(rt, 1, 0, dst->w, dst->h);
    vkCmdBindPipeline(s_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pso);
    s_cur_pso = pso;
    VkViewport vp = { 0.0f, (float)dst->h, (float)dst->w, -(float)dst->h, 0.0f, 1.0f };
    VkRect2D sc = { { 0, 0 }, { dst->w, dst->h } };
    vkCmdSetViewport(s_cmd, 0, 1, &vp);
    vkCmdSetScissor(s_cmd, 0, 1, &sc);
    vkCmdSetStencilReference(s_cmd, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    vkCmdSetPrimitiveTopology(s_cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST); s_cur_topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    EngSetDesc d; memset(&d, 0, sizeof d);
    for (u32 u = 0; u < 16; u++) { d.fs_view[u] = null_view(0); d.fs_smp[u] = s_point_sampler; }
    for (u32 u = 0; u < 4; u++) { d.vs_view[u] = null_view(0); d.vs_smp[u] = s_point_sampler; }
    d.fs_view[0] = src->view;
    d.vs_buf = d.fs_buf = s_zero_buf;
    const VkDescriptorSet set = eng_set(&d);
    if (!set) return;
    const u32 dyn[2] = { 0, 0 };
    vkCmdBindDescriptorSets(s_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pl, 0, 1, &set, 2, dyn);
    vkCmdDraw(s_cmd, 3, 1, 0, 0);
    render_end();
}

static void eng_encode_records(EngStage* stage)
{
    s_in_render = 0; s_cur_pso = VK_NULL_HANDLE; s_cur_set = VK_NULL_HANDLE; s_cur_set_key = 0;
    s_cur_topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    for (u32 i = 0; i < s_rec_count; i++) {
        const EngRecord* r = &s_rec[i];
        switch (r->kind) {
        case ENG_REC_DRAW:
            eng_encode_draw(r, stage);
            break;
        case ENG_REC_UPLOAD: {
            EngObj* o = eng_owner(r->depth);
            if (!o || !stage || !stage->buf) break;
            render_end();
            obj_layout(o, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            bar_flush();
            VkBufferImageCopy c = {0};
            c.bufferOffset = r->up_off; c.bufferRowLength = r->up_row_texels; c.bufferImageHeight = r->up_rows_texels;
            c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            c.imageSubresource.mipLevel = r->up_mip; c.imageSubresource.baseArrayLayer = r->up_layer;
            c.imageSubresource.layerCount = 1;
            c.imageExtent.width = r->up_w; c.imageExtent.height = r->up_h; c.imageExtent.depth = 1;
            vkCmdCopyBufferToImage(s_cmd, stage->buf, o->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
            s_bar_global = 1;
            break; }
        case ENG_REC_CLEAR_COLOR: {
            EngObj* o = eng_owner(r->rt[0]);
            if (!o || o->kind == OBJ_DEPTH) break;
            int att = -1;
            if (s_in_render) for (u32 k = 0; k < s_cur_nrt; k++) if (eng_owner(s_cur_rt[k]) == o) att = (int)k;
            VkClearColorValue cv; memcpy(cv.float32, r->clear_rgba, sizeof cv.float32);
            if (att >= 0 && o->w == s_cur_tw && o->h == s_cur_th) {
                VkClearAttachment ca = { VK_IMAGE_ASPECT_COLOR_BIT, (u32)att };
                ca.clearValue.color = cv;
                VkClearRect rc = { { { 0, 0 }, { s_cur_tw, s_cur_th } }, 0, 1 };
                vkCmdClearAttachments(s_cmd, 1, &ca, 1, &rc);
                break;
            }
            render_end();
            obj_layout(o, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            bar_flush();
            VkImageSubresourceRange rg = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(s_cmd, o->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &rg);
            s_bar_global = 1;
            break; }
        case ENG_REC_CLEAR_DS:
        case ENG_REC_CLEAR_DS_RECT: {
            EngObj* z = eng_owner(r->depth);
            if (!z || z->kind != OBJ_DEPTH) break;
            VkImageAspectFlags asp = 0;
            if (r->clear_flags & RSX_BE_CLEAR_DEPTH)   asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
            if (r->clear_flags & RSX_BE_CLEAR_STENCIL) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
            if (!asp) break;
            VkClearDepthStencilValue dv = { r->clear_depth, r->clear_stencil };
            if (r->kind == ENG_REC_CLEAR_DS_RECT || (s_in_render && eng_owner(s_cur_depth) == z)) {
                u32 x = 0, y = 0, w = z->w, h = z->h;
                if (r->kind == ENG_REC_CLEAR_DS_RECT) {
                    x = r->clear_rect[0]; y = r->clear_rect[1]; w = r->clear_rect[2]; h = r->clear_rect[3];
                    if (x >= z->w || y >= z->h) break;
                    if (x + w > z->w) w = z->w - x;
                    if (y + h > z->h) h = z->h - y;
                }
                if (!(s_in_render && eng_owner(s_cur_depth) == z)) {
                    render_end();
                    obj_layout(z, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                    render_begin(NULL, 0, r->depth, z->w, z->h);
                    s_cur_pso = VK_NULL_HANDLE;
                }
                if (x + w > s_cur_tw) w = s_cur_tw > x ? s_cur_tw - x : 0;
                if (y + h > s_cur_th) h = s_cur_th > y ? s_cur_th - y : 0;
                if (!w || !h) break;
                VkClearAttachment ca = { asp, 0 };
                ca.clearValue.depthStencil = dv;
                VkClearRect rc = { { { (int32_t)x, (int32_t)y }, { w, h } }, 0, 1 };
                vkCmdClearAttachments(s_cmd, 1, &ca, 1, &rc);
                break;
            }
            render_end();
            obj_layout(z, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            bar_flush();
            VkImageSubresourceRange rg = { asp, 0, 1, 0, 1 };
            vkCmdClearDepthStencilImage(s_cmd, z->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &dv, 1, &rg);
            s_bar_global = 1;
            break; }
        case ENG_REC_COLOR_COPY: {
            EngObj* src = eng_owner(r->depth); EngObj* dst = eng_owner(r->resolve_dst);
            if (!src || !dst) break;
            render_end();
            obj_layout(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            obj_layout(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            bar_flush();
            VkImageCopy c = {0};
            c.srcSubresource.aspectMask = c.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            c.srcSubresource.layerCount = c.dstSubresource.layerCount = 1;
            c.extent.width = src->w < dst->w ? src->w : dst->w;
            c.extent.height = src->h < dst->h ? src->h : dst->h;
            c.extent.depth = 1;
            vkCmdCopyImage(s_cmd, src->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
            s_bar_global = 1;
            break; }
        case ENG_REC_DEPTH_RESOLVE:
            eng_encode_depth_resolve(r);
            break;
        }
    }
    render_end();
    bar_flush();
}

/* ---- submit ------------------------------------------------------------------------- */

static void eng_poll_submits(void)
{
    for (;;) {
        EngSubmit* s = NULL;
        for (int i = 0; i < ENG_MAX_SUBMITS; i++)
            if (s_sub[i].live && (!s || s_sub[i].fence < s->fence)) s = &s_sub[i];
        if (!s || !fence_done(s->fence)) break;
        if (s->q_count) {
            const u64* p = s_qmap[s->q_slot];
            for (u32 k = 0; k < s->q_count; k++)
                s_vis_count[s->q_vis[k] & ~ENG_Q_SCALED] += eng_q_count(s->q_vis[k], p[k]);
        }
        for (u32 k = 0; k < s->n_reports; k++)
            rsx_draw_engine_query_result(s->reports[k].index, s_vis_count[s->reports[k].slot]);
        free(s->q_vis); free(s->reports);
        memset(s, 0, sizeof *s);
    }
}

static int eng_swapchain_create(void);
static void eng_dump_frame(u32 surface);

static void eng_submit(u32 present_surface, int wait)
{
    if (!s_walker_tid) s_walker_tid = GetCurrentThreadId();
    if (!s_ready) { s_rec_count = 0; s_stage_used = 0; s_stage_cur = -1; s_vis_npending = 0; return; }
    if (!s_rec_count && !present_surface) return;
    eng_poll_submits();

    s_slot = s_submit_seq % ENG_FRAMES;
    fence_wait(s_slot_fence[s_slot]);
    eng_poll_submits();   /* the slot's query results are read before its pool is reset */
    vkResetCommandPool(s_dev, s_pool[s_slot], 0);
    vkResetDescriptorPool(s_dev, s_dpool[s_slot], 0);
    s_cmd = s_cmdbuf[s_slot];
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(s_cmd, &bi);
    vkCmdResetQueryPool(s_cmd, s_qpool[s_slot], 0, ENG_QUERIES_PER_SLOT);
    s_q_used = 0;

    EngStage* stage = s_stage_cur >= 0 ? &s_stage[s_stage_cur] : NULL;
    eng_encode_records(stage);

    /* Query results into the slot's readback buffer. */
    if (s_q_used) {
        vkCmdCopyQueryPoolResults(s_cmd, s_qpool[s_slot], 0, s_q_used, s_qbuf[s_slot], 0, 8,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    }

    /* Present: the surface blitted to the swap chain image. */
    u32 img_index = 0; int presenting = 0;
    if (present_surface && !s_headless && s_swap) {
        EngObj* src = eng_owner(present_surface);
        if (src) {
            VkResult ar = vkAcquireNextImageKHR(s_dev, s_swap, 2000000000ull, s_sem_acq[s_slot], VK_NULL_HANDLE, &img_index);
            if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
                /* Leave this frame unpresented; the chain is rebuilt below. */
            } else if (ar == VK_SUCCESS || ar == VK_SUBOPTIMAL_KHR) {
                presenting = 1;
                obj_layout(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                VkImageLayout sl = VK_IMAGE_LAYOUT_UNDEFINED;
                img_layout(s_swap_img[img_index], &sl, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                bar_flush();
                VkImageBlit bl = {0};
                bl.srcSubresource.aspectMask = bl.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                bl.srcSubresource.layerCount = bl.dstSubresource.layerCount = 1;
                bl.srcOffsets[1].x = (int32_t)src->w; bl.srcOffsets[1].y = (int32_t)src->h; bl.srcOffsets[1].z = 1;
                bl.dstOffsets[1].x = (int32_t)s_swap_ext.width; bl.dstOffsets[1].y = (int32_t)s_swap_ext.height; bl.dstOffsets[1].z = 1;
                vkCmdBlitImage(s_cmd, src->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_swap_img[img_index],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
                img_layout(s_swap_img[img_index], &sl, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
                bar_flush();
            }
        }
    }
    vkEndCommandBuffer(s_cmd);

    EngSubmit* sub = NULL;
    for (int i = 0; i < ENG_MAX_SUBMITS && !sub; i++) if (!s_sub[i].live) sub = &s_sub[i];
    if (!sub) {
        int oldest = 0;
        for (int i = 1; i < ENG_MAX_SUBMITS; i++) if (s_sub[i].fence < s_sub[oldest].fence) oldest = i;
        fence_wait(s_sub[oldest].fence);
        eng_poll_submits();
        sub = &s_sub[oldest];
    }
    memset(sub, 0, sizeof *sub);
    sub->q_slot = s_slot; sub->q_count = s_q_used;
    if (s_q_used) { sub->q_vis = (u32*)malloc(s_q_used * sizeof(u32)); if (sub->q_vis) memcpy(sub->q_vis, s_q_vis, s_q_used * sizeof(u32)); else sub->q_count = 0; }
    if (s_vis_npending) {
        sub->reports = (EngVisReport*)malloc(s_vis_npending * sizeof(EngVisReport));
        if (sub->reports) { memcpy(sub->reports, s_vis_pending, s_vis_npending * sizeof(EngVisReport)); sub->n_reports = s_vis_npending; }
        s_vis_npending = 0;
    }

    const u64 f = ++s_fence_value;
    VkSemaphore sig[2] = { s_timeline, presenting ? s_sem_done[img_index] : VK_NULL_HANDLE };
    const u64 sigv[2] = { f, 0 };
    VkTimelineSemaphoreSubmitInfo ts = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    ts.signalSemaphoreValueCount = presenting ? 2 : 1; ts.pSignalSemaphoreValues = sigv;
    const u64 waitv = 0;
    ts.waitSemaphoreValueCount = presenting ? 1 : 0; ts.pWaitSemaphoreValues = &waitv;
    VkPipelineStageFlags wstage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.pNext = &ts;
    si.waitSemaphoreCount = presenting ? 1 : 0; si.pWaitSemaphores = &s_sem_acq[s_slot]; si.pWaitDstStageMask = &wstage;
    si.commandBufferCount = 1; si.pCommandBuffers = &s_cmd;
    si.signalSemaphoreCount = presenting ? 2 : 1; si.pSignalSemaphores = sig;
    const VkResult sr = vkQueueSubmit(s_queue, 1, &si, VK_NULL_HANDLE);
    if (sr != VK_SUCCESS) {
        static int n = 0;
        if (n++ < 8) fprintf(stderr, "[rsx engine/vulkan] submit failed: %d\n", (int)sr);
        if (sr == VK_ERROR_DEVICE_LOST) s_ready = 0;
    }
    s_slot_fence[s_slot] = f;
    sub->fence = f; sub->live = 1; sub->stage = s_stage_cur;
    if (s_stage_cur >= 0) { s_stage[s_stage_cur].fence = f; s_stage[s_stage_cur].in_use = 1; }
    s_stage_cur = -1; s_stage_used = 0;
    s_rec_count = 0;
    s_submit_seq++;

    if (presenting) {
        VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &s_sem_done[img_index];
        pi.swapchainCount = 1; pi.pSwapchains = &s_swap; pi.pImageIndices = &img_index;
        const VkResult pr = vkQueuePresentKHR(s_queue, &pi);
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
            vkDeviceWaitIdle(s_dev);
            eng_swapchain_create();
        }
        if (f > ENG_INFLIGHT) fence_wait(f - ENG_INFLIGHT);
    } else if (present_surface && !s_headless && s_swap) {
        vkDeviceWaitIdle(s_dev);
        eng_swapchain_create();
    }
    if (wait || (present_surface && s_headless)) fence_wait(f);
    if (s_dropped) {
        fprintf(stderr, "[rsx engine/vulkan] dropped %u record(s) (cap %d)\n", s_dropped, ENG_MAX_RECORDS);
        s_dropped = 0;
    }
    eng_poll_submits();
    eng_collect_retired();
    eng_collect_retired_buffers();
}

static void eng_submit_and_wait(void* user, u32 reason) { (void)user; (void)reason; eng_submit(0, 1); }

/* ---- readback and present ------------------------------------------------------------- */

static VkBuffer s_rb_buf; static MemAlloc s_rb_mem; static u32 s_rb_cap;

static void eng_readback(void* user, u32 surface, u32 x, u32 y, u32 w, u32 h, void* out, u32 out_pitch)
{
    (void)user;
    EngObj* o = eng_owner(surface);
    if (!o || !out || !out_pitch || !w || !h || !s_ready || o->kind == OBJ_DEPTH) return;
    const u32 gw = w, gh = h;
    const int scaled = o->gw && o->gh && (o->gw != o->w || o->gh != o->h);
    if (scaled) {
        if (x + w > o->gw || y + h > o->gh) return;
        const u32 x0 = (u32)((u64)x * o->w / o->gw), x1 = (u32)((u64)(x + w) * o->w / o->gw);
        const u32 y0 = (u32)((u64)y * o->h / o->gh), y1 = (u32)((u64)(y + h) * o->h / o->gh);
        x = x0; y = y0; w = x1 > x0 ? x1 - x0 : 1u; h = y1 > y0 ? y1 - y0 : 1u;
    }
    if (x + w > o->w || y + h > o->h) return;
    const u32 bpp = eng_bpp(o->fmt);
    const u32 pitch = w * bpp;
    const u32 need = pitch * h;
    if (need > s_rb_cap) {
        if (s_rb_buf) { vkDestroyBuffer(s_dev, s_rb_buf, NULL); mem_free(&s_rb_mem); }
        s_rb_buf = make_buffer(need, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               HOST_MEM | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &s_rb_mem);
        if (!s_rb_buf) s_rb_buf = make_buffer(need, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST_MEM, &s_rb_mem);
        s_rb_cap = s_rb_buf ? need : 0;
        if (!s_rb_buf) return;
    }
    fence_wait(s_fence_value);
    vkResetCommandPool(s_dev, s_aux_pool, 0);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(s_aux_cmd, &bi);
    VkCommandBuffer saved = s_cmd; s_cmd = s_aux_cmd;
    obj_layout(o, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    s_bar_global = 1;
    bar_flush();
    VkBufferImageCopy c = {0};
    c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; c.imageSubresource.layerCount = 1;
    c.imageOffset.x = (int32_t)x; c.imageOffset.y = (int32_t)y;
    c.imageExtent.width = w; c.imageExtent.height = h; c.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(s_cmd, o->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_rb_buf, 1, &c);
    s_bar_global = 1; bar_flush();
    s_cmd = saved;
    vkEndCommandBuffer(s_aux_cmd);
    const u64 f = ++s_fence_value;
    VkTimelineSemaphoreSubmitInfo ts = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    ts.signalSemaphoreValueCount = 1; ts.pSignalSemaphoreValues = &f;
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.pNext = &ts; si.commandBufferCount = 1; si.pCommandBuffers = &s_aux_cmd;
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &s_timeline;
    vkQueueSubmit(s_queue, 1, &si, VK_NULL_HANDLE);
    fence_wait(f);
    const u8* p = mem_ptr(&s_rb_mem);
    if (!p) return;
    if (!scaled) {
        for (u32 r = 0; r < h; r++) memcpy((u8*)out + (size_t)r * out_pitch, p + (size_t)r * pitch, (size_t)w * bpp);
    } else {
        for (u32 r = 0; r < gh; r++) {
            const u8* srow = p + (size_t)(((u64)r * h + h / 2) / gh) * pitch;
            u8* drow = (u8*)out + (size_t)r * out_pitch;
            for (u32 c2 = 0; c2 < gw; c2++)
                memcpy(drow + (size_t)c2 * bpp, srow + (size_t)(((u64)c2 * w + w / 2) / gw) * bpp, bpp);
        }
    }
}

static const char* env2(const char* a, const char* b)
{
    const char* v = getenv(a);
    if (!v || !*v) v = getenv(b);
    return (v && *v) ? v : NULL;
}
/* RSX_REPLAY_DUMP_PATH, PS3RECOMP_FRAME_GRAB and PS3RECOMP_FRAME_DUMP, as the
 * D3D12 backend's eng_dump_frame. */
static void eng_dump_frame(u32 surface)
{
    EngObj* o = eng_owner(surface);
    if (!o || o->fmt != VK_FORMAT_R8G8B8A8_UNORM) return;
    const char* path = getenv("RSX_REPLAY_DUMP_PATH");
    if (path && !*path) path = NULL;
    char seqpath[1024], grab_req[1024], grab_tmp[1024], grab_out[1024];
    int grabbing = 0;
    static const char* grab = (const char*)1;
    if (grab == (const char*)1) grab = env2("PS3RECOMP_FRAME_GRAB", "PS3RECOMP_METAL_FRAME_GRAB");
    if (!path && grab) {
        snprintf(grab_req, sizeof grab_req, "%s.req", grab);
        if (_access(grab_req, 0) == 0) {
            snprintf(grab_tmp, sizeof grab_tmp, "%s.ppm.tmp", grab);
            snprintf(grab_out, sizeof grab_out, "%s.ppm", grab);
            path = grab_tmp; grabbing = 1;
        }
    }
    if (!path) {
        static const char* dump = (const char*)1;
        if (dump == (const char*)1) dump = env2("PS3RECOMP_FRAME_DUMP", "PS3RECOMP_METAL_FRAME_DUMP");
        if (!dump) return;
        static unsigned frame, every; static int seq = -1;
        if (!every) { const char* e = env2("PS3RECOMP_FRAME_EVERY", "PS3RECOMP_METAL_FRAME_EVERY"); every = e ? (unsigned)atoi(e) : 120u; if (!every) every = 120u; }
        if (seq < 0) seq = env2("PS3RECOMP_FRAME_SEQ", "PS3RECOMP_METAL_FRAME_SEQ") ? 1 : 0;
        if ((++frame % every) != 0) return;
        path = dump;
        if (seq) { snprintf(seqpath, sizeof seqpath, "%s.%06u.ppm", dump, frame); path = seqpath; }
    }
    const u32 w = o->gw ? o->gw : o->w, h = o->gh ? o->gh : o->h;
    if (!w || !h || w > 16384 || h > 16384) return;
    u8* rgba = (u8*)malloc((size_t)w * h * 4);
    u8* rgb = (u8*)malloc((size_t)w * h * 3);
    if (!rgba || !rgb) { free(rgba); free(rgb); return; }
    eng_readback(NULL, surface, 0, 0, w, h, rgba, w * 4);
    for (size_t p = 0; p < (size_t)w * h; p++) { rgb[p*3] = rgba[p*4]; rgb[p*3+1] = rgba[p*4+1]; rgb[p*3+2] = rgba[p*4+2]; }
    FILE* f = fopen(path, "wb");
    if (f) {
        fprintf(f, "P6\n%u %u\n255\n", w, h);
        fwrite(rgb, 1, (size_t)w * h * 3, f);
        fclose(f);
        if (grabbing) { remove(grab_out); rename(grab_tmp, grab_out); remove(grab_req); }
        else fprintf(stderr, "[rsx engine/vulkan] captured frame to %s\n", path);
    }
    free(rgba); free(rgb);
}

static void eng_present(void* user, u32 surface)
{
    (void)user;
    if (!eng_owner(surface)) { eng_submit(0, 0); return; }
    eng_submit(surface, 0);
    eng_dump_frame(surface);
}

static int eng_init(void* user, u32 width, u32 height) { (void)user; (void)width; (void)height; return s_dev ? 0 : -1; }
static void eng_shutdown(void* user);

static const rsx_draw_backend s_engine_backend = {
    .user                 = NULL,
    .init                 = eng_init,
    .shutdown             = eng_shutdown,
    .submit_and_wait      = eng_submit_and_wait,
    .texture_create       = eng_texture_create,
    .texture_upload       = eng_texture_upload,
    .texture_release      = eng_obj_release,
    .color_target_create  = eng_color_target_create,
    .color_target_release = eng_obj_release,
    .surface_view         = eng_surface_view,
    .depth_target_create  = eng_depth_target_create,
    .depth_target_release = eng_obj_release,
    .depth_snapshot       = eng_depth_snapshot,
    .depth_snapshot_rgba8 = eng_depth_snapshot_rgba8,
    .color_snapshot       = eng_color_snapshot,
    .query_begin          = eng_query_begin,
    .query_set            = eng_query_set,
    .query_report         = eng_query_report,
    .pipeline_create      = eng_pipeline_create,
    .pipeline_release     = eng_pipeline_release,
    .bind_targets         = eng_bind_targets,
    .bind_pipeline        = eng_bind_pipeline,
    .bind_vs_constants    = eng_bind_vs_constants,
    .reuse_vs_constants   = eng_reuse_vs_constants,
    .bind_ps_constants    = eng_bind_ps_constants,
    .bind_textures        = eng_bind_textures,
    .bind_vertex_textures = eng_bind_vertex_textures,
    .set_viewport         = eng_set_viewport,
    .set_scissor          = eng_set_scissor,
    .set_stencil_ref      = eng_set_stencil_ref,
    .draw                 = eng_draw,
    .buffer_wrap          = eng_buffer_wrap,
    .buffer_release       = eng_buffer_release,
    .draw_buffer          = eng_draw_buffer,
    .clear_color          = eng_clear_color,
    .clear_depth_stencil  = eng_clear_depth_stencil,
    .clear_depth_stencil_rect = eng_clear_depth_stencil_rect,
    .present              = eng_present,
    .readback             = eng_readback,
};

/* ---- device ---------------------------------------------------------------------------- */

static VKAPI_ATTR VkBool32 VKAPI_CALL eng_debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type,
                                                   const VkDebugUtilsMessengerCallbackDataEXT* data, void* user)
{
    (void)type; (void)user;
    static int n = 0;
    if (n < 200 && (sev & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT))) {
        n++;
        fprintf(stderr, "[vk-validation] %s\n", data && data->pMessage ? data->pMessage : "?");
    }
    return VK_FALSE;
}

static int eng_load_vulkan(void)
{
    s_vk_dll = LoadLibraryA("vulkan-1.dll");
    if (!s_vk_dll) { fprintf(stderr, "[RSX vulkan] vulkan-1.dll not found (no Vulkan driver)\n"); return -1; }
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(s_vk_dll, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) return -1;
#define VK_LOAD_G(n) n = (PFN_##n)vkGetInstanceProcAddr(VK_NULL_HANDLE, #n);
    VK_GLOBAL_FNS(VK_LOAD_G)
#undef VK_LOAD_G
    return vkCreateInstance ? 0 : -1;
}

static int has_ext(const VkExtensionProperties* e, u32 n, const char* name)
{
    for (u32 i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

static int eng_swapchain_create(void)
{
    VkSurfaceCapabilitiesKHR caps;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(s_phys, s_surface, &caps) != VK_SUCCESS) return -1;
    if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0) return -1;   /* minimised */
    u32 nf = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(s_phys, s_surface, &nf, NULL);
    VkSurfaceFormatKHR fmts[64]; if (nf > 64) nf = 64;
    vkGetPhysicalDeviceSurfaceFormatsKHR(s_phys, s_surface, &nf, fmts);
    VkSurfaceFormatKHR sf = fmts[0];
    for (u32 i = 0; i < nf; i++)
        if ((fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM || fmts[i].format == VK_FORMAT_R8G8B8A8_UNORM) &&
            fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { sf = fmts[i]; break; }
    u32 npm = 0; vkGetPhysicalDeviceSurfacePresentModesKHR(s_phys, s_surface, &npm, NULL);
    VkPresentModeKHR pms[16]; if (npm > 16) npm = 16;
    vkGetPhysicalDeviceSurfacePresentModesKHR(s_phys, s_surface, &npm, pms);
    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
    if (!s_vsync) {
        for (u32 i = 0; i < npm; i++) if (pms[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) pm = pms[i];
        if (pm == VK_PRESENT_MODE_FIFO_KHR) for (u32 i = 0; i < npm; i++) if (pms[i] == VK_PRESENT_MODE_MAILBOX_KHR) pm = pms[i];
    }
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        fprintf(stderr, "[RSX vulkan] the swap chain cannot be a blit destination\n");
        return -1;
    }
    VkSwapchainCreateInfoKHR ci = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    ci.surface = s_surface;
    ci.minImageCount = caps.minImageCount < 3 ? 3 : caps.minImageCount;
    if (caps.maxImageCount && ci.minImageCount > caps.maxImageCount) ci.minImageCount = caps.maxImageCount;
    ci.imageFormat = sf.format; ci.imageColorSpace = sf.colorSpace;
    ci.imageExtent = caps.currentExtent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = pm; ci.clipped = VK_TRUE;
    ci.oldSwapchain = s_swap;
    VkSwapchainKHR sw = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(s_dev, &ci, NULL, &sw) != VK_SUCCESS) return -1;
    if (s_swap) vkDestroySwapchainKHR(s_dev, s_swap, NULL);
    s_swap = sw;
    s_swap_fmt = sf.format; s_swap_ext = caps.currentExtent;
    s_swap_n = 8;
    vkGetSwapchainImagesKHR(s_dev, s_swap, &s_swap_n, s_swap_img);
    for (u32 i = 0; i < s_swap_n; i++) {
        s_swap_layout[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!s_sem_done[i]) {
            VkSemaphoreCreateInfo si = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            vkCreateSemaphore(s_dev, &si, NULL, &s_sem_done[i]);
        }
    }
    return 0;
}

static const char kHelperVS[] =
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "VSOut main(uint vid : SV_VertexID) {\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    VSOut o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); o.uv = float2(p.x, 1.0 - p.y); return o;\n"
    "}\n";
static const char kHelperDepthPS[] =
    "Texture2D<float> dsrc : register(t0);\n"
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "float main(VSOut i) : SV_Target { return dsrc.Load(int3(i.pos.xy, 0)); }\n";
static const char kHelperPackPS[] =
    "Texture2D<float> dsrc : register(t0);\n"
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "float4 main(VSOut i) : SV_Target {\n"
    "    uint v = uint(saturate(dsrc.Load(int3(i.pos.xy, 0))) * 16777215.0 + 0.5);\n"
    "    return float4(float((v >> 8) & 255u), float(v & 255u), 0.0, float((v >> 16) & 255u)) / 255.0;\n"
    "}\n";

static VkPipeline eng_helper_pso(VkShaderModule ps, VkFormat rt)
{
    if (!s_helper_vs || !ps) return VK_NULL_HANDLE;
    EngPipeline p; memset(&p, 0, sizeof p);
    p.vs = s_helper_vs; p.fs = ps; p.nattr = 0; p.stride = 16; p.rt_count = 1; p.rt_fmt = rt;
    p.rs.color_mask = 0xFFFFFFFFu;
    p.no_depth = 1;
    return eng_build_pso(&p, 2, ps);
}

static int eng_init_device(void)
{
    if (eng_load_vulkan() != 0) return -1;
    u32 api = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&api);
    if (api < VK_API_VERSION_1_3) { fprintf(stderr, "[RSX vulkan] the loader is Vulkan %u.%u; 1.3 is needed\n", VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api)); return -1; }

    const char* exts[4]; u32 next = 0;
    if (!s_headless) { exts[next++] = VK_KHR_SURFACE_EXTENSION_NAME; exts[next++] = VK_KHR_WIN32_SURFACE_EXTENSION_NAME; }
    const char* layers[1]; u32 nlay = 0;
    const int validate = getenv("RSX_VK_VALIDATION") && getenv("RSX_VK_VALIDATION")[0] == '1';
    if (validate) {
        u32 n = 0; vkEnumerateInstanceLayerProperties(&n, NULL);
        VkLayerProperties* lp = (VkLayerProperties*)calloc(n ? n : 1, sizeof *lp);
        vkEnumerateInstanceLayerProperties(&n, lp);
        for (u32 i = 0; i < n; i++) if (!strcmp(lp[i].layerName, "VK_LAYER_KHRONOS_validation")) layers[nlay++] = lp[i].layerName, i = n;
        if (nlay) layers[0] = "VK_LAYER_KHRONOS_validation";
        free(lp);
        exts[next++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        if (!nlay) fprintf(stderr, "[RSX vulkan] RSX_VK_VALIDATION: the validation layer is not installed\n");
    }
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "ps3recomp"; app.pEngineName = "ps3recomp"; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = next; ici.ppEnabledExtensionNames = exts;
    ici.enabledLayerCount = nlay; ici.ppEnabledLayerNames = layers;
    VKCHECK(vkCreateInstance(&ici, NULL, &s_inst), "vkCreateInstance");
#define VK_LOAD_I(n) n = (PFN_##n)vkGetInstanceProcAddr(s_inst, #n);
    VK_INSTANCE_FNS(VK_LOAD_I)
#undef VK_LOAD_I
    if (validate) {
        vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(s_inst, "vkCreateDebugUtilsMessengerEXT");
        vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(s_inst, "vkDestroyDebugUtilsMessengerEXT");
        if (vkCreateDebugUtilsMessengerEXT) {
            VkDebugUtilsMessengerCreateInfoEXT di = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
            di.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            di.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            di.pfnUserCallback = eng_debug_cb;
            vkCreateDebugUtilsMessengerEXT(s_inst, &di, NULL, &s_dbg);
        }
    }
    if (!s_headless) {
        VkWin32SurfaceCreateInfoKHR si = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
        si.hinstance = GetModuleHandle(NULL); si.hwnd = s_hwnd;
        VKCHECK(vkCreateWin32SurfaceKHR(s_inst, &si, NULL, &s_surface), "vkCreateWin32SurfaceKHR");
    }

    /* A device with Vulkan 1.3 and a graphics queue that can present; a
     * discrete GPU first. */
    u32 nd = 0; vkEnumeratePhysicalDevices(s_inst, &nd, NULL);
    VkPhysicalDevice devs[16]; if (nd > 16) nd = 16;
    vkEnumeratePhysicalDevices(s_inst, &nd, devs);
    int best = -1, best_score = -1; u32 best_q = 0;
    for (u32 i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(devs[i], &pr);
        if (pr.apiVersion < VK_API_VERSION_1_3) continue;
        u32 nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, NULL);
        VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, qp);
        for (u32 q = 0; q < nq; q++) {
            if (!(qp[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            VkBool32 pres = VK_TRUE;
            if (s_surface) vkGetPhysicalDeviceSurfaceSupportKHR(devs[i], q, s_surface, &pres);
            if (!pres) continue;
            const int score = pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 :
                              pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
            if (score > best_score) { best = (int)i; best_score = score; best_q = q; }
            break;
        }
    }
    if (best < 0) { fprintf(stderr, "[RSX vulkan] no Vulkan 1.3 device with a graphics queue\n"); return -1; }
    s_phys = devs[best]; s_qfam = best_q;
    vkGetPhysicalDeviceProperties(s_phys, &s_props);
    vkGetPhysicalDeviceMemoryProperties(s_phys, &s_memprops);
    fprintf(stderr, "[rsx engine/vulkan] device: %s (Vulkan %u.%u.%u)\n", s_props.deviceName,
            VK_API_VERSION_MAJOR(s_props.apiVersion), VK_API_VERSION_MINOR(s_props.apiVersion), VK_API_VERSION_PATCH(s_props.apiVersion));
    {   VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(s_phys, ENG_DEPTH_FMT, &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
            fprintf(stderr, "[RSX vulkan] D32_SFLOAT_S8_UINT depth is not supported\n"); return -1; } }

    u32 ne = 0; vkEnumerateDeviceExtensionProperties(s_phys, NULL, &ne, NULL);
    VkExtensionProperties* ep = (VkExtensionProperties*)calloc(ne ? ne : 1, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(s_phys, NULL, &ne, ep);
    const char* dext[4]; u32 ndext = 0;
    if (!s_headless) dext[ndext++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    s_has_maint5 = has_ext(ep, ne, "VK_KHR_maintenance5");
    if (s_has_maint5) dext[ndext++] = "VK_KHR_maintenance5";   /* point size 1 without the shader writing it */
    free(ep);

    VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceMaintenance5FeaturesKHR fm5 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR };
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2.pNext = &f12; f12.pNext = &f13; f13.pNext = s_has_maint5 ? &fm5 : NULL;
    vkGetPhysicalDeviceFeatures2(s_phys, &f2);
    if (!f13.dynamicRendering || !f12.timelineSemaphore) { fprintf(stderr, "[RSX vulkan] dynamic rendering / timeline semaphores missing\n"); return -1; }
    VkPhysicalDeviceVulkan13Features e13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceVulkan12Features e12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceMaintenance5FeaturesKHR em5 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR };
    VkPhysicalDeviceFeatures2 e2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    e2.pNext = &e12; e12.pNext = &e13; e13.pNext = (s_has_maint5 && fm5.maintenance5) ? &em5 : NULL;
    e13.dynamicRendering = VK_TRUE;
    e12.timelineSemaphore = VK_TRUE;
    e12.samplerMirrorClampToEdge = f12.samplerMirrorClampToEdge;
    em5.maintenance5 = fm5.maintenance5;
    e2.features.textureCompressionBC = f2.features.textureCompressionBC;
    e2.features.occlusionQueryPrecise = f2.features.occlusionQueryPrecise;
    s_query_precise = f2.features.occlusionQueryPrecise ? 1 : 0;
    e2.features.independentBlend = f2.features.independentBlend;
    e2.features.samplerAnisotropy = f2.features.samplerAnisotropy;
    s_has_aniso = f2.features.samplerAnisotropy ? 1 : 0;
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qi.queueFamilyIndex = s_qfam; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &e2;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qi;
    dci.enabledExtensionCount = ndext; dci.ppEnabledExtensionNames = dext;
    VKCHECK(vkCreateDevice(s_phys, &dci, NULL, &s_dev), "vkCreateDevice");
#define VK_LOAD_D(n) n = (PFN_##n)vkGetDeviceProcAddr(s_dev, #n);
    VK_DEVICE_FNS(VK_LOAD_D)
#undef VK_LOAD_D
    vkGetDeviceQueue(s_dev, s_qfam, 0, &s_queue);

    /* The timeline, the per-slot pools, query pools and semaphores. */
    VkSemaphoreTypeCreateInfo st = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    sci.pNext = &st;
    VKCHECK(vkCreateSemaphore(s_dev, &sci, NULL, &s_timeline), "timeline semaphore");
    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpi.queueFamilyIndex = s_qfam; cpi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkDescriptorPoolSize ps[3] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 * ENG_SETS_PER_SLOT },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 35 * ENG_SETS_PER_SLOT },
        { VK_DESCRIPTOR_TYPE_SAMPLER, 20 * ENG_SETS_PER_SLOT } };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpi.maxSets = ENG_SETS_PER_SLOT; dpi.poolSizeCount = 3; dpi.pPoolSizes = ps;
    VkQueryPoolCreateInfo qpi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpi.queryType = VK_QUERY_TYPE_OCCLUSION; qpi.queryCount = ENG_QUERIES_PER_SLOT;
    for (int i = 0; i < ENG_FRAMES; i++) {
        VKCHECK(vkCreateCommandPool(s_dev, &cpi, NULL, &s_pool[i]), "command pool");
        VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        ai.commandPool = s_pool[i]; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        VKCHECK(vkAllocateCommandBuffers(s_dev, &ai, &s_cmdbuf[i]), "command buffer");
        VKCHECK(vkCreateDescriptorPool(s_dev, &dpi, NULL, &s_dpool[i]), "descriptor pool");
        VKCHECK(vkCreateQueryPool(s_dev, &qpi, NULL, &s_qpool[i]), "query pool");
        s_qbuf[i] = make_buffer((VkDeviceSize)ENG_QUERIES_PER_SLOT * 8u, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST_MEM, &s_qmem[i]);
        if (!s_qbuf[i]) return -1;
        s_qmap[i] = (u64*)mem_ptr(&s_qmem[i]);
        VkSemaphoreCreateInfo bsi = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VKCHECK(vkCreateSemaphore(s_dev, &bsi, NULL, &s_sem_acq[i]), "semaphore");
    }
    VKCHECK(vkCreateCommandPool(s_dev, &cpi, NULL, &s_aux_pool), "aux command pool");
    { VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
      ai.commandPool = s_aux_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
      VKCHECK(vkAllocateCommandBuffers(s_dev, &ai, &s_aux_cmd), "aux command buffer"); }

    /* The one set layout and pipeline layout (rsx_shader_spirv.h's plan). */
    VkDescriptorSetLayoutBinding b[2 + 1 + 15 + 4 + 1 + 4]; u32 nb = 0;
#define BIND(bn, type, cnt, stages) do { VkDescriptorSetLayoutBinding* x = &b[nb++]; memset(x, 0, sizeof *x); \
        x->binding = (bn); x->descriptorType = (type); x->descriptorCount = (cnt); x->stageFlags = (stages); } while (0)
    BIND(RSX_SPV_BIND_VS_CB, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT);
    BIND(RSX_SPV_BIND_FS_CB, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    BIND(RSX_SPV_BIND_FS_TEX, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 16, VK_SHADER_STAGE_FRAGMENT_BIT);
    for (u32 u = 1; u < 16; u++) BIND(RSX_SPV_BIND_FS_TEX + u, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    for (u32 u = 0; u < 4; u++) BIND(RSX_SPV_BIND_VS_TEX + u, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_VERTEX_BIT);
    BIND(RSX_SPV_BIND_FS_SAMP, VK_DESCRIPTOR_TYPE_SAMPLER, 16, VK_SHADER_STAGE_FRAGMENT_BIT);
    for (u32 u = 0; u < 4; u++) BIND(RSX_SPV_BIND_VS_SAMP + u, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT);
#undef BIND
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dli.bindingCount = nb; dli.pBindings = b;
    VKCHECK(vkCreateDescriptorSetLayout(s_dev, &dli, NULL, &s_dsl), "descriptor set layout");
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1; pli.pSetLayouts = &s_dsl;
    VKCHECK(vkCreatePipelineLayout(s_dev, &pli, NULL, &s_pl), "pipeline layout");

    /* Shared bits: a zero constant buffer, a point sampler, the null textures. */
    s_zero_buf = make_buffer(ENG_UBO_RANGE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, HOST_MEM, &s_zero_mem);
    if (!s_zero_buf) return -1;
    memset(mem_ptr(&s_zero_mem), 0, ENG_UBO_RANGE);
    { VkSamplerCreateInfo si = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
      si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.maxAnisotropy = 1.0f;
      VKCHECK(vkCreateSampler(s_dev, &si, NULL, &s_point_sampler), "sampler"); }
    s_null_tex = eng_image(OBJ_TEXTURE, VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    s_null_cube = eng_image(OBJ_TEXTURE, VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 6, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (!s_null_tex || !s_null_cube) return -1;
    s_obj[s_null_tex - 1].view = make_view(s_obj[s_null_tex - 1].img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, k_identity);
    s_obj[s_null_cube - 1].view = make_view(s_obj[s_null_cube - 1].img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT, 1, 6, k_identity);
    {   /* Cleared to black and put in the read layout once. */
        vkResetCommandPool(s_dev, s_aux_pool, 0);
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(s_aux_cmd, &bi);
        s_cmd = s_aux_cmd;
        const u32 nulls[2] = { s_null_tex, s_null_cube };
        for (int k = 0; k < 2; k++) {
            EngObj* o = &s_obj[nulls[k] - 1];
            obj_layout(o, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL); bar_flush();
            VkClearColorValue cv; memset(&cv, 0, sizeof cv);
            VkImageSubresourceRange rg = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, o->faces };
            vkCmdClearColorImage(s_cmd, o->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &rg);
            obj_layout(o, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        bar_flush();
        vkEndCommandBuffer(s_aux_cmd);
        const u64 f = ++s_fence_value;
        VkTimelineSemaphoreSubmitInfo ts = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        ts.signalSemaphoreValueCount = 1; ts.pSignalSemaphoreValues = &f;
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.pNext = &ts; si.commandBufferCount = 1; si.pCommandBuffers = &s_aux_cmd;
        si.signalSemaphoreCount = 1; si.pSignalSemaphores = &s_timeline;
        VKCHECK(vkQueueSubmit(s_queue, 1, &si, VK_NULL_HANDLE), "init submit");
        fence_wait(f);
    }

    /* The depth resolve helpers. */
    s_helper_vs = eng_module(kHelperVS, RSX_SPV_STAGE_VERTEX, "helper vs");
    s_helper_ps_depth = eng_module(kHelperDepthPS, RSX_SPV_STAGE_FRAGMENT, "helper depth");
    s_helper_ps_pack = eng_module(kHelperPackPS, RSX_SPV_STAGE_FRAGMENT, "helper pack");
    s_depth_pso = eng_helper_pso(s_helper_ps_depth, VK_FORMAT_R32_SFLOAT);
    s_depth_pack_pso = eng_helper_pso(s_helper_ps_pack, VK_FORMAT_R8G8B8A8_UNORM);

    if (!s_headless && eng_swapchain_create() != 0) { fprintf(stderr, "[RSX vulkan] swap chain creation failed\n"); return -1; }
    return 0;
}

static void eng_shutdown(void* user)
{
    (void)user;
    if (s_dev) vkDeviceWaitIdle(s_dev);
    for (u32 i = 0; i < s_obj_count; i++) if (s_obj[i].kind) eng_obj_destroy(&s_obj[i]);
    s_obj_count = s_obj_free_count = s_retired_count = 0;
    for (u32 i = 0; i < s_pipe_count; i++) {
        for (int k = 0; k < 3; k++) {
            if (s_pipe[i].pso[k]) vkDestroyPipeline(s_dev, s_pipe[i].pso[k], NULL);
            if (s_pipe[i].pso1[k]) vkDestroyPipeline(s_dev, s_pipe[i].pso1[k], NULL);
        }
        free(s_pipe[i].ps_plain);
    }
    s_pipe_count = 0;
    for (u32 i = 0; i < s_mod_count; i++) vkDestroyShaderModule(s_dev, s_mod[i].mod, NULL);
    s_mod_count = 0;
    for (u32 i = 0; i < s_samp_count; i++) vkDestroySampler(s_dev, s_samp[i].smp, NULL);
    s_samp_count = s_view_count = s_rec_count = 0;
    for (u32 i = 0; i < s_buf_count; i++) if (s_buf[i]) { vkDestroyBuffer(s_dev, s_buf[i], NULL); mem_free(&s_buf_mem[i]); s_buf[i] = VK_NULL_HANDLE; }
    for (u32 i = 0; i < s_buf_pool_n; i++) { vkDestroyBuffer(s_dev, s_buf_pool[i].buf, NULL); mem_free(&s_buf_pool[i].mem); }
    s_buf_pool_n = 0; s_buf_count = s_buf_free_count = 0;
    for (int i = 0; i < ENG_STAGE_POOL; i++) eng_stage_free(&s_stage[i]);
    s_stage_cur = -1; s_stage_used = 0;
    for (int i = 0; i < ENG_MAX_SUBMITS; i++) { free(s_sub[i].q_vis); free(s_sub[i].reports); memset(&s_sub[i], 0, sizeof s_sub[i]); }
    s_vs_cb_seq = ~0u; s_vis_npending = 0; s_fallback_n = 0;
    s_active = 0;
}

static void eng_release_device(void)
{
    if (!s_inst) return;
    if (s_dev) {
        vkDeviceWaitIdle(s_dev);
        if (s_depth_pso) vkDestroyPipeline(s_dev, s_depth_pso, NULL);
        if (s_depth_pack_pso) vkDestroyPipeline(s_dev, s_depth_pack_pso, NULL);
        if (s_point_sampler) vkDestroySampler(s_dev, s_point_sampler, NULL);
        if (s_zero_buf) vkDestroyBuffer(s_dev, s_zero_buf, NULL);
        if (s_rb_buf) vkDestroyBuffer(s_dev, s_rb_buf, NULL);
        if (s_pl) vkDestroyPipelineLayout(s_dev, s_pl, NULL);
        if (s_dsl) vkDestroyDescriptorSetLayout(s_dev, s_dsl, NULL);
        for (int i = 0; i < ENG_FRAMES; i++) {
            if (s_pool[i]) vkDestroyCommandPool(s_dev, s_pool[i], NULL);
            if (s_dpool[i]) vkDestroyDescriptorPool(s_dev, s_dpool[i], NULL);
            if (s_qpool[i]) vkDestroyQueryPool(s_dev, s_qpool[i], NULL);
            if (s_qbuf[i]) vkDestroyBuffer(s_dev, s_qbuf[i], NULL);
            if (s_sem_acq[i]) vkDestroySemaphore(s_dev, s_sem_acq[i], NULL);
        }
        for (int i = 0; i < 8; i++) if (s_sem_done[i]) vkDestroySemaphore(s_dev, s_sem_done[i], NULL);
        if (s_aux_pool) vkDestroyCommandPool(s_dev, s_aux_pool, NULL);
        if (s_timeline) vkDestroySemaphore(s_dev, s_timeline, NULL);
        if (s_swap) vkDestroySwapchainKHR(s_dev, s_swap, NULL);
        for (u32 b = 0; b < s_nblk; b++) if (s_blk[b].mem) { vkFreeMemory(s_dev, s_blk[b].mem, NULL); free(s_blk[b].fr); }
        memset(s_blk, 0, sizeof s_blk); s_nblk = 0;
        vkDestroyDevice(s_dev, NULL);
    }
    if (s_surface) vkDestroySurfaceKHR(s_inst, s_surface, NULL);
    if (s_dbg && vkDestroyDebugUtilsMessengerEXT) vkDestroyDebugUtilsMessengerEXT(s_inst, s_dbg, NULL);
    vkDestroyInstance(s_inst, NULL);
    s_inst = VK_NULL_HANDLE; s_dev = VK_NULL_HANDLE; s_surface = VK_NULL_HANDLE; s_swap = VK_NULL_HANDLE;
    if (s_hwnd) { DestroyWindow(s_hwnd); s_hwnd = NULL; }
}

/* ---- public API ---------------------------------------------------------------------- */

int rsx_vulkan_engine_wanted(void)
{
    const char* b = getenv("RSX_BACKEND");
    return b && !_stricmp(b, "vulkan");
}

int rsx_vulkan_engine_init(u32 width, u32 height, const char* title)
{
    if (!rsx_hlsl_to_spirv_available()) { fprintf(stderr, "[RSX vulkan] built without glslang\n"); return -1; }
    const char* hl = getenv("PS3RECOMP_VULKAN_HEADLESS");
    if (!hl || !*hl) hl = getenv("PS3RECOMP_D3D12_HEADLESS");
    s_headless = (hl && *hl && *hl != '0');
    s_width = width ? width : 1280; s_height = height ? height : 720;
    { const char* v = getenv("RSX_VSYNC"); if (v && *v) s_vsync = atoi(v) ? 1 : 0; }
    eng_read_scale();
    { HMODULE u32m = GetModuleHandleA("user32.dll");
      typedef BOOL (WINAPI *SetCtxFn)(DPI_AWARENESS_CONTEXT);
      SetCtxFn set_ctx = u32m ? (SetCtxFn)(void*)GetProcAddress(u32m, "SetProcessDpiAwarenessContext") : NULL;
      if (!set_ctx || !set_ctx(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) SetProcessDPIAware(); }
    s_win_w = s_width; s_win_h = s_height;
    { const char* dm = getenv("RSX_DISPLAY");
      const char* fs = getenv("RSX_FULLSCREEN");
      s_display = DISP_WINDOWED;
      if (dm && *dm) {
          if (!_stricmp(dm, "borderless") || !_stricmp(dm, "fullscreen") || !_stricmp(dm, "exclusive")) s_display = DISP_BORDERLESS;
      } else if (fs && *fs && *fs != '0') s_display = DISP_BORDERLESS;
      const char* ws = getenv("RSX_WINDOW");
      unsigned ww = 0, wh = 0;
      const int have_ws = ws && *ws && sscanf(ws, "%ux%u", &ww, &wh) == 2 && ww >= 320 && wh >= 240 && ww <= 16384 && wh <= 16384;
      if (s_display == DISP_WINDOWED) { if (have_ws) { s_win_w = ww; s_win_h = wh; } }
      else { s_win_w = (u32)GetSystemMetrics(SM_CXSCREEN); s_win_h = (u32)GetSystemMetrics(SM_CYSCREEN); } }

    rsx_draw_engine_set_backend(&s_engine_backend);
    rsx_draw_engine_set_default(1);
    if (!rsx_draw_engine_enabled()) { rsx_draw_engine_set_backend(NULL); return -1; }
    if (!s_headless) {
        s_hwnd = eng_create_window(s_win_w, s_win_h, title);
        if (!s_hwnd) { fprintf(stderr, "[RSX vulkan] window creation failed\n"); rsx_draw_engine_set_backend(NULL); return -1; }
    }
    if (eng_init_device() != 0) {
        fprintf(stderr, "[RSX vulkan] device initialisation failed\n");
        eng_release_device();
        rsx_draw_engine_set_backend(NULL);
        return -1;
    }
    s_ready = 1;
    if (rsx_draw_engine_init(s_width, s_height) != 0) {
        fprintf(stderr, "[RSX vulkan] draw engine init failed\n");
        eng_release_device();
        rsx_draw_engine_set_backend(NULL);
        return -1;
    }
    s_active = 1;
    fprintf(stderr, "[RSX vulkan] %s (%ux%u), register-file draw engine, vsync %d, internal resolution x%.2f (%ux%u)\n",
            s_headless ? "headless" : s_display == DISP_BORDERLESS ? "borderless" : "windowed",
            s_win_w, s_win_h, s_vsync, s_scale, sc_dim(s_width), sc_dim(s_height));
    return 0;
}

void rsx_vulkan_engine_shutdown(void)
{
    if (!s_active) return;
    rsx_draw_engine_shutdown();
    rsx_draw_engine_set_backend(NULL);
    eng_shutdown(NULL);
    eng_release_device();
    s_active = 0;
}

int rsx_vulkan_engine_active(void) { return s_active; }

int rsx_vulkan_engine_pump_messages(void)
{
    if (s_headless) return 0;
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) return -1;
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return s_window_closed ? -1 : 0;
}
