/* Present-time passes shared by the D3D12 and Vulkan engines (HLSL; the
 * Vulkan engine lowers it through glslang). Fullscreen triangle, t0 the
 * frame, s1 a linear clamp sampler (s0 is the helpers' point one).
 *
 * ps_blit_area: the present blit. Scaling down -- an internal resolution
 * above the window's, which is what the Resolution setting's supersampling
 * is -- it averages taps spread over each window pixel's whole footprint in
 * the frame (exact box filter at 2x and 3x) instead of one bilinear tap.
 * Equal or upscaling it is the one tap it was.
 *
 * ps_fxaa: fast approximate anti-aliasing on the finished frame, at its own
 * resolution, before the blit (RSX_AA=fxaa). This port's implementation of
 * the published algorithm (Lottes, 2009): luma contrast decides whether a
 * pixel is on an edge; the edge's direction and its two ends along it are
 * searched for; the pixel is re-sampled across the edge by how far it sits
 * from the nearer end, and blended for sub-pixel aliasing. */
/* The present blit for an HDR swap chain (rsx_d3d12_engine.c, RSX_HDR):
 * scRGB, linear light with 1.0 at 80 nits. The frame is an SDR picture: each
 * of its pixels is decoded to linear light (a 2.2 power, what an SDR display
 * shows it with; hdr.z = 1: the sRGB curve) and scaled to the paper white
 * (hdr.x = its nits / 80). With hdr.w = 1, t1 is the HDR plane the title's
 * own composite wrote (packed: x its HDR colour as RGB9E5, linear light with
 * 1 at SDR white; y the 8-bit colour it wrote to the frame, bit 24 set): a
 * pixel whose frame colour is still that -- nothing drawn over it since, as
 * the HUD is -- is the plane's colour instead. hdr2.y is how much of the
 * plane's own hue is kept: 1 all of it, 0 the frame pixel's colour at the
 * plane's luminance (the SDR picture's whitening of bright colours). Pixels
 * are resolved one by one before they are filtered: scaling down, an
 * average over each window pixel's footprint (as ps_blit_area); at the same
 * size, the pixel itself; scaling up, the four around the tap blended. Above
 * a shoulder (a third of the peak, at least the paper white) the brightest
 * channel rolls off toward the peak (hdr.y = its nits / 80) on the PQ curve,
 * the hue kept, instead of clipping. hdr.w = 2 (RSX_HDR_DEBUG): bands
 * instead -- grey where the plane has nothing, purple where something was
 * drawn over it, blue, green, yellow and orange above 1, 2, 4 and 8 times the
 * paper white, red near the peak. hdr2.x = 1: the output is HDR10 instead --
 * the BT.2020 primaries and the PQ curve (ST 2084), for a swap chain that has
 * no scRGB (rsx_vulkan_engine.c). */
static const char kPresentHdrHLSL[] =
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "VSOut vs_main(uint vid : SV_VertexID) {\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    VSOut o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); o.uv = float2(p.x, 1.0 - p.y); return o;\n"
    "}\n"
    "Texture2D<float4> src : register(t0);\n"
    "SamplerState smp : register(s1);\n"
    "Texture2D<uint2> plane : register(t1);\n"
    "cbuffer HdrConstants : register(b0) { float4 hdr; float4 hdr2; };\n"
    "float3 decode(float3 c) {\n"
    "    c = saturate(c);\n"
    "    if (hdr.z > 0.5) return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);\n"
    "    return pow(c, 2.2);\n"
    "}\n"
    "float3 rgb9e5(uint p) {\n"
    "    return float3(p & 511u, (p >> 9) & 511u, (p >> 18) & 511u) * exp2((float)(p >> 27) - 24.0);\n"
    "}\n"
    "float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }\n"
    "float3 texel(int2 t) {\n"
    "    const float3 s = saturate(src.Load(int3(t, 0)).rgb);\n"
    "    const float3 lin = decode(s);\n"
    "    if (hdr.w < 0.5) return lin;\n"
    "    const uint2 p = plane.Load(int3(t, 0));\n"
    "    const float3 q = float3(p.y & 255u, (p.y >> 8) & 255u, (p.y >> 16) & 255u);\n"
    "    const float3 dd = abs(s * 255.0 - q);\n"
    "    const bool written = (p.y >> 24) != 0u;\n"
    "    if (!written || max(max(dd.x, dd.y), dd.z) > 3.5) {\n"
    "        if (hdr.w > 1.5) return written ? float3(0.3, 0.0, 0.3) : lin * 0.3;\n"
    "        return lin;\n"
    "    }\n"
    "    float3 h = rgb9e5(p.x) + (lin - pow(s, 2.2));\n"
    "    const float yl = luma(lin);\n"
    "    const float3 native = yl > 1e-5 ? lin * (luma(h) / yl) : h;\n"
    "    h = max(lerp(native, h, hdr2.y), 0.0);\n"
    "    if (hdr.w > 1.5) {\n"
    "        const float r = max(max(h.r, h.g), h.b);\n"
    "        float3 c = lin * 0.3;\n"
    "        if (r > 1.02) c = float3(0.0, 0.0, 1.0);\n"
    "        if (r > 2.0) c = float3(0.0, 1.0, 0.0);\n"
    "        if (r > 4.0) c = float3(1.0, 1.0, 0.0);\n"
    "        if (r > 8.0) c = float3(1.0, 0.5, 0.0);\n"
    "        if (r * hdr.x > 0.9 * hdr.y) c = float3(1.0, 0.0, 0.0);\n"
    "        return c;\n"
    "    }\n"
    "    return h;\n"
    "}\n"
    "float pq(float x) {\n"
    "    const float p = pow(saturate(x), 0.1593017578125);\n"
    "    return pow((0.8359375 + 18.8515625 * p) / (1.0 + 18.6875 * p), 78.84375);\n"
    "}\n"
    "float pq_inv(float e) {\n"
    "    const float p = pow(saturate(e), 1.0 / 78.84375);\n"
    "    return pow(max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);\n"
    "}\n"
    "float3 hdr10(float3 o) {\n"
    "    const float3x3 m = float3x3(0.6274040, 0.3292820, 0.0433136,\n"
    "                                0.0690970, 0.9195400, 0.0113612,\n"
    "                                0.0163916, 0.0880132, 0.8955950);\n"
    "    const float3 p = pow(saturate(mul(m, o) * (80.0 / 10000.0)), 0.1593017578125);\n"
    "    return pow((0.8359375 + 18.8515625 * p) / (1.0 + 18.6875 * p), 78.84375);\n"
    "}\n"
    "float4 ps_blit_area_hdr(VSOut i) : SV_Target {\n"
    "    uint w, h; src.GetDimensions(w, h);\n"
    "    const float2 size = float2(w, h);\n"
    "    const int2 last = int2(w, h) - 1;\n"
    "    const float2 d = float2(ddx(i.uv.x), ddy(i.uv.y));\n"
    "    const float2 fp = abs(d) * size;\n"
    "    float3 lin;\n"
    "    if (fp.x <= 1.25 && fp.y <= 1.25) {\n"
    "        const float2 p = i.uv * size - 0.5;\n"
    "        const float2 c = round(p);\n"
    "        if (abs(p.x - c.x) < 0.002 && abs(p.y - c.y) < 0.002) lin = texel(clamp((int2)c, 0, last));\n"
    "        else {\n"
    "            const int2 t = (int2)floor(p);\n"
    "            const float2 f = p - floor(p);\n"
    "            const float3 a = texel(clamp(t, 0, last)), b = texel(clamp(t + int2(1, 0), 0, last));\n"
    "            const float3 e = texel(clamp(t + int2(0, 1), 0, last)), g = texel(clamp(t + int2(1, 1), 0, last));\n"
    "            lin = lerp(lerp(a, b, f.x), lerp(e, g, f.x), f.y);\n"
    "        }\n"
    "    } else {\n"
    "        const int nx = clamp((int)ceil(fp.x - 0.01), 1, 4), ny = clamp((int)ceil(fp.y - 0.01), 1, 4);\n"
    "        float3 acc = 0;\n"
    "        for (int y = 0; y < ny; y++)\n"
    "            for (int x = 0; x < nx; x++)\n"
    "                acc += texel(clamp((int2)floor((i.uv + (float2(x + 0.5, y + 0.5) / float2(nx, ny) - 0.5) * d) * size), 0, last));\n"
    "        lin = acc / (float)(nx * ny);\n"
    "    }\n"
    "    float3 o = lin * hdr.x;\n"
    "    if (hdr.w < 1.5) {\n"
    "        const float m = max(max(o.r, o.g), o.b);\n"
    "        const float sh = min(max(hdr.y / 3.0, hdr.x), hdr.y * 0.9);\n"
    "        if (m > sh) {\n"
    "            const float s = pq(sh * 0.008), top = pq(hdr.y * 0.008), l = pq(m * 0.008);\n"
    "            const float room = max(top - s, 1e-4);\n"
    "            o *= pq_inv(s + room * (1.0 - exp(-(l - s) / room))) * 125.0 / m;\n"
    "        }\n"
    "    }\n"
    "    return float4(hdr2.x > 0.5 ? hdr10(o) : o, 1.0);\n"
    "}\n";

/* The host overlay (rsx_overlay_set: a port's pop-ups) over an HDR back buffer
 * (rsx_d3d12_engine.c): the overlay draws as it would into an SDR target --
 * an RGBA8 one cleared to nothing, so its colour is premultiplied by its
 * cover -- and this pass lays it over the scRGB frame: each pixel
 * un-premultiplied, decoded as the frame is (hdr.z: the sRGB curve, else a
 * 2.2 power), at the paper white (hdr.x = its nits / 80), and drawn with
 * premultiplied alpha (ONE, INV_SRC_ALPHA). */
static const char kOverlayHdrHLSL[] =
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "VSOut vs_main(uint vid : SV_VertexID) {\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    VSOut o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); o.uv = float2(p.x, 1.0 - p.y); return o;\n"
    "}\n"
    "Texture2D<float4> src : register(t0);\n"
    "cbuffer HdrConstants : register(b0) { float4 hdr; float4 hdr2; };\n"
    "float4 ps_overlay_hdr(VSOut i) : SV_Target {\n"
    "    const float4 c = src.Load(int3(i.pos.xy, 0));\n"
    "    if (c.a <= 0.0) return float4(0.0, 0.0, 0.0, 0.0);\n"
    "    const float3 e = saturate(c.rgb / c.a);\n"
    "    const float3 lin = hdr.z > 0.5 ? (e <= 0.04045 ? e / 12.92 : pow((e + 0.055) / 1.055, 2.4)) : pow(e, 2.2);\n"
    "    return float4(lin * hdr.x * c.a, c.a);\n"
    "}\n";

static const char kPresentHLSL[] =
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "VSOut vs_main(uint vid : SV_VertexID) {\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    VSOut o; o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0); o.uv = float2(p.x, 1.0 - p.y); return o;\n"
    "}\n"
    "Texture2D<float4> src : register(t0);\n"
    "SamplerState smp : register(s1);\n"
    "float4 ps_blit_area(VSOut i) : SV_Target {\n"
    "    uint w, h; src.GetDimensions(w, h);\n"
    "    float2 d = float2(ddx(i.uv.x), ddy(i.uv.y));\n"
    "    float2 fp = abs(d) * float2(w, h);\n"
    "    if (fp.x <= 1.25 && fp.y <= 1.25) return src.SampleLevel(smp, i.uv, 0);\n"
    "    int nx = clamp((int)ceil(fp.x - 0.01), 1, 4), ny = clamp((int)ceil(fp.y - 0.01), 1, 4);\n"
    "    float4 acc = 0;\n"
    "    for (int y = 0; y < ny; y++)\n"
    "        for (int x = 0; x < nx; x++)\n"
    "            acc += src.SampleLevel(smp, i.uv + (float2(x + 0.5, y + 0.5) / float2(nx, ny) - 0.5) * d, 0);\n"
    "    return acc / (float)(nx * ny);\n"
    "}\n"
    "float fxaa_luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }\n"
    "float fxaa_l(float2 uv) { return fxaa_luma(src.SampleLevel(smp, uv, 0).rgb); }\n"
    "float4 ps_fxaa(VSOut i) : SV_Target {\n"
    "    uint w, h; src.GetDimensions(w, h);\n"
    "    float2 rcp = 1.0 / float2(w, h);\n"
    "    float2 uv = i.uv;\n"
    "    float4 rgbM = src.SampleLevel(smp, uv, 0);\n"
    "    float lM = fxaa_luma(rgbM.rgb);\n"
    "    float lN = fxaa_l(uv + float2(0, -rcp.y)), lS = fxaa_l(uv + float2(0, rcp.y));\n"
    "    float lW = fxaa_l(uv + float2(-rcp.x, 0)), lE = fxaa_l(uv + float2(rcp.x, 0));\n"
    "    float lMax = max(max(max(lN, lS), max(lW, lE)), lM);\n"
    "    float lMin = min(min(min(lN, lS), min(lW, lE)), lM);\n"
    "    float range = lMax - lMin;\n"
    "    if (range < max(0.0312, lMax * 0.125)) return float4(rgbM.rgb, 1.0);\n"
    "    float lNW = fxaa_l(uv - rcp), lSE = fxaa_l(uv + rcp);\n"
    "    float lNE = fxaa_l(uv + float2(rcp.x, -rcp.y)), lSW = fxaa_l(uv + float2(-rcp.x, rcp.y));\n"
    "    float lNS = lN + lS, lWE = lW + lE;\n"
    "    float sub = saturate(abs((2.0 * (lNS + lWE) + lNW + lNE + lSW + lSE) / 12.0 - lM) / range);\n"
    "    sub = (-2.0 * sub + 3.0) * sub * sub;\n"
    "    float subpix = sub * sub * 0.75;\n"
    "    float edgeH = abs(-2.0 * lW + lNW + lSW) + 2.0 * abs(-2.0 * lM + lNS) + abs(-2.0 * lE + lNE + lSE);\n"
    "    float edgeV = abs(-2.0 * lN + lNW + lNE) + 2.0 * abs(-2.0 * lM + lWE) + abs(-2.0 * lS + lSW + lSE);\n"
    "    bool horz = edgeH >= edgeV;\n"
    "    float l1 = horz ? lN : lW, l2 = horz ? lS : lE;\n"
    "    float g1 = l1 - lM, g2 = l2 - lM;\n"
    "    bool pairN = abs(g1) >= abs(g2);\n"
    "    float grad = max(abs(g1), abs(g2));\n"
    "    float stepL = horz ? rcp.y : rcp.x;\n"
    "    if (pairN) stepL = -stepL;\n"
    "    float lAvg = 0.5 * ((pairN ? l1 : l2) + lM);\n"
    "    float2 posB = uv;\n"
    "    if (horz) posB.y += stepL * 0.5; else posB.x += stepL * 0.5;\n"
    "    float2 off = horz ? float2(rcp.x, 0) : float2(0, rcp.y);\n"
    "    float2 posN = posB - off, posP = posB + off;\n"
    "    float gs = grad * 0.25;\n"
    "    float eN = fxaa_l(posN) - lAvg, eP = fxaa_l(posP) - lAvg;\n"
    "    bool dN = abs(eN) >= gs, dP = abs(eP) >= gs;\n"
    "    const float q[11] = { 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0, 8.0 };\n"
    "    [loop] for (int k = 0; k < 11 && !(dN && dP); k++) {\n"
    "        if (!dN) { posN -= off * q[k]; eN = fxaa_l(posN) - lAvg; dN = abs(eN) >= gs; }\n"
    "        if (!dP) { posP += off * q[k]; eP = fxaa_l(posP) - lAvg; dP = abs(eP) >= gs; }\n"
    "    }\n"
    "    float distN = horz ? uv.x - posN.x : uv.y - posN.y;\n"
    "    float distP = horz ? posP.x - uv.x : posP.y - uv.y;\n"
    "    bool nearN = distN < distP;\n"
    "    float eEnd = nearN ? eN : eP;\n"
    "    bool good = (eEnd < 0.0) != ((lM - lAvg) < 0.0);\n"
    "    float px = good ? 0.5 - min(distN, distP) / (distN + distP) : 0.0;\n"
    "    px = max(px, subpix);\n"
    "    float2 fuv = uv;\n"
    "    if (horz) fuv.y += px * stepL; else fuv.x += px * stepL;\n"
    "    return float4(src.SampleLevel(smp, fuv, 0).rgb, 1.0);\n"
    "}\n";
