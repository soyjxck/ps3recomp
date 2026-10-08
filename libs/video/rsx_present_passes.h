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
