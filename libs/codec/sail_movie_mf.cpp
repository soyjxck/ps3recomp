/*
 * ps3recomp - host movie decoding for cellSail through Media Foundation.
 * See sail_movie.h. Two source readers on the same file: one for video
 * (decoded a frame at a time as the title's clock reaches it), one for audio
 * (decoded in full at open), so neither stream has to buffer the other.
 */
#include "sail_movie.h"

#ifdef _WIN32

#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <Mferror.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

struct sail_movie {
    IMFSourceReader* video = nullptr;
    int w = 0, h = 0;
    long long duration_us = 0;
    std::vector<unsigned char> frame;   /* the current frame, NV12, tight */
    std::vector<unsigned char> next;    /* read ahead: the following frame */
    long long next_us = -1;             /* its time; -1 none loaded */
    bool have_frame = false, ended = false;
    std::vector<float> audio;           /* float stereo 48 kHz */
};

static bool s_mf_started = false;

static bool mf_start(void)
{
    if (s_mf_started) return true;
    /* The calling thread may already be in an apartment; either is fine. */
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    (void)hr;
    if (FAILED(MFStartup(MF_VERSION, MFSTARTUP_LITE))) return false;
    s_mf_started = true;
    return true;
}

static IMFSourceReader* open_reader(const wchar_t* path, bool video)
{
    IMFAttributes* at = nullptr;
    MFCreateAttributes(&at, 1);
    if (at && video) at->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
    IMFSourceReader* r = nullptr;
    HRESULT hr = MFCreateSourceReaderFromURL(path, at, &r);
    if (at) at->Release();
    if (FAILED(hr)) return nullptr;
    r->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, FALSE);
    const DWORD s = video ? (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM
                          : (DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM;
    if (FAILED(r->SetStreamSelection(s, TRUE))) { r->Release(); return nullptr; }
    IMFMediaType* t = nullptr;
    MFCreateMediaType(&t);
    if (video) {
        t->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        t->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
    } else {
        t->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
        t->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);
        t->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, 2);
        t->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, 48000);
        t->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32);
        t->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, 8);
        t->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 48000 * 8);
    }
    hr = r->SetCurrentMediaType(s, nullptr, t);
    t->Release();
    if (FAILED(hr)) { r->Release(); return nullptr; }
    return r;
}

/* Copy one decoded sample's frame into dst as tight NV12. */
static bool copy_frame(sail_movie* m, IMFSample* s, std::vector<unsigned char>& dst)
{
    IMFMediaBuffer* b = nullptr;
    if (FAILED(s->ConvertToContiguousBuffer(&b))) return false;
    const size_t ysz = (size_t)m->w * m->h;
    dst.resize(ysz * 3 / 2);
    bool ok = false;
    IMF2DBuffer* b2 = nullptr;
    BYTE* p = nullptr; LONG pitch = 0;
    if (SUCCEEDED(b->QueryInterface(IID_PPV_ARGS(&b2))) && SUCCEEDED(b2->Lock2D(&p, &pitch))) {
        if (pitch > 0) {
            /* The U/V plane follows the Y plane, which may have padded rows. */
            for (int y = 0; y < m->h; y++)
                memcpy(&dst[(size_t)y * m->w], p + (size_t)y * pitch, (size_t)m->w);
            const BYTE* uv = p + (size_t)pitch * m->h;
            for (int y = 0; y < m->h / 2; y++)
                memcpy(&dst[ysz + (size_t)y * m->w], uv + (size_t)y * pitch, (size_t)m->w);
            ok = true;
        }
        b2->Unlock2D();
    }
    if (b2) b2->Release();
    if (!ok) {
        DWORD len = 0;
        if (SUCCEEDED(b->Lock(&p, nullptr, &len))) {
            if (len >= dst.size()) { memcpy(dst.data(), p, dst.size()); ok = true; }
            b->Unlock();
        }
    }
    b->Release();
    return ok;
}

/* Read the next video frame into m->next. False at the end. */
static bool read_ahead(sail_movie* m)
{
    for (;;) {
        DWORD si = 0, fl = 0; LONGLONG ts = 0; IMFSample* s = nullptr;
        HRESULT hr = m->video->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0,
                                          &si, &fl, &ts, &s);
        if (FAILED(hr) || (fl & (MF_SOURCE_READERF_ENDOFSTREAM | MF_SOURCE_READERF_ERROR))) {
            if (s) s->Release();
            return false;
        }
        if (!s) continue;                  /* a stream tick or format change */
        bool ok = copy_frame(m, s, m->next);
        s->Release();
        if (!ok) return false;
        m->next_us = ts / 10;              /* 100 ns units */
        return true;
    }
}

extern "C" sail_movie* sail_movie_open(const char* host_path)
{
    if (!host_path || !mf_start()) return nullptr;
    wchar_t wpath[MAX_PATH * 2];
    if (!MultiByteToWideChar(CP_UTF8, 0, host_path, -1, wpath, (int)(sizeof(wpath) / sizeof(wpath[0]))))
        return nullptr;
    sail_movie* m = new sail_movie();

    m->video = open_reader(wpath, true);
    if (m->video) {
        IMFMediaType* cur = nullptr;
        UINT32 w = 0, h = 0;
        if (SUCCEEDED(m->video->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &cur))) {
            MFGetAttributeSize(cur, MF_MT_FRAME_SIZE, &w, &h);
            cur->Release();
        }
        m->w = (int)w & ~1; m->h = (int)h & ~1;
        if (!m->w || !m->h) { m->video->Release(); m->video = nullptr; m->w = m->h = 0; }
    }

    if (IMFSourceReader* a = open_reader(wpath, false)) {
        for (;;) {
            DWORD si = 0, fl = 0; LONGLONG ts = 0; IMFSample* s = nullptr;
            HRESULT hr = a->ReadSample((DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, &si, &fl, &ts, &s);
            if (s) {
                IMFMediaBuffer* b = nullptr;
                if (SUCCEEDED(s->ConvertToContiguousBuffer(&b))) {
                    BYTE* p = nullptr; DWORD len = 0;
                    if (SUCCEEDED(b->Lock(&p, nullptr, &len))) {
                        const float* f = (const float*)p;
                        m->audio.insert(m->audio.end(), f, f + len / sizeof(float));
                        b->Unlock();
                    }
                    b->Release();
                }
                s->Release();
            }
            if (FAILED(hr) || (fl & (MF_SOURCE_READERF_ENDOFSTREAM | MF_SOURCE_READERF_ERROR))) break;
        }
        a->Release();
    }

    if (!m->video && m->audio.empty()) { delete m; return nullptr; }

    /* Length: the source's own duration, or the audio's. */
    PROPVARIANT v; PropVariantInit(&v);
    if (m->video && SUCCEEDED(m->video->GetPresentationAttribute((DWORD)MF_SOURCE_READER_MEDIASOURCE,
                                                                  MF_PD_DURATION, &v)) && v.vt == VT_UI8)
        m->duration_us = (long long)(v.uhVal.QuadPart / 10);
    PropVariantClear(&v);
    const long long a_us = (long long)(m->audio.size() / 2) * 1000000LL / 48000;
    if (a_us > m->duration_us) m->duration_us = a_us;

    if (m->video && !read_ahead(m)) m->ended = true;
    fprintf(stderr, "[sail-movie] %s: %dx%d, %.2f s, audio %.2f s\n", host_path, m->w, m->h,
            m->duration_us / 1e6, a_us / 1e6);
    return m;
}

extern "C" void sail_movie_close(sail_movie* m)
{
    if (!m) return;
    if (m->video) m->video->Release();
    delete m;
}

extern "C" int sail_movie_width(const sail_movie* m)  { return m ? m->w : 0; }
extern "C" int sail_movie_height(const sail_movie* m) { return m ? m->h : 0; }
extern "C" long long sail_movie_duration_us(const sail_movie* m) { return m ? m->duration_us : 0; }

extern "C" int sail_movie_video_at(sail_movie* m, long long t_us, const unsigned char** nv12)
{
    if (!m || !m->video) return -1;
    bool advanced = false;
    while (m->next_us >= 0 && m->next_us <= t_us) {
        m->frame.swap(m->next);
        m->have_frame = true;
        advanced = true;
        m->next_us = -1;
        if (!read_ahead(m)) { m->ended = true; break; }
    }
    if (nv12) *nv12 = m->have_frame ? m->frame.data() : nullptr;
    if (advanced) return 1;
    if (m->ended && m->next_us < 0 && t_us >= m->duration_us) return -1;
    return 0;
}

extern "C" const float* sail_movie_audio(const sail_movie* m, unsigned* frames)
{
    if (frames) *frames = m ? (unsigned)(m->audio.size() / 2) : 0;
    return (m && !m->audio.empty()) ? m->audio.data() : nullptr;
}

#else   /* !_WIN32: no host decoder yet; cellSail plays an ended stream */

extern "C" sail_movie* sail_movie_open(const char*) { return 0; }
extern "C" void sail_movie_close(sail_movie*) {}
extern "C" int sail_movie_width(const sail_movie*) { return 0; }
extern "C" int sail_movie_height(const sail_movie*) { return 0; }
extern "C" long long sail_movie_duration_us(const sail_movie*) { return 0; }
extern "C" int sail_movie_video_at(sail_movie*, long long, const unsigned char**) { return -1; }
extern "C" const float* sail_movie_audio(const sail_movie*, unsigned* f) { if (f) *f = 0; return 0; }

#endif
