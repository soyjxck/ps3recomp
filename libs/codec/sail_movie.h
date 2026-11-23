/*
 * ps3recomp - host movie decoding for cellSail
 *
 * cellSail hands a title decoded frames and PCM through its graphics and sound
 * adapters; the container and codecs are the host's business. On Windows this
 * is Media Foundation, which ships with the OS and opens what PS3 titles put
 * in AVI and MP4 (MPEG-4 Part 2 / DivX, H.264, MP3, AAC, PCM), so nothing has
 * to be downloaded. Elsewhere (and on an "N" edition of Windows without the
 * Media Feature Pack) sail_movie_open returns NULL and cellSail falls back to
 * a stream that has already ended.
 *
 * Video comes out as NV12 (Y plane, then interleaved U/V at half size); audio
 * as interleaved float stereo at 48 kHz, decoded in full at open (a logo or a
 * short movie is a few MB).
 */
#ifndef PS3RECOMP_SAIL_MOVIE_H
#define PS3RECOMP_SAIL_MOVIE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sail_movie sail_movie;

/* Open a host file. NULL if the host cannot decode it. */
sail_movie* sail_movie_open(const char* host_path);
void        sail_movie_close(sail_movie* m);

/* Frame size; 0 when the file has no video. */
int  sail_movie_width(const sail_movie* m);
int  sail_movie_height(const sail_movie* m);
/* Length in microseconds (the longer of the two streams). */
long long sail_movie_duration_us(const sail_movie* m);

/* Advance the video to the last frame whose time is <= t_us. Returns 1 when
 * that is a different frame from the one returned before (*nv12 points at it,
 * width x height Y bytes then width x height/2 U/V bytes, tightly packed),
 * 0 when the current frame is still the right one, -1 once the stream has
 * ended or failed. */
int  sail_movie_video_at(sail_movie* m, long long t_us, const unsigned char** nv12);

/* The whole audio track: interleaved float stereo, 48 kHz. NULL/0 if none. */
const float* sail_movie_audio(const sail_movie* m, unsigned* frames);

#ifdef __cplusplus
}
#endif

#endif
