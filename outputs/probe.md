# raw probe

## ffprobe
```json
{
    "streams": [
        {
            "index": 0,
            "codec_name": "h264",
            "codec_long_name": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
            "profile": "High",
            "codec_type": "video",
            "codec_tag_string": "avc1",
            "codec_tag": "0x31637661",
            "width": 1080,
            "height": 1920,
            "coded_width": 1080,
            "coded_height": 1920,
            "has_b_frames": 2,
            "sample_aspect_ratio": "1:1",
            "display_aspect_ratio": "9:16",
            "pix_fmt": "yuv420p",
            "level": 40,
            "color_range": "tv",
            "color_transfer": "iec61966-2-1",
            "color_primaries": "bt709",
            "chroma_location": "left",
            "field_order": "progressive",
            "refs": 1,
            "is_avc": "true",
            "nal_length_size": "4",
            "id": "0x1",
            "r_frame_rate": "30/1",
            "avg_frame_rate": "30/1",
            "time_base": "1/90000",
            "start_pts": 0,
            "start_time": "0.000000",
            "duration_ts": 1350000,
            "duration": "15.000000",
            "bit_rate": "2333498",
            "bits_per_raw_sample": "8",
            "nb_frames": "450",
            "extradata_size": 51,
            "disposition": {
                "default": 1,
                "dub": 0,
                "original": 0,
                "comment": 0,
                "lyrics": 0,
                "karaoke": 0,
                "forced": 0,
                "hearing_impaired": 0,
                "visual_impaired": 0,
                "clean_effects": 0,
                "attached_pic": 0,
                "timed_thumbnails": 0,
                "non_diegetic": 0,
                "captions": 0,
                "descriptions": 0,
                "metadata": 0,
                "dependent": 0,
                "still_image": 0,
                "multilayer": 0
            },
            "tags": {
                "language": "und",
                "handler_name": "VideoHandler",
                "vendor_id": "[0][0][0][0]"
            }
        },
        {
            "index": 1,
            "codec_name": "aac",
            "codec_long_name": "AAC (Advanced Audio Coding)",
            "profile": "LC",
            "codec_type": "audio",
            "codec_tag_string": "mp4a",
            "codec_tag": "0x6134706d",
            "sample_fmt": "fltp",
            "sample_rate": "48000",
            "channels": 2,
            "channel_layout": "stereo",
            "bits_per_sample": 0,
            "initial_padding": 0,
            "id": "0x2",
            "r_frame_rate": "0/0",
            "avg_frame_rate": "0/0",
            "time_base": "1/48000",
            "start_pts": 0,
            "start_time": "0.000000",
            "duration_ts": 720912,
            "duration": "15.019000",
            "bit_rate": "317367",
            "nb_frames": "706",
            "extradata_size": 2,
            "disposition": {
                "default": 1,
                "dub": 0,
                "original": 0,
                "comment": 0,
                "lyrics": 0,
                "karaoke": 0,
                "forced": 0,
                "hearing_impaired": 0,
                "visual_impaired": 0,
                "clean_effects": 0,
                "attached_pic": 0,
                "timed_thumbnails": 0,
                "non_diegetic": 0,
                "captions": 0,
                "descriptions": 0,
                "metadata": 0,
                "dependent": 0,
                "still_image": 0,
                "multilayer": 0
            },
            "tags": {
                "language": "und",
                "handler_name": "SoundHandler",
                "vendor_id": "[0][0][0][0]"
            }
        }
    ],
    "format": {
        "filename": "outputs/demo_reel.mp4",
        "nb_streams": 2,
        "nb_programs": 0,
        "nb_stream_groups": 0,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "format_long_name": "QuickTime / MOV",
        "start_time": "0.000000",
        "duration": "15.019000",
        "size": "4990440",
        "bit_rate": "2658200",
        "probe_score": 100,
        "tags": {
            "major_brand": "isom",
            "minor_version": "512",
            "compatible_brands": "isomiso2avc1mp41",
            "encoder": "Lavf61.7.100",
            "comment": "Made with Remotion 4.0.547"
        }
    }
}
```

## volumedetect
```
      handler_name    : SoundHandler
      vendor_id       : [0][0][0][0]
[Parsed_volumedetect_0 @ 0x73d418002000] n_samples: 1441792
[Parsed_volumedetect_0 @ 0x73d418002000] mean_volume: -24.4 dB
[Parsed_volumedetect_0 @ 0x73d418002000] max_volume: -6.0 dB
[Parsed_volumedetect_0 @ 0x73d418002000] histogram_5db: 2
[Parsed_volumedetect_0 @ 0x73d418002000] histogram_6db: 58
[Parsed_volumedetect_0 @ 0x73d418002000] histogram_7db: 226
[Parsed_volumedetect_0 @ 0x73d418002000] histogram_8db: 634
[Parsed_volumedetect_0 @ 0x73d418002000] histogram_9db: 1400
[out#0/null @ 0x57f5d8596f00] video:0KiB audio:2816KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: unknown
size=N/A time=00:00:15.01 bitrate=N/A speed=1.29e+03x elapsed=0:00:00.01    
```

## size
-rw-r--r-- 1 runner runner 4990440 Nov 28 16:24 outputs/demo_reel.mp4
