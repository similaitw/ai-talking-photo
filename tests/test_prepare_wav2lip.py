from scripts.prepare_wav2lip import SOFT_BLEND_MARKER, patch_inference


def _upstream_sample() -> str:
    return """parser.add_argument('--nosmooth', default=False, action='store_true')
args = parser.parse_args()
if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    pass
			f[y1:y2, x1:x2] = p
	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')
"""


def test_patch_inference_adds_quality_and_compatibility_changes() -> None:
    patched = patch_inference(_upstream_sample())
    assert "--soft_blend" in patched
    assert SOFT_BLEND_MARKER in patched
    assert "cv2.ellipse" in patched
    assert "os.path.splitext(args.face)" in patched
    assert "subprocess.run(command, check=True)" in patched


def test_patch_inference_is_idempotent() -> None:
    once = patch_inference(_upstream_sample())
    twice = patch_inference(once)
    assert twice == once
    assert twice.count("--soft_blend") == 1
    assert twice.count(SOFT_BLEND_MARKER) == 1
