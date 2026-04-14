import argparse
import glob
import os

import cv2
import numpy as np

from edge_detection import (
	detect_BDCN_edge,
	detect_SE_edge,
	detect_hed_edge,
	get_BDCN_model,
	get_SE_model,
	get_hed_model,
)


def collect_images(input_dir: str):
	patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.WEBP"]
	files = []
	for pat in patterns:
		files.extend(glob.glob(os.path.join(input_dir, pat)))
	return sorted(files)


def to_uint8(x: np.ndarray) -> np.ndarray:
	return np.clip(x * 255.0, 0, 255).astype(np.uint8)

"""
バッチ線画生成スクリプト。

指定ディレクトリ内の画像に対して 3 種類のエッジ検出（BDCN / HED / SE）を実行し、
選択したモードで合成して PNG として保存する。

Args:
    --input_dir (str): 入力画像ディレクトリ。
    --output_dir (str): 出力先ディレクトリ。
    --device (str): 推論デバイス（"cpu" または "cuda"）。
    --mode (str): 出力モード。
        - "bdcn": BDCN の結果のみ
        - "hed": HED の結果のみ
        - "se": SE の結果のみ
        - "mean": 3 手法の平均
        - "max": 3 手法の各画素の最大値（最も強い反応）を採用
        - "stack": 3ch 画像として [BDCN, HED, SE] をそのまま保存
    --invert: 指定時、出力を反転（白背景＋黒線向け）。

Notes:
    「max」は、同じ画素位置で BDCN/HED/SE の値を比較し、
    その中で最も大きい値を採用する合成方法。
"""
if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=True)
	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
	WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

	_INPUT_CANDIDATES = [
		os.path.join(WORKSPACE_ROOT, "zikken", "photos"),
		os.path.join(PROJECT_ROOT, "zikken", "photos"),
	]
	DEFAULT_INPUT_DIR = next((p for p in _INPUT_CANDIDATES if os.path.isdir(p)), _INPUT_CANDIDATES[0])

	_OUTPUT_CANDIDATES = [
		os.path.join(WORKSPACE_ROOT, "zikken", "output"),
		os.path.join(PROJECT_ROOT, "zikken", "output"),
	]
	DEFAULT_OUTPUT_DIR = next((p for p in _OUTPUT_CANDIDATES if os.path.isdir(os.path.dirname(p))), _OUTPUT_CANDIDATES[0])
	
	parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
	parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
	parser.add_argument(
		"--mode",
		type=str,
		default="mean",
		choices=["mean", "max", "stack", "bdcn", "hed", "se"],
		help="線画の作り方。mean/max は3手法を合成、stack は3ch(BDCN,HED,SE)をそのまま保存",
	)
	parser.add_argument(
		"--invert",
		action="store_true",
		help="白背景＋黒線で保存したい場合に指定",
	)
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	image_files = collect_images(args.input_dir)

	if len(image_files) == 0:
		raise RuntimeError(f"No images found in: {args.input_dir}")

	se_model = get_SE_model()

	bdcn_model = get_BDCN_model()
	bdcn_model.to(args.device).eval()

	hed_model = get_hed_model()
	hed_model.to(args.device).eval()

	for image_path in image_files:
		image_bgr = cv2.imread(image_path)
		if image_bgr is None:
			print(f"[WARN] skip unreadable image: {image_path}")
			continue

		se = detect_SE_edge(se_model, image_bgr)
		bdcn = detect_BDCN_edge(bdcn_model, image_bgr, args.device)
		hed = detect_hed_edge(hed_model, image_bgr, args.device)

		if args.mode == "stack":
			out = np.stack([bdcn, hed, se], axis=-1)
			out = to_uint8(out)
		elif args.mode == "bdcn":
			out = to_uint8(bdcn)
		elif args.mode == "hed":
			out = to_uint8(hed)
		elif args.mode == "se":
			out = to_uint8(se)
		elif args.mode == "max":
			out = to_uint8(np.maximum(np.maximum(bdcn, hed), se))
		else:
			out = to_uint8((bdcn + hed + se) / 3.0)

		if args.invert:
			out = 255 - out

		stem = os.path.splitext(os.path.basename(image_path))[0]
		output_path = os.path.join(args.output_dir, f"{stem}_edge.png")
		cv2.imwrite(output_path, out)
		print(f"[OK] {image_path} -> {output_path}")
		

