from paddlelite.lite import *
import numpy as np
from PIL import Image

def load_predictor(model_path):
    cfg = MobileConfig()
    cfg.set_model_from_file(model_path)
    return create_paddle_predictor(cfg)

def preprocess(im, size):
    im = im.convert("RGB").resize(size)
    arr = np.array(im).astype("float32") / 255.0
    arr = arr.transpose(2,0,1)[None,:]   # NCHW
    return arr

def load_keys(path_txt):
    with open(path_txt, "r", encoding="utf-8") as f:
        return [line.strip("\n\r") for line in f if line.strip("\n\r")]

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def ctc_greedy_decode(rec_out, keys):
    """
    rec_out: (T, C) 或 (1, T, C) 的 numpy array（logits）
    keys:    list[str], 由 ppocr_keys_v1.txt 載入（無 'blank'）
    return: (text, avg_conf)
    """
    # squeeze 到 (T, C)
    if rec_out.ndim == 3:
        rec_out = rec_out[0]
    T, C = rec_out.shape

    # logits -> prob
    prob = softmax(rec_out, axis=1)          # (T, C)
    ids  = prob.argmax(axis=1)               # (T,)
    conf = prob.max(axis=1)                  # (T,)

    blank_idx = 0  # PaddleOCR CTC 的 blank 通常是 0
    text_chars, kept_conf = [], []

    prev = -1
    for t in range(T):
        idx = ids[t]
        # CTC 規則：去重、跳過 blank
        if idx == blank_idx or idx == prev:
            prev = idx
            continue
        # 字元索引對應到 keys（因為 0 是 blank，所以實際字元是 idx-1）
        if 1 <= idx <= len(keys):
            text_chars.append(keys[idx - 1])
            kept_conf.append(float(conf[t]))
        prev = idx

    text = "".join(text_chars)
    avg_conf = float(np.mean(kept_conf)) if kept_conf else 0.0
    return text, avg_conf

det = load_predictor("./Lite_model_det.nb")
rec = load_predictor("./Lite_model_rec.nb")


im = Image.open("test.jpg")

det_inp = rec.get_input(0)
det_inp.from_numpy(preprocess(im, (640,640)))  # 常用640
det.run()
det_out = det.get_output(0).numpy()
print("det out shape:", det_out.shape)

rec_inp = rec.get_input(0)
rec_inp.from_numpy(preprocess(im, (320,32)))   # rec 常用 (W,H)=(320,32)
rec.run()
rec_out = rec.get_output(0).numpy()
print("rec out shape:", rec_out.shape)

keys = load_keys("ppocr_keys_v1.txt")
text, score = ctc_greedy_decode(rec_out, keys)
print("REC:", text, "  conf:", f"{score:.3f}")


#b = np.array(rec_out, dtype='u4')

#for index in b[0][0]:
	#print(b)
