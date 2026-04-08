"""
Gemma-Falcon Vision Agent
Agentic pipeline combining Falcon Perception (segmentation/detection via MLX)
with Gemma 4 (vision reasoning via LM Studio) for grounded visual Q&A.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# ── Falcon Perception Setup ──────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent / "falcon-perception"))

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    load_and_prepare_model,
)
from falcon_perception.data import load_image
from falcon_perception.mlx.batch_inference import (
    BatchInferenceEngine,
    process_batch_and_generate,
)

# ── Constants ────────────────────────────────────────────────────────

PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
]

LM_STUDIO_BASE = "http://localhost:1234"
LM_STUDIO_URL = f"{LM_STUDIO_BASE}/v1/chat/completions"
MAX_AGENT_STEPS = 8
OUTPUT_DIR = Path("./outputs/agent")
VLM_MODEL_NAME = None  # populated at startup


# ── Falcon Perception Functions ──────────────────────────────────────


def load_falcon():
    """Load Falcon Perception model (MLX backend)."""
    print("[falcon] Loading model...")
    t0 = time.perf_counter()
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=PERCEPTION_MODEL_ID,
        dtype="float16",
        backend="mlx",
    )
    engine = BatchInferenceEngine(model, tokenizer)
    print(f"[falcon] Ready in {time.perf_counter() - t0:.1f}s")
    return engine, model, tokenizer, model_args


def pair_bbox_entries(raw: list[dict]) -> list[dict]:
    """Pair [{x,y}, {h,w}, ...] into [{x,y,h,w}, ...]."""
    bboxes, current = [], {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        current.update(entry)
        if all(k in current for k in ("x", "y", "h", "w")):
            bboxes.append(dict(current))
            current = {}
    return bboxes


def decode_rle_mask(rle: dict) -> np.ndarray | None:
    try:
        from pycocotools import mask as mask_utils
        return mask_utils.decode(rle).astype(np.uint8)
    except Exception:
        return None


def detect_objects(engine, model, tokenizer, model_args, image: Image.Image,
                   query: str, task: str = "segmentation") -> dict:
    """Run Falcon Perception on an image for a given query."""
    print(f"[falcon] Detecting '{query}' ({task})...")
    t0 = time.perf_counter()

    prompt = build_prompt_for_task(query, task)
    batch = process_batch_and_generate(
        tokenizer,
        [(image, prompt)],
        max_length=model_args.max_seq_len,
        min_dimension=256,
        max_dimension=1024,
    )

    output_tokens, aux_outputs = engine.generate(
        tokens=batch["tokens"],
        pos_t=batch["pos_t"],
        pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        max_new_tokens=200,
        temperature=0.0,
        task=task,
    )

    aux = aux_outputs[0]
    bboxes = pair_bbox_entries(aux.bboxes_raw)
    elapsed = time.perf_counter() - t0
    print(f"[falcon] Found {len(bboxes)} instances of '{query}' in {elapsed:.1f}s")

    return {
        "query": query,
        "count": len(bboxes),
        "bboxes": bboxes,
        "masks_rle": aux.masks_rle,
    }


def annotate_image(image: Image.Image, detections: list[dict],
                   interior_opacity: float = 0.35) -> Image.Image:
    """Draw bounding boxes and masks on image for all detection groups."""
    img = image.copy().convert("RGB")
    W, H = img.size
    overlay = np.array(img, dtype=np.float32)
    draw_items = []  # (label_text, bbox, color_idx)

    global_idx = 0
    for det_group in detections:
        query = det_group["query"]
        bboxes = det_group["bboxes"]
        masks_rle = det_group["masks_rle"]

        # Decode and overlay masks
        for i, rle in enumerate(masks_rle):
            if i >= len(bboxes):
                break
            m = decode_rle_mask(rle)
            if m is not None:
                if m.shape != (H, W):
                    m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
                color = np.array(PALETTE[global_idx % len(PALETTE)], dtype=np.float32)
                region = m > 0
                overlay[region] = overlay[region] * (1 - interior_opacity) + color * interior_opacity
            global_idx += 1

        # Collect bbox drawing info
        for i, bbox in enumerate(bboxes):
            color_idx = (global_idx - len(bboxes)) + i
            draw_items.append((f"{query} #{i+1}", bbox, color_idx))

    result = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)

    for label, bbox, cidx in draw_items:
        cx, cy = bbox["x"] * W, bbox["y"] * H
        bw, bh = bbox["w"] * W, bbox["h"] * H
        x0, y0 = cx - bw / 2, cy - bh / 2
        x1, y1 = cx + bw / 2, cy + bh / 2
        color = PALETTE[cidx % len(PALETTE)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), label, fill=color)

    return result


# ── Gemma 4 (LM Studio) Functions ───────────────────────────────────


def detect_vlm_model() -> str:
    """Query LM Studio for the currently loaded model name."""
    global VLM_MODEL_NAME
    try:
        resp = requests.get(f"{LM_STUDIO_BASE}/v1/models", timeout=5)
        models = resp.json().get("data", [])
        if models:
            VLM_MODEL_NAME = models[0]["id"]
        else:
            VLM_MODEL_NAME = "unknown"
    except Exception:
        VLM_MODEL_NAME = "unknown"
    return VLM_MODEL_NAME


def encode_image_b64(image: Image.Image) -> str:
    """Convert PIL Image to base64 JPEG string."""
    from io import BytesIO
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_gemma_response(msg: dict) -> str:
    """Extract response from Gemma, handling thinking models.

    Thinking models put reasoning in 'reasoning_content' and the final
    answer in 'content'. If content is empty, the model ran out of tokens
    while still thinking — fall back to reasoning_content.
    """
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    if content:
        return content
    return reasoning


def gemma_text(prompt: str, max_tokens: int = 2000) -> str:
    """Send a text-only query to Gemma 4 via LM Studio."""
    resp = requests.post(LM_STUDIO_URL, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }, timeout=180)
    return _extract_gemma_response(resp.json()["choices"][0]["message"])


def gemma_vision(prompt: str, image: Image.Image, max_tokens: int = 2000) -> str:
    """Send an image + text query to Gemma 4 via LM Studio."""
    img_b64 = encode_image_b64(image)
    resp = requests.post(LM_STUDIO_URL, json={
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }, timeout=180)
    return _extract_gemma_response(resp.json()["choices"][0]["message"])


# ── Agent Logic ──────────────────────────────────────────────────────


def _parse_json_list(text: str) -> list[str] | None:
    """Try to find and parse a JSON list of strings from text."""
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, list) and all(isinstance(x, str) for x in result):
                return result
        except json.JSONDecodeError:
            pass
    return None


def identify_objects_in_image(query: str, image: Image.Image) -> list[str]:
    """Use Gemma vision to look at the image and identify specific object types to detect."""
    prompt = f"""Look at this image. A user asked: "{query}"

To answer this question, I need to run an object detection model to find and count specific object types.

What specific object types should I detect? Keep all qualifiers like colors, sizes, etc.
Examples:
- "Are there more red cars than blue cars?" -> ["red cars", "blue cars"]
- "How many large dogs?" -> ["large dogs"]
- "What fruits are here?" -> ["apples", "oranges", "limes"] (be specific, not generic like "fruits")

Return ONLY a JSON list of strings, nothing else.
JSON list:"""

    img_b64 = encode_image_b64(image)
    resp = requests.post(LM_STUDIO_URL, json={
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        "max_tokens": 2000,
        "temperature": 0.1,
    }, timeout=180)
    msg = resp.json()["choices"][0]["message"]

    # Try content first, then reasoning_content — either may contain the JSON list
    for field in ["content", "reasoning_content"]:
        text = msg.get(field, "")
        if text:
            result = _parse_json_list(text)
            if result:
                return result

    # Fallback: just return the query as a single item
    print(f"[agent] Warning: Could not parse object types from vision response, using query directly")
    return [query]


def run_agent(image_path: str, query: str):
    """Run the full agentic pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Gemma-Falcon Vision Agent")
    print(f"{'='*60}")
    print(f"  Image: {image_path}")
    print(f"  Query: {query}")

    # Detect VLM model
    vlm_name = detect_vlm_model()
    print(f"  VLM  : {vlm_name}")
    print(f"{'='*60}\n")

    # Load image
    image = Image.open(image_path).convert("RGB")
    image.save(OUTPUT_DIR / "input.jpg")

    # Load Falcon
    engine, model, tokenizer, model_args = load_falcon()

    # ── Phase 1: Plan ────────────────────────────────────────────────
    print(f"\n--- Phase 1: Identify objects to detect [{vlm_name}] ---")
    t0 = time.perf_counter()
    object_types = identify_objects_in_image(query, image)
    print(f"[phase1] Objects to detect: {object_types} ({time.perf_counter()-t0:.1f}s)")

    # ── Phase 2: Detect + Segment ────────────────────────────────────
    print(f"\n--- Phase 2: Detect + Segment [Falcon Perception] ---")
    all_detections = []
    for obj_type in object_types:
        result = detect_objects(engine, model, tokenizer, model_args,
                                image, obj_type, task="segmentation")
        all_detections.append(result)

    # Build summary of what was found
    detection_summary = []
    for det in all_detections:
        detection_summary.append(f"- {det['query']}: {det['count']} instances detected")
    summary_text = "\n".join(detection_summary)
    print(f"\n[phase2] Detection summary:\n{summary_text}")

    # ── Phase 3: Annotate ────────────────────────────────────────────
    print(f"\n--- Phase 3: Annotate image [PIL] ---")
    annotated = annotate_image(image, all_detections)
    annotated_path = OUTPUT_DIR / "annotated.jpg"
    annotated.save(annotated_path)
    print(f"[phase3] Saved annotated image: {annotated_path}")

    # ── Phase 4: Visual Reasoning ────────────────────────────────────
    print(f"\n--- Phase 4: Visual reasoning [{vlm_name}] ---")
    reasoning_prompt = f"""Object detection results for this image:

{summary_text}

Question: {query}

Answer concisely using the detection counts above. Do not describe your reasoning process."""

    t0 = time.perf_counter()
    answer = gemma_vision(reasoning_prompt, annotated)
    print(f"[phase4] Gemma reasoning complete ({time.perf_counter()-t0:.1f}s)")

    # ── Final Answer ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL ANSWER")
    print(f"{'='*60}")
    print(f"\n{answer}\n")
    print(f"{'='*60}")
    print(f"  Output files: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")

    return answer


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma-Falcon Vision Agent")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--query", type=str, required=True, help="Question about the image")
    parser.add_argument("--output", type=str, default="./outputs/agent", help="Output directory")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output)
    run_agent(args.image, args.query)
