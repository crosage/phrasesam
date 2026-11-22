"""Configure PhraseSAM from RGB images or predict with saved phrase sets."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import MethodType

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def readout(fields, background, profile):
    scores = fields.clone()
    scores[background].clamp_min_(profile["background_floor"])
    initial = scores.argmax(0)
    scores[background] = fields[background]
    scores.scatter_(0, initial[None], -torch.inf)
    runner_up = scores.argmax(0)
    logs = fields.clamp_min(1e-4).log()
    final = initial.clone()
    for edge in profile["directions"]:
        source, target = edge["source"], edge["target"]
        change = (initial == source) & (runner_up == target)
        change &= logs[target] - logs[source] > edge["threshold"]
        final[change] = target
    return final


def class_scores(logits, presence, queries, rows, size, device):
    """Restore native logits before sigmoid and concept-presence weighting."""
    index = {query: i for i, query in enumerate(queries)}
    native = torch.as_tensor(logits, device=device, dtype=torch.float16)
    weights = torch.as_tensor(presence, device=device, dtype=torch.float16)
    fields = []
    for row in rows:
        field = None
        for query in row:
            i = index[query]
            score = F.interpolate(native[i][None, None], size=size,
                                  mode="bilinear", align_corners=False)[0, 0]
            score = score.sigmoid() * weights[i]
            field = score if field is None else torch.maximum(field, score)
        fields.append(field.float())
    return torch.stack(fields)


class SAM3:
    def __init__(self, checkpoint, sam3_root, device):
        if sam3_root:
            sys.path.insert(0, str(sam3_root.resolve()))
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        tokenizer = Path(sam3.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(
            bpe_path=str(tokenizer), checkpoint_path=str(checkpoint),
            device=device, load_from_HF=False, compile=False,
        )
        self.processor = Sam3Processor(model.eval(), resolution=1008, device=device)

        def raw_forward(processor, state):
            output = processor.model.forward_grounding(
                backbone_out=state["backbone_out"], find_input=processor.find_stage,
                geometric_prompt=state["geometric_prompt"], find_target=None,
            )
            state["native_logits"] = output["semantic_seg"].squeeze().float().cpu().numpy()
            state["presence"] = output["presence_logit_dec"].sigmoid().item()
            return state

        self.processor._forward_grounding = MethodType(raw_forward, self.processor)

    @torch.inference_mode()
    def observe(self, image, queries):
        logits, presence = [], []
        with torch.autocast("cuda", dtype=torch.float16):
            state = self.processor.set_image(image)
            for query in queries:
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(prompt=query, state=state)
                logits.append(state["native_logits"])
                presence.append(state["presence"])
        return np.asarray(logits, dtype=np.float16), np.asarray(presence, dtype=np.float16)


def unique_queries(rows):
    return list(dict.fromkeys(query for row in rows for query in row))


def read_images(source):
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    images = [source] if source.is_file() else sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )
    if not images:
        raise ValueError(f"No images found in {source}")
    return images


@torch.inference_mode()
def main():
    inventories = json.loads(Path(__file__).with_name("vocabularies.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["configure", "predict"])
    parser.add_argument("--dataset", choices=sorted(inventories), required=True)
    parser.add_argument("--images", type=Path, required=True, help="RGB image or flat directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam3-root", type=Path, help="Path to the SegEarth-OV3 checkout")
    parser.add_argument("--profile", choices=["main", "separate"], default="main")
    parser.add_argument("--configuration", type=Path, help="Custom profile JSON for predict")
    parser.add_argument("--background-floor", type=float, help="Override the floor during configure")
    parser.add_argument("--cache-dir", type=Path, help="Parent directory for temporary configuration observations")
    args = parser.parse_args()

    vocabulary = inventories[args.dataset]
    profile = vocabulary["profiles"][args.profile]
    images = read_images(args.images)
    if args.configuration is not None:
        profile = json.loads(args.configuration.read_text())
        if args.stage != "predict" or profile.get("dataset") != args.dataset:
            parser.error("--configuration is for prediction on its recorded dataset")
    if args.background_floor is not None and (
            args.stage != "configure" or not 0 <= args.background_floor <= 1):
        parser.error("--background-floor requires configure and a value in [0, 1]")
    if not args.checkpoint.is_file():
        parser.error("The SAM 3 checkpoint file does not exist")
    device = "cuda"
    model = SAM3(args.checkpoint, args.sam3_root, device)

    if args.stage == "configure":
        from configure import configure

        rows = vocabulary["candidate_rows"]
        queries = unique_queries(rows)
        with TemporaryDirectory(prefix="phrasesam-", dir=args.cache_dir) as folder:
            for i, path in enumerate(images):
                with Image.open(path) as image:
                    logits, presence = model.observe(image.convert("RGB"), queries)
                    np.savez(Path(folder) / f"{i}.npz", logits=logits, presence=presence,
                             size=(image.height, image.width))
                print(f"Observed {i + 1}/{len(images)}: {path.name}", flush=True)

            def observations():
                for i in range(len(images)):
                    with np.load(Path(folder) / f"{i}.npz", allow_pickle=False) as bank:
                        yield bank["logits"], bank["presence"], tuple(bank["size"])

            floor = profile["background_floor"] if args.background_floor is None else args.background_floor
            fitted = configure(vocabulary, queries, observations, floor, device)
        fitted["dataset"] = args.dataset
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(fitted, indent=2) + "\n")
        return

    names = [f"{path.stem}.png" for path in images]
    if len(set(names)) != len(names):
        parser.error("Input image stems must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = profile["selected_rows"]
    queries = unique_queries(rows)
    for path, output_name in zip(images, names):
        with Image.open(path) as image:
            logits, presence = model.observe(image.convert("RGB"), queries)
            size = image.height, image.width
        fields = class_scores(logits, presence, queries, rows, size, device)
        labels = readout(fields, vocabulary["background"], profile)
        Image.fromarray(labels.byte().cpu().numpy()).save(args.output / output_name)
        print(output_name, flush=True)


if __name__ == "__main__":
    main()
