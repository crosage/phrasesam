"""Image-pool phrase screening and correction threshold calibration."""
import numpy as np
import torch

from phrasesam import class_scores


def grid_responses(logits, presence):
    grid = logits[:, ::3, ::3].astype(np.float32)
    return presence.astype(np.float32)[:, None, None] / (1 + np.exp(-np.clip(grid, -25, 25)))


def confident_classes(scores):
    top, labels = torch.as_tensor(scores).topk(2, dim=0)
    return torch.where((top[0] >= 0.5) & (top[0] >= 1.5 * top[1]), labels[0], -1)


def screen(rows, anchors, class_names, background, counts, sums, unique):
    selected = []
    for c, row in enumerate(rows):
        initial = anchors[c]
        rivals = [d for d in range(len(rows)) if d != c and counts[d] >= 32]
        if c == background:
            selected.append(row)
            continue
        if counts[c] < 32 or not rivals:
            selected.append([initial])
            continue
        own = sums[:, c] / counts[c]
        rival = (sums[:, rivals] / counts[rivals]).max(1)
        admitted = [w for w in row if w != initial and own[w] >= 0.2
                    and own[w] >= 1.5 * rival[w] and unique[w, c] > 0]
        replacements = [w for w in admitted if (own[w] - rival[w]) / own[w] >= 0.8]
        if rival[initial] > 0.4 and replacements:
            admitted = replacements
            if initial != class_names[c]:
                selected.append(admitted)
                continue
        selected.append([initial, *admitted])
    return selected


def configure(vocabulary, queries, observations, background_floor, device):
    index = {query: i for i, query in enumerate(queries)}
    query_rows = vocabulary["candidate_rows"]
    rows = [[index[q] for q in row] for row in query_rows]
    anchors = [index[q] for q in vocabulary["initial_queries"]]
    class_name_indices = [index.get(q.casefold(), -1) for q in vocabulary["class_names"]]
    for c, query in enumerate(vocabulary["initial_queries"]):
        if query.casefold() == vocabulary["class_names"][c].casefold():
            class_name_indices[c] = anchors[c]
    background = vocabulary["background"]
    k = len(rows)
    counts = np.zeros(k, dtype=np.int64)
    sums = np.zeros((len(queries), k), dtype=np.float64)
    unique = np.zeros((len(queries), k), dtype=np.int64)

    for logits, presence, _ in observations():
        grid = grid_responses(logits, presence)
        cores = confident_classes(grid[anchors]).numpy()
        winners = np.stack([grid[row].max(0) for row in rows]).argmax(0)
        for c, row in enumerate(rows):
            mask = cores == c
            counts[c] += mask.sum()
            sums[:, c] += grid[:, mask].sum(1, dtype=np.float64)
            responses = grid[row][:, winners == c]
            second = np.partition(responses, -2, axis=0)[-2] if len(row) > 1 else 0
            unique[row, c] += (responses > second + 1e-4).sum(1)

    selected = screen(rows, anchors, class_name_indices, background, counts, sums, unique)
    selected_queries = [[queries[w] for w in row] for row in selected]
    eligible = [c for c in range(k) if c != background and counts[c] >= 32]
    overlap = np.zeros((k, k), dtype=np.float64)
    proposals = []
    if len(eligible) >= 2:
        for logits, presence, _ in observations():
            grid = grid_responses(logits, presence)
            cores = confident_classes(grid[anchors]).numpy()
            fields = np.stack([grid[row].max(0) for row in selected])
            for c in eligible:
                overlap[:, c] += fields[:, cores == c].sum(1, dtype=np.float64)
        overlap[:, eligible] /= counts[eligible]
        proposals = [(a, max((b for b in eligible if b != a),
                             key=lambda b: min(overlap[a, b], overlap[b, a])))
                     for a in eligible]

    directions = []
    if proposals:
        medians = {edge: ([], []) for edge in proposals}
        for logits, presence, size in observations():
            scores = class_scores(logits, presence, queries, selected_queries, size, device)
            sampled = scores[:, ::8, ::8].contiguous()
            del scores
            labels = confident_classes(sampled)
            logs = sampled.clamp_min(1e-4).log()
            for a, b in proposals:
                margin = logs[b] - logs[a]
                for group, c in zip(medians[a, b], (a, b)):
                    values = margin[labels == c]
                    if values.numel():
                        group.append(values.median().item())
        for (a, b), (source, target) in medians.items():
            if source and target:
                threshold = float((np.median(source) + np.median(target)) / 2)
                if threshold < 0:
                    directions.append(dict(source=a, target=b, threshold=threshold))

    return dict(selected_rows=selected_queries, background_floor=background_floor,
                directions=directions)
