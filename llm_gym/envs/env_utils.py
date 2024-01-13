from typing import Dict, List


def run_python_code_unsafe(code: str) -> str:
    prints = []

    def _print(*args):
        prints.append(args)

    try:
        exec(code, {"print": _print})
    except Exception as e:
        return str(e)
    else:
        return "\n".join(" ".join(str(x) for x in p) for p in prints)


def aggregate_stats(stats: List[Dict]) -> Dict:
    if len(stats) == 0:
        return {}
    numeric_keys = [
        k for k in stats[0].keys() if isinstance(stats[0][k], (int, float, bool))
    ]
    non_numeric_keys = [k for k in stats[0].keys() if k not in numeric_keys]
    agg_stats = {}
    for k in numeric_keys:
        agg_stats[k + "_mean"] = sum([s[k] for s in stats]) / len(stats)
        agg_stats[k + "_sum"] = sum([s[k] for s in stats])
        agg_stats[k + "_min"] = min([s[k] for s in stats])
        agg_stats[k + "_max"] = max([s[k] for s in stats])
    for k in non_numeric_keys:
        agg_stats[k + "_nunique"] = len(set([s[k] for s in stats]))
        agg_stats[k + "_examples"] = [repr(s[k]) for s in stats][:10]
    return agg_stats
