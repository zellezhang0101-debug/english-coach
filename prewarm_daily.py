import os
import sys
import time


def _best_provider(prefer: str | None) -> str:
    # Import lazily so this script can print a clear error if deps are missing.
    import app as appmod

    p = (prefer or "").strip().lower()
    if p not in ("gemini", "deepseek"):
        p = "gemini"
    if p == "gemini" and not appmod.GEMINI_API_KEY and appmod.DEEPSEEK_API_KEY:
        p = "deepseek"
    if p == "deepseek" and not appmod.DEEPSEEK_API_KEY and appmod.GEMINI_API_KEY:
        p = "gemini"
    return p


def main() -> int:
    import app as appmod

    prefer = os.getenv("PREWARM_PROVIDER")
    provider = _best_provider(prefer)
    diffs = (os.getenv("PREWARM_DIFFICULTIES") or "B1,B2,C1").split(",")
    diffs = [d.strip().upper() for d in diffs if d.strip()]
    if not diffs:
        diffs = ["B1", "B2", "C1"]

    print(f"[prewarm] DATA_DIR={appmod.DATA_DIR}")
    print(f"[prewarm] provider={provider} diffs={diffs}")

    if provider == "gemini" and not appmod.GEMINI_API_KEY:
        print("[prewarm] GEMINI_API_KEY not set; cannot prewarm with gemini.")
        return 2
    if provider == "deepseek" and not appmod.DEEPSEEK_API_KEY:
        print("[prewarm] DEEPSEEK_API_KEY not set; cannot prewarm with deepseek.")
        return 2

    t0 = time.time()
    try:
        print("[prewarm] expanding today pool…")
        pool = appmod._crawl_daily_pool(provider)
        print(f"[prewarm] today pool size={len(pool)}")
    except Exception as e:
        print(f"[prewarm] pool crawl failed: {e}")

    ok = 0
    for d in diffs:
        try:
            print(f"[prewarm] ensure daily practice article {d}…")
            a = appmod.get_daily_practice_article(d, provider, topic_for_fallback="")
            title = (a or {}).get("title") or ""
            print(f"[prewarm] {d} done title={title[:80]!r}")
            ok += 1
        except Exception as e:
            print(f"[prewarm] {d} failed: {e}")

    dt = time.time() - t0
    print(f"[prewarm] finished ok={ok}/{len(diffs)} in {dt:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

