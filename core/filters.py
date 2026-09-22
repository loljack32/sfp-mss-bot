501
502
503
504
505
506
507
508
509
510
511
512
513
514
515
516
517
518
519
520
521
522
523
524
525
526
527
528
529
530
531
532
533
534
535
536
537
538
539
540
541
542
543
544
545
546
547
548
549
550
551
552
553
# ============================================================
    htf_component = score_htf(setup_type)
    liquidity_component = score_liquidity(target)
    sfp_component = score_sfp(sfp)
    mss_component = score_mss(mss)
    displacement_component = score_displacement(mss)
    volume_component = score_volume(volume_ratio)
    rr_component = score_rr(rr)

    final_score = calculate_signal_score(
        htf_score=htf_component,
        liquidity_score=liquidity_component,
        sfp_score=sfp_component,
        mss_score=mss_component,
        displacement_score=displacement_component,
        volume_score=volume_component,
        rr_score=rr_component,
    )

    metrics["final_score"] = final_score

    if final_score < MIN_SIGNAL_SCORE:
        reasons.append(f"Signal score {final_score:.1f} is below minimum {MIN_SIGNAL_SCORE:.1f}")

    passed = len(reasons) == 0

    return FilterResult(
        passed=passed,
        score=final_score,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
        setup_type=setup_type,
    )


# ============================================================
# SERIALIZATION
# ============================================================

def filter_result_to_dict(
    result: FilterResult,
) -> dict:
    """
    Преобразует результат фильтрации в dict.
    """
    return {
        "passed": result.passed,
        "score": result.score,
        "reasons": result.reasons,
        "warnings": result.warnings,
        "metrics": result.metrics,
        "setup_type": getattr(result, "setup_type", "TREND"),
    }
