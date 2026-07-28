"""
variants.py — the incremental study design.

Each variant changes **exactly one thing** relative to V0 (the published
method), so that any accuracy difference is attributable. V-ALL turns
everything on.

    V0    published behaviour: topology scored by accuracy on the ~30-trial
          validation split; stacking meta-learner fitted and scored in-sample
          on that same split; CSP without shrinkage; CSP/CSSP/log-var pool.

    V1    item 1 - out-of-fold objective. The topology is scored on
          out-of-fold predictions computed inside the training split (the
          whole feature pipeline is refitted per inner fold), and stacking
          nodes are cross-validated rather than scored in-sample.

    V2    item 2 - regularised selection: a complexity penalty plus the
          one-standard-error rule (return the simplest topology within one SE
          of the best).

    V3    item 3 - top-k averaging: return the weighted average of the k best
          topologies visited instead of the single argmax.

    V4    item 4 - enriched pool: per-band Riemannian tangent-space views and
          filter-bank CSP views become selectable members, so the search can
          reach the accuracy level of the strongest baselines.

    V5    item 5 - diversity-regularised objective: reward mean pairwise
          disagreement between committee members alongside accuracy.

    V6    item 6 - OAS shrinkage for the CSP covariance estimate.

    VALL  every change at once.

Grouping note: V2, V3 and V5 need a scorer object (to keep a history of
visited topologies and to compute the diversity term), so they use the *val*
scorer, which reproduces V0's in-sample stacking behaviour. That keeps them
one-change-at-a-time with respect to V0.
"""
from __future__ import annotations

import copy


# --------------------------------------------------------------------------- #
# Defaults: the published behaviour
# --------------------------------------------------------------------------- #
BASE = {
    "objective": "native",      # 'native' = V0 path | 'val' | 'oof'
    "oof_folds": 5,
    "stacking_cv": 0,           # 0/1 = in-sample (published), >1 = cross-validated
    "complexity_penalty": 0.0,
    "one_se_rule": False,
    "top_k_average": 1,
    "diversity_weight": 0.0,
    "csp_reg": None,            # None (published) | 'oas' | 'ledoit_wolf'
    "include_riemannian": False,        # band-passed tangent-space views
    "include_riemannian_exact": False,  # the paper's B5 itself, as a member
    "include_eegnet": False,            # the paper's B4 itself, as a member
    "include_fbcsp": False,
    "fbcsp_select_k": 8,
    # None = take it from config.yaml (same_family, the published setting).
    # NOTE: in dag_core, 'partial' and 'unconstrained' follow the same code
    # path -- _pick_family returns None for both -- which is why the published
    # ablation reports identical numbers for them.
    "member_constraint": None,
    # Seed the initial committee from this family ("STRONG" = the exact B5 and
    # EEGNet members). None = the published random initialisation.
    "init_family": None,
}


def _v(**kw):
    d = copy.deepcopy(BASE)
    d.update(kw)
    return d


VARIANTS = {
    "V0_published": _v(),

    "V1_oof_objective": _v(objective="oof", stacking_cv=5),

    "V2_regularised_selection": _v(objective="val", complexity_penalty=0.02,
                                   one_se_rule=True),

    "V3_topk_average": _v(objective="val", top_k_average=5),

    # item 4 proper: the strong baselines are reachable, not just cousins of
    # them. V4 embeds B5 exactly (unfiltered epochs, logistic-regression head)
    # alongside the band-passed tangent-space and FBCSP views.
    "V4_enriched_pool": _v(include_riemannian=True, include_fbcsp=True,
                           include_riemannian_exact=True),

    # B4 (EEGNet) as a member too. Kept separate because it costs one network
    # training per unit (and five more per unit when combined with the
    # out-of-fold objective), and because it is only clearly worth it on
    # Dataset 2a, where EEGNet leads by 6.4 points.
    "V7_strong_members": _v(include_riemannian_exact=True, include_eegnet=True,
                            include_fbcsp=True),

    # The same-family constraint makes the strong members effectively
    # unreachable: the family is drawn once at initialisation, so a committee
    # of strong baselines happens in only ~1 run in 5, and the STRONG family
    # holds fewer members than M, so get_random_distinct silently relaxes to
    # the whole pool. These twins lift the constraint, which is also what one
    # actually wants from a heterogeneous committee: strong baselines combined
    # WITH the CSP views. They differ from V0 in two respects (pool and
    # constraint) by design; V4/V7 above isolate the pool alone.
    "V4u_enriched_unconstrained": _v(include_riemannian=True,
                                     include_fbcsp=True,
                                     include_riemannian_exact=True,
                                     member_constraint="unconstrained"),

    "V7u_strong_unconstrained": _v(include_riemannian_exact=True,
                                   include_eegnet=True, include_fbcsp=True,
                                   member_constraint="unconstrained"),

    # The decisive test of item 4: start from a committee of the strong
    # baselines themselves and let the search improve on it. If DAG-SA cannot
    # beat its own members from that starting point, the question is settled.
    "V7w_strong_warmstart": _v(include_riemannian_exact=True,
                               include_eegnet=True, include_fbcsp=True,
                               member_constraint="unconstrained",
                               init_family="STRONG"),

    # Locked variant: the committee stays inside the STRONG family, so the
    # model is "EEGNet and the exact B5 baseline fused by the searched
    # operators". This asks whether fusing the strong baselines beats either
    # of them alone -- the cleanest possible reading of item 4.
    "V7l_strong_locked": _v(include_riemannian_exact=True,
                            include_eegnet=True, include_fbcsp=False,
                            member_constraint="same_family",
                            init_family="STRONG"),

    "V5_diversity": _v(objective="val", diversity_weight=0.10),

    "V6_shrinkage_csp": _v(csp_reg="oas"),

    "VALL_all_six": _v(objective="oof", stacking_cv=5,
                       complexity_penalty=0.02, one_se_rule=True,
                       top_k_average=5, diversity_weight=0.10,
                       csp_reg="oas", include_riemannian=True,
                       include_riemannian_exact=True, include_fbcsp=True),
}

DEFAULT_ORDER = ["V0_published", "V1_oof_objective", "V2_regularised_selection",
                 "V3_topk_average", "V4_enriched_pool", "V5_diversity",
                 "V6_shrinkage_csp", "VALL_all_six"]

# V7 is opt-in: it needs torch and costs an EEGNet training per unit.
ALL_ORDER = DEFAULT_ORDER[:-1] + ["V7_strong_members", "VALL_all_six"]


def pool_signature(v):
    """Variants with the same signature can share a fitted pool and the
    out-of-fold matrix, which is what keeps the study affordable."""
    return (v["csp_reg"], v["include_riemannian"],
            v["include_riemannian_exact"], v["include_eegnet"],
            v["include_fbcsp"],
            v["fbcsp_select_k"] if v["include_fbcsp"] else None)


def needs_oof(v):
    return v["objective"] == "oof"


def describe(name):
    return {
        "V0_published": "published method (validation objective, in-sample stacking)",
        "V1_oof_objective": "item 1: out-of-fold objective + cross-validated stacking",
        "V2_regularised_selection": "item 2: complexity penalty + one-standard-error rule",
        "V3_topk_average": "item 3: average of the 5 best topologies",
        "V4_enriched_pool": "item 4: tangent-space, FBCSP and the exact B5 baseline as members",
        "V7_strong_members": "item 4+: the B4 (EEGNet) and B5 baselines themselves as members",
        "V4u_enriched_unconstrained": "item 4 with the same-family constraint lifted",
        "V7u_strong_unconstrained": "item 4+ with the same-family constraint lifted",
        "V7w_strong_warmstart": "item 4+ started from a committee of the strong baselines",
        "V7l_strong_locked": "item 4+ locked to a committee of the strong baselines",
        "V5_diversity": "item 5: diversity-regularised objective",
        "V6_shrinkage_csp": "item 6: OAS shrinkage for the CSP covariance",
        "VALL_all_six": "all six changes together",
    }.get(name, name)
