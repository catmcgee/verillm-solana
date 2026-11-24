"""The honest run and the attacks of Appendix A, each as a population and the role draw it needs."""

from __future__ import annotations

from .protocol import Scenario

PROMPT = "In two sentences, why does it matter that a model's weights are open?"
TOKENS = 96
# A coalition holding the first four of the sixteen nodes in every group.
COALITION = {stage: {index: "forged-output" for index in range(4)} for stage in range(1, 5)}


def all_scenarios() -> list[Scenario]:
    return [
        Scenario(
            id="honest",
            title="Honest run",
            paper="Section 5",
            summary=(
                "Every node follows the protocol. The verifiers' recomputed hidden states match the inferencers' within "
                "floating-point noise, and every segment is accepted."
            ),
            prompt=PROMPT,
            max_new_tokens=TOKENS,
            adversaries={},
            wanted=lambda roles: True,
        ),
        Scenario(
            id="quantized",
            title="Quantized model",
            paper="Appendix A.1",
            summary=(
                "The node drawn to run segment 2 serves an 8-bit version of its layers while registered as full precision. "
                "Verifiers recompute that segment from its recorded inputs and the hidden states do not match."
            ),
            prompt=PROMPT,
            max_new_tokens=TOKENS,
            adversaries={2: {4: "quantized"}},
            wanted=lambda roles: roles[1]["inferencer"] == 4,
        ),
        Scenario(
            id="early-termination",
            title="Early termination",
            paper="Appendix A.2",
            summary=(
                "The node running the last segment stops the answer early by reporting an end-of-sequence token that its own hidden state does not produce."
            ),
            prompt=PROMPT,
            max_new_tokens=TOKENS,
            adversaries={4: {4: "early-termination"}},
            wanted=lambda roles: roles[3]["inferencer"] == 4,
            options={"stop_step": 12},
        ),
        Scenario(
            id="forged-output",
            title="Forged output, caught",
            paper="Appendix A.3",
            summary=(
                "A coalition holding a quarter of each group draws all four inference roles. It writes the answer with a much "
                "smaller model, then runs the real model once over that text so the hidden states look right. The last "
                "segment's verifiers decode the hidden states and find tokens the real model would not have chosen."
            ),
            prompt=PROMPT,
            max_new_tokens=TOKENS,
            adversaries=COALITION,
            wanted=lambda roles: all(role["inferencer"] < 4 for role in roles) and sum(1 for v in roles[3]["verifiers"] if v < 4) < 2,
            options={"forge": True},
        ),
        Scenario(
            id="forged-accepted",
            title="Forged output, accepted",
            paper="Appendix A.3, Section 5.7",
            summary=(
                "The same forgery, on a draw where the coalition also holds two of the three verifier seats on the last "
                "segment. The forged answer is accepted and the one honest verifier, who found the wrong tokens, is penalized. "
                "The paper's dispute step is open only to a rejected inferencer, so nothing in what we built reverses this."
            ),
            prompt=PROMPT,
            max_new_tokens=TOKENS,
            adversaries=COALITION,
            wanted=lambda roles: all(role["inferencer"] < 4 for role in roles) and sum(1 for v in roles[3]["verifiers"] if v < 4) >= 2,
            options={"forge": True},
        ),
        Scenario(
            id="lazy-verifier",
            title="Lazy verifier",
            paper="Appendix A.4",
            summary="A verifier on segment 3 skips the recomputation, commits to states it did not compute and votes to accept, hoping to be paid for nothing.",
            prompt=PROMPT,
            max_new_tokens=TOKENS,
            adversaries={3: {4: "lazy-verifier"}},
            wanted=lambda roles: 4 in roles[2]["verifiers"],
        ),
        Scenario(
            id="dispute",
            title="Captured committee and dispute",
            paper="Section 5.7, Appendix E",
            summary=(
                "Two colluding verifiers are both drawn onto the three-member committee for segment 2 and vote down honest work "
                "with matching fabricated states. The inferencer disputes, and a larger second committee overturns the verdict."
            ),
            prompt=PROMPT,
            max_new_tokens=TOKENS,
            adversaries={2: {4: "colluding-verifier", 9: "colluding-verifier"}},
            wanted=lambda roles: {4, 9} <= set(roles[1]["verifiers"]),
        ),
    ]
