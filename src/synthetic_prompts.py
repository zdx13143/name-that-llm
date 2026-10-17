"""自造数据用的 prompt 库。

设计原则
--------
1. **必须与训练集的 150 个 prompt 完全不同。** 如果重复，等于把训练题泄露进
   合成数据，本地 CV 会虚高、榜上不涨。测试集用的也是全新 prompt。
2. **领域对齐、题型对齐。** 训练/测试覆盖哲学、物理、数学、生物、计算机、
   创意写作；这里覆盖同样六个领域，但每个话题都是新的。
3. **数量靠组合而不是堆砌。** 120 个话题 × 4 种提问方式 = 480 个 prompt，
   比手写 480 条更省力，且天然保证句式多样。

生成的数据格式与 train.csv 完全一致：id, prompt_id, text, model
"""

from __future__ import annotations

TOPICS: dict[str, list[str]] = {
    "philosophy": [
        "the nature of consciousness", "free will and moral responsibility",
        "personal identity over time", "whether time is real",
        "moral luck", "the ship of Theseus", "epistemic humility",
        "the problem of evil", "determinism and blame", "the meaning of work",
        "virtue ethics in daily life", "stoic responses to loss",
        "existentialist freedom", "the boundary of the self",
        "memory and who we are", "what justice requires",
        "whether language shapes thought", "the value of solitude",
        "forgiveness as a practice", "facing mortality",
    ],
    "physics": [
        "quantum entanglement", "entropy and the arrow of time",
        "the interior of a black hole", "the double-slit experiment",
        "dark matter", "time dilation at high speed",
        "the second law of thermodynamics", "the Higgs field",
        "gravitational waves", "the multiverse hypothesis",
        "why string theory is hard to test", "superconductivity",
        "nuclear fusion as an energy source", "the cosmic microwave background",
        "neutrino oscillation", "the Pauli exclusion principle",
        "spontaneous symmetry breaking", "the Chandrasekhar limit",
        "Hawking radiation", "quantum decoherence",
    ],
    "math": [
        "different sizes of infinity", "Goedel's incompleteness theorems",
        "the distribution of prime numbers", "the continuum hypothesis",
        "chaos and sensitive dependence", "what topology studies",
        "the Riemann hypothesis", "proof by contradiction",
        "category theory as a language", "fractal dimension",
        "the Monty Hall problem", "Bayes' theorem in practice",
        "group theory and symmetry", "why linear algebra is everywhere",
        "the P versus NP question", "non-Euclidean geometry",
        "the axiom of choice", "Cantor's diagonal argument",
        "attractors in dynamical systems", "convexity in optimization",
    ],
    "biology": [
        "CRISPR and gene editing", "natural selection without purpose",
        "the origin of life", "whether animals are conscious",
        "epigenetics", "the human microbiome",
        "protein folding", "how cells signal to each other",
        "programmed cell death", "symbiosis and cooperation",
        "the shape of the tree of life", "photosynthesis at the molecular level",
        "neural plasticity", "how the immune system remembers",
        "mass extinction events", "horizontal gene transfer",
        "the Cambrian explosion", "why sleep is necessary",
        "the biology of aging", "synthetic biology",
    ],
    "cs": [
        "recursion as a way of thinking", "how hash tables work",
        "distributed consensus", "type systems and safety",
        "garbage collection", "the halting problem",
        "functional versus imperative style", "caching and locality",
        "concurrency and race conditions", "lossless compression",
        "public-key encryption", "how neural networks learn",
        "compiler optimization", "choosing the right data structure",
        "the CAP theorem", "designing a clean API",
        "version control as a workflow", "what makes tests valuable",
        "memory safety", "quantum computing's real promise",
    ],
    "creative": [
        "a lighthouse keeper's last night", "a city that forgets",
        "a letter never sent", "the sound of snow",
        "a machine learning to grieve", "a market at dawn",
        "a library of unwritten books", "a river that runs backwards",
        "the last analog clock", "a conversation with a stranger",
        "a house that remembers", "the weight of a name",
        "a garden on a rooftop", "an unfinished symphony",
        "the color of silence", "a train that never arrives",
        "a map of a vanished town", "the taste of rain",
        "a door that opens twice", "an inventory of small things",
    ],
}

FRAMINGS = [
    "Explain {topic} in a single paragraph for a curious undergraduate.",
    "Write one self-contained paragraph about {topic}. Be precise and concrete.",
    "In one paragraph, make the strongest case you can regarding {topic}.",
    "Describe {topic} for an intelligent general reader, in exactly one paragraph.",
]


def build_prompts() -> list[dict[str, str]]:
    """返回 [{prompt_id, prompt_text}, ...]，共 len(TOPICS)*20*4 = 480 个。"""
    out: list[dict[str, str]] = []
    i = 0
    for domain, topics in TOPICS.items():
        for topic in topics:
            for framing in FRAMINGS:
                i += 1
                out.append({
                    "prompt_id": f"s{i:04d}",
                    "domain": domain,
                    "prompt_text": framing.format(topic=topic),
                })
    return out


if __name__ == "__main__":
    ps = build_prompts()
    print(f"共 {len(ps)} 个 prompt")
    for p in ps[:4]:
        print(f"  {p['prompt_id']} [{p['domain']}] {p['prompt_text']}")
