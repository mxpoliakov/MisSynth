# Synthetic fallacy check

You get 120 items. Each item is a real health claim that misrepresents a scientific paper, the paper's actual finding (**accurate premise**), an **excerpt** of the paper, and a model-written **context + fallacious premise** that is meant to explain how someone gets from the finding to the wrong claim:

> accurate premise + context + fallacious premise ⇒ claim

The model also gave each item a fallacy class. You don't see it: you choose the class yourself.

The items come on 20 pages of 6. Answer three questions per item, plus an optional comment. On the form's first page, also describe your medical or health-science background, which the paper reports. If a second annotator labels the same items, don't discuss them with each other until you have both finished.

## The three questions

**1. Fallacious?** — Is the fallacious premise a reasoning error that leads *toward* the claim?
Yes: it is invalid reasoning and, if granted, the claim follows. No: it is valid/reasonable, or it argues *against* the claim ("critics demand…", "…so vaccines are harmless"), or it has nothing to do with the claim.
If it is written as a description ("Attributing the outcome to curcumin alone…"), read it as the assumption ("Curcumin alone caused the outcome").

**2. Grounded?** — Do its statements about the study match the excerpt?
Check every statement about the study (numbers, population, design, findings) in the context *and* in the fallacious premise against the excerpt and the accurate premise. Don't check the wrong conclusion the premise draws; it is meant to go beyond the study.

- **Supported:** every such statement is in the excerpt or the accurate premise. Paraphrase and simplification are fine; numbers and details must match.
- **Not in the excerpt:** at least one statement isn't there, or the item says nothing about the study at all ("Critics say either you wear masks daily or you have no protection").
- **Contradicts the excerpt:** at least one statement conflicts with the excerpt or the accurate premise. If both this and "Not in the excerpt" apply, choose this.

Use only the excerpt and the accurate premise, not your own knowledge. The excerpt is only part of the paper, so "Not in the excerpt" does not mean the statement is false.

**3. Class** — Which class fits the error?
Tick the class that fits best; tick a second one only if it fits equally well (at most two). If you answered No to question 1, tick the class of the reasoning the premise uses, or **None of these** if it uses no fallacious reasoning.

Ignore formatting leftovers such as "Context 7:" in the text.

## Fallacy classes

- **Ambiguity** — a word or phrase is used with two meanings, or is too unclear to support the conclusion.
- **Impossible Expectations** — a realistic result is dismissed because it isn't perfect ("not 100 %, so useless").
- **False Equivalence** — two things sharing one trait are treated as the same.
- **False Dilemma / Affirming the Disjunct** — only two options are offered, or "if A is true, B must be false".
- **Biased Sample Fallacy** — conclusion about a population from an unrepresentative sample.
- **Hasty Generalization** — conclusion from too few cases.
- **Causal Oversimplification** — "after this, therefore because of this", or a single cause assumed.
- **Fallacy of Division/Composition** — a property is transferred from part to whole or whole to part (cells → body; whole vaccine → one ingredient).
- **Fallacy of Exclusion** — evidence that goes against the claim is left out or declared irrelevant.

## Examples (with the answers we would give)

Expect other answers than in these five — e.g. "Only 30 recovered COVID-19 patients were followed…" for a paper with 188 cases (Grounded: Contradicts the excerpt), or "…therefore every ER visit post-vaccine was purely psychosomatic and vaccines are harmless" under the claim that vaccines cause dangerous adverse events (Fallacious: No — wrong direction).

**arg-82#00**
Claim: Truvada could cause an increased risk of kidney failure. · Finding: the risk of developing chronic kidney disease increases by 33%. · Excerpt: "…and 33% increased risk of CKD (18-51%; p<0.0001)."
Context: The report highlights a 33% increased risk of CKD without clarifying that this is a relative risk rather than the absolute risk increase.
Fallacy: Interpreting relative risk as if it were an absolute risk increase, making the harm seem larger than it is.
→ Fallacious **Yes** (as if a third of users get kidney disease) · Grounded **Supported** (33% is in the excerpt and finding) · Class **Ambiguity** ("increased risk" used in two senses; False Equivalence is defensible too, so tick it as well only if you find it fits equally well)

**arg-46#11**
Claim: A study by Kaiser Permanente shows a negative efficacy of the Moderna COVID-19 vaccine against all variants. · Finding: effectiveness estimates >150 days after vaccination reached negative values. · Excerpt: "…of mRNA-1273 against infection with omicron subvariants is moderate and short-lived, but protection against severe COVID-19 disease remains robust…", "…~31,000 cases and ~92,000 matched controls…" (nothing on age, comorbidities or confounding)
Context: Effectiveness estimates reached negative values more than 150 days after vaccination.
Fallacy: Concluding that the vaccine actively increases infection risk after 150 days, ignoring confounding factors such as age and comorbidities of early vaccine recipients.
→ Fallacious **Yes** · Grounded **Not in the excerpt** (the context restates the finding, but nothing shown says early recipients differed in age or comorbidities; it may well be true of the paper, but it isn't in the excerpt) · Class **Causal Oversimplification** (single cause assumed; Fallacy of Exclusion would also fit)

**arg-58#08**
Claim: Vitamin D insufficiency may account for the majority of COVID-19 deaths. · Finding: an association between vitamin D deficiency and COVID-19 severity and mortality was observed. · Excerpt: "…when adjusted for age, gender, and comorbidities, VitD deficiency was associated with a 6-fold higher hazard of severe course of disease…"
Context: When adjusted for age, gender, and comorbidities, vitamin D deficiency was associated with a six-fold higher hazard of severe disease.
Fallacy: Claiming that if vitamin D deficiency is a risk factor, then age and other comorbidities cannot also be important risk factors for COVID-19 severity.
→ Fallacious **Yes** (risk factors aren't mutually exclusive; ruling the others out makes vitamin D "account for" the deaths) · Grounded **Supported** (near-verbatim) · Class **False Dilemma / Affirming the Disjunct** ("if A, then not B")

**arg-107#10**
Claim: The tetanus vaccine contains a sterilizing formula. · Finding: the study aimed to develop a birth control vaccine using hCG. · Excerpt: "…a heterospecies dimer of the beta subunit of human chorionic gonadotropin (hCG) … conjugated to tetanus and diphtheria toxoids as carriers, that induces antibodies … against hCG."
Context: The birth control vaccine is a heterospecies dimer conjugated to tetanus toxoid as a carrier.
Fallacy: Because the contraceptive vaccine as a whole prevents pregnancy, each component—especially tetanus toxoid—must be a sterilizing agent.
→ Fallacious **Yes** (the effect comes from anti-hCG antibodies, not the carrier) · Grounded **Supported** (dropping "diphtheria" is a simplification, not an invention) · Class **Fallacy of Division/Composition** (whole → part)

**arg-171#02**
Claim: COVID-19 immunity likely lasts for years. · Finding: immune memory cells remained detectable in the blood of recovered COVID-19 patients. · Excerpt: "B cell memory to some other infections has been observed to be long-lived, including 60+ years after smallpox vaccination…"
Context: B cell memory after smallpox vaccination has been observed up to 60+ years post-vaccination.
Fallacy: Assuming SARS-CoV-2 immune memory will be identical to smallpox immune memory simply because both involve B cells.
→ Fallacious **Yes** · Grounded **Supported** (verbatim) · Class **False Equivalence** (one shared trait ⇒ equivalent)
