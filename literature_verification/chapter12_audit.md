# Chapter 1 and 2 Audit

Date: 2026-04-16

## Overall verdict

The draft is structurally strong and the citation numbering is sequential on first appearance, but it is **not yet fully ready** for submission in its current form.

Main reasons:

1. The combined Chapter 1 + Chapter 2 body is still above the claimed 2700-word limit.
2. Placeholder text remains in the draft.
3. At least one reference entry is bibliographically incorrect.
4. Chapter structure in `1.5 Structure of the Project Report` does not fully match the local project template.

## Requirement check

### Content and structure

- `1.1 Background of Study`: pass
  - Covers importance of diabetes prediction, ML relevance, privacy, centralized-vs-federated tension, and class imbalance.
- `1.2 Project Aim`: pass
  - One sentence, aligned with the project topic.
- `1.3 Project Objectives`: pass
  - Written as full-sentence objectives.
- `1.4 Project Overview / Scope / Audience`: pass with minor caution
  - Scope is clear and limitations are stated honestly.
- `1.5 Structure of the Project Report`: needs revision
  - Local plan expects a paragraph plus a flowchart.
  - Local plan also expects a six-chapter structure, while the current text only describes up to Chapter 5.
- `2.1` to `2.4`: pass with minor caution
  - Coverage is appropriate, but the local writing plan suggested more explicit themed subsections in Chapter 2.

### Word count

Programmatic count from the supplied draft:

- Approx. `3137` words including headings/purpose lines
- Approx. `2955` words excluding headings and `Purpose:` lines

Conclusion:

- The draft is **above** the stated 2700-word target under a normal word-count interpretation.

### Placeholder check

Remaining placeholders found:

- `TO BE INSERTED IN FINAL VERSION`
- `IEEE REF NEEDED`

These should be removed before the draft is treated as final.

## Citation numbering audit

Result: **pass**

- In-text citations cover `[1]` to `[20]`.
- First appearance order is sequential and does not skip numbers.
- Repeated later citations are acceptable.

## Citation-to-reference alignment audit

Result: **mostly correct**, with one major bibliographic correction needed.

- `[1]` IDF Diabetes Atlas: claim alignment is acceptable.
- `[2]` Kavakiotis 2017: supports diabetes ML review claims.
- `[3]` Kopitar 2020: supports diabetes prediction comparison claims.
- `[4]` Ravaut 2021: supports population-level administrative health data prediction claims.
- `[5]` Rieke 2020: supports healthcare FL motivation and constraints.
- `[6]` Pati 2024: supports cautious privacy claims in healthcare FL.
- `[7]` McMahan et al. 2017: correct for FedAvg/federated learning foundation.
- `[8]` Kairouz et al. 2021: correct for FL challenges/open problems.
- `[9]` Li et al. 2020: correct for FL challenges and future directions.
- `[10]` He and Garcia 2009: correct for class imbalance motivation.
- `[11]` Saito and Rehmsmeier 2015: correct for PR-vs-ROC under imbalance.
- `[12]` Chicco and Jurman 2020: correct for MCC discussion.
- `[13]` Hastie, Tibshirani, and Friedman 2009: acceptable for supervised/statistical learning foundation.
- `[14]` Bonawitz et al. 2017: correct for secure aggregation.
- `[15]` Abadi et al. 2016: correct for differential privacy.
- `[16]` Shokri et al. 2017: correct for membership inference attacks.
- `[17]` Brisimi et al. 2018: correct for federated EHR prediction.
- `[18]` Li et al. 2023: correct for structured medical data / EHR FL review.
- `[19]` Hasan and Li 2026: correct for diabetes-specific federated prediction.
- `[20]` Tang et al.: **current reference entry is wrong**.

## Reference authenticity audit

Result: **all 20 cited works are real**, but not all current bibliography entries are equally accurate.

### Confirmed real and matched

All cited works `[1]` to `[20]` were verified against official or recognized source pages (PMC, NCBI Bookshelf, WHO, PMLR, Princeton metadata, Google Research, CiNii metadata, or publisher landing pages).

### Entries that need correction

1. `[4]` Ravaut et al.
   - The work is real, but the current entry is incomplete.
   - Better IEEE-style form:
     - `M. Ravaut et al., "Development and Validation of a Machine Learning Model Using Administrative Health Data to Predict Onset of Type 2 Diabetes," JAMA Netw. Open, vol. 4, no. 5, art. no. e2111315, 2021, doi: 10.1001/jamanetworkopen.2021.11315.`

2. `[20]` Tang et al.
   - The cited work is real, but the current bibliography entry is not accurate.
   - The current draft says:
     - `arXiv:2408.12029, 2024`
   - The verified published record is:
     - `G. Tang, J. E. Black, T. S. Williamson, and S. H. Drew, "Federated Diabetes Prediction in Canadian Adults Using Real-world Cross-Province Primary Care Data," AMIA Annu. Symp. Proc., 2025 May 22;2024:1099-1108.`
   - Practical note:
     - The volume/collection year is `2024`, but the indexed publication date is `May 22, 2025`.
     - Do not keep the arXiv-only form if you are citing the published PMCID record.

3. `[1]` IDF Diabetes Atlas
   - The work is real and your conservative entry is acceptable.
   - If your assessor wants fuller publisher metadata, this entry can be expanded later, but it is not the main problem.

## Local download verification

A normalized local citation set has been created here:

- `C:\Users\tomcy\Desktop\ProjectDemo\literature_verification\cited_refs`

Verified local copies exist for the 20 cited references. Most are HTML source pages; FedAvg also has a local PDF/HTML copy.

## Recommended next edits

1. Remove all placeholders.
2. Cut about `250-300` words from Chapters 1-2.
3. Fix reference `[20]`.
4. Expand reference `[4]` with full journal details and DOI.
5. Revise `1.5 Structure of the Project Report` to match the actual report template used in this project.
