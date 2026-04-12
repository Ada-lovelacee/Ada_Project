# 10-Day Execution Plan for Diabetes Federated Learning Thesis

> Version: v1.0  
> Purpose: This document is the **hard execution plan** for finishing **all experiments + thesis writing within 10 days**.  
> Rule: **No delay. Only finish early.**  
> Update this file on GitHub every day after completing the daily tasks.

---

## 1. Project Positioning

### 1.1 Core strategy
Use the uploaded stroke federated learning paper as the **structural reference paper**, but replace the disease domain with **diabetes** and replace its hospital EHR setting with **our diabetes dataset + Flower/Python experiments**. The thesis should therefore keep a clear, proven research story:

1. medical prediction is useful;
2. privacy makes centralized aggregation difficult;
3. federated learning is a reasonable solution;
4. diabetes is used as the target disease instead of stroke;
5. experiments compare **local / centralized / federated** settings;
6. results are interpreted from both **performance** and **privacy-preserving** perspectives.

### 1.2 Working thesis title
You can use one of these as a working title for the report and repository:

- **Privacy-Preserving Federated Learning for Diabetes Risk Prediction**
- **A Federated Prediction Model for Diabetes Risk Assessment under Privacy Constraints**
- **Design and Evaluation of a Privacy-Preserving Federated Learning Framework for Diabetes Prediction**

Recommended default:

**Design and Evaluation of a Privacy-Preserving Federated Learning Framework for Diabetes Prediction**

### 1.3 Fixed thesis story
This report is **not** trying to claim that federated learning always beats centralized learning.

The thesis story should be:

- centralized learning is the upper-bound style reference when data can be pooled;
- local single-site training is the realistic weak baseline under isolated data;
- federated learning aims to approach centralized performance **without raw data sharing**;
- if federated learning is close to centralized and better than isolated local models, the project is successful.

That story is academically safer and easier to defend.

---

## 2. How the Uploaded Paper Should Be Used

The stroke paper gives a very useful logic template:

- **Abstract**: background -> privacy problem -> federated solution -> experiment summary -> practical value
- **Introduction**: why disease prediction matters, why privacy matters, why FL is suitable
- **Related Work**: prior AI in healthcare + prior FL in healthcare
- **Methodology**: preprocessing, missing-value handling, feature selection, classifiers, federated averaging, system workflow
- **Experimental Analysis**: compare local, federated, centralized results, explain why results differ
- **Discussion**: social value, privacy significance, deployment meaning

Our diabetes thesis should borrow this **logic**, not copy wording.

---

## 3. How to Fill the University Template Chapter by Chapter

## Front Matter

### Title Page
Fill in:
- final title
- student name / ID / major
- supervisor
- date submitted

### Acknowledgement
Keep it short. Thank supervisor, dataset authors if appropriate, and anyone who helped with environment or feedback.

### Abstract (<= 250 words)
Write this **last**, not first.
Required structure:
1. one sentence on diabetes as an important health problem;
2. one sentence on privacy/data-isolation challenge;
3. one sentence on the proposed Flower-based federated learning framework;
4. one sentence on datasets/models/baselines;
5. one sentence on main result trend;
6. one sentence on practical value and limitation.

### Keywords
Use 5-6 only. Recommended:
- Federated Learning
- Diabetes Prediction
- Privacy Preservation
- Machine Learning
- Healthcare AI
- Flower Framework

### Abbreviations
Must include at least:
- FL — Federated Learning
- EHR — Electronic Health Record
- LR — Logistic Regression
- RF — Random Forest
- XGBoost — Extreme Gradient Boosting
- NN — Neural Network
- AUC — Area Under the Curve
- ACC — Accuracy
- F1 — F1-score
- IID / non-IID — Independent and Identically Distributed / Non-Independent and Identically Distributed

### Glossary
Only keep thesis-specific terms, for example:
- federated averaging
- client
- server aggregation
- local model
- centralized baseline
- privacy-preserving training
- ablation study

---

## Chapter 1. Introduction

### 1.1 Background of Study
Write:
- diabetes is a major chronic disease with early screening value;
- machine learning can support diabetes risk prediction;
- medical/health data are privacy-sensitive and often siloed;
- centralized training is effective but not always feasible;
- federated learning allows collaborative training without raw-data sharing.

This section should answer: **why this topic matters**.

### 1.2 Project Aim
Use **one sentence only**.
Recommended wording:

> The aim of this project is to design, implement, and evaluate a privacy-preserving federated learning framework for diabetes prediction, and to compare its performance against local and centralized baselines.

### 1.3 Project Objectives
Write in full sentences, not fragments. Recommended 5 objectives:
1. To review literature on diabetes prediction and federated learning in healthcare.
2. To prepare and preprocess a diabetes dataset suitable for centralized and federated experiments.
3. To implement baseline machine learning models in Python.
4. To implement a Flower-based federated learning pipeline and evaluate its performance under multiple settings.
5. To analyse the results in terms of predictive performance, efficiency, and practical privacy-preserving value.

### 1.4 Project Overview
Short summary of the whole project:
- data -> preprocessing -> model building -> Flower FL simulation -> comparison -> discussion.

#### 1.4.1 Scope
State clearly what is included and excluded.

Included:
- diabetes prediction task;
- structured/tabular medical-style dataset;
- centralized baseline;
- federated baseline using Flower;
- evaluation with standard metrics;
- limited ablation on FL settings.

Excluded:
- real hospital deployment;
- cryptographic secure aggregation implementation from scratch;
- large-scale production system;
- real-time clinical validation.

#### 1.4.2 Audience
Possible audience:
- researchers interested in privacy-preserving healthcare AI;
- students learning federated learning;
- practitioners exploring distributed training for health prediction;
- supervisors/examiners evaluating research-oriented ML work.

### 1.5 Structure of the Project Report
Use one paragraph + one flowchart.
Suggested chapter summary:
- Chapter 1 introduces the problem, aim, objectives, scope, and report structure.
- Chapter 2 reviews diabetes prediction, healthcare AI, and federated learning literature.
- Chapter 3 presents the problem statement, dataset, preprocessing, model design, and experimental setup.
- Chapter 4 reports implementation details and experimental results.
- Chapter 5 discusses project management, risks, and professional issues.
- Chapter 6 concludes the project and outlines limitations and future work.

---

## Chapter 2. Literature Review

### Writing principle
Do **not** make this chapter a pile of summaries. Build it around a question:

> Why is federated learning a reasonable approach for diabetes prediction under privacy constraints?

### 2.1 Fundamental Theories
Recommended subsections:
- Diabetes prediction and risk screening
- Supervised learning for tabular healthcare data
- Federated learning and FedAvg
- Data privacy in healthcare AI
- Evaluation metrics for imbalanced medical prediction

### 2.2 Background Review of Proposed Systems
Recommended themed structure:

#### 2.2.1 Traditional centralized diabetes prediction systems
Review models such as logistic regression, random forest, XGBoost, MLP.
Focus on:
- common datasets
- common features
- common metrics
- strengths and weaknesses

#### 2.2.2 Federated learning in healthcare prediction
Review FL healthcare papers.
Focus on:
- why FL is used
- typical system architecture
- trade-off between privacy and performance
- data heterogeneity issue

#### 2.2.3 Flower or practical FL frameworks
Briefly review Flower as an implementation framework.
Explain why it is suitable for student experimentation.

### 2.3 Gaps in the Existing Literature
This is critical.
Suggested gap statement:
- many diabetes prediction studies assume centralized access to data;
- many FL healthcare papers focus on concept demonstration rather than a clear student-reproducible workflow;
- few small-scale educational implementations explain how to compare local, centralized, and Flower-based federated settings in one consistent pipeline;
- therefore, this project provides a practical comparative implementation for diabetes prediction under privacy constraints.

### 2.4 Summary
One short paragraph:
- literature supports diabetes prediction with ML;
- privacy remains a challenge;
- FL is promising;
- gap motivates this project.

---

## Chapter 3. Methodology

This chapter should be the **most concrete chapter**.

### 3.1 Problem Statement
Clearly state:
- raw medical-style data are privacy-sensitive and may be separated across institutions/clients;
- centralized collection may be restricted;
- isolated local training may underuse distributed data;
- therefore, a federated framework is needed for diabetes prediction.

### 3.2 Approach
This section should mirror the strong logic of the stroke paper but be adapted to your real setup.

#### 3.2.1 Mathematical Basis
Keep it simple and defendable.
Write:
- classification setting;
- binary output (diabetes / no diabetes) or risk label, depending on your dataset;
- loss function such as binary cross-entropy or cross-entropy;
- FedAvg aggregation formula;
- evaluation metrics.

Minimum equations to include:
1. model prediction objective;
2. loss function;
3. FedAvg aggregation equation.

#### 3.2.2 Overview of Model Architecture
State exactly what you actually run.
Recommended model set:
- Logistic Regression
- Random Forest or XGBoost as strong tabular baseline
- MLP / simple neural network

If time becomes tight, keep **one centralized baseline + one federated NN pipeline + one traditional ML baseline**.

#### 3.2.3 Dataset and Data Processing
Must include:
- dataset name and source;
- number of samples and features;
- label definition;
- missing-value handling;
- encoding and normalization;
- train/validation/test split;
- how data are partitioned into clients for FL;
- IID and, if possible, non-IID setup.

#### 3.2.4 Experimental Design
Add this subsection even if the template does not force it.
State:
- local baseline setting;
- centralized setting;
- federated setting;
- number of clients;
- rounds;
- local epochs;
- batch size;
- learning rate;
- hardware/software environment.

### 3.3 Technology
List the actual tools:
- Python
- Flower
- PyTorch and/or scikit-learn
- pandas, NumPy, matplotlib
- AutoDL or local GPU environment
- GitHub for version control

### 3.4 Project Version Management Plan
Write this like evidence of disciplined work:
- GitHub repository used for source code and report markdown drafts;
- folders for code, logs, figures, report, and references;
- commit daily with meaningful messages;
- keep experiment logs versioned;
- save plotted figures with fixed filenames used in report.

### 3.5 Summary
Summarize dataset, models, FL setup, and evaluation design.

---

## Chapter 4. Implementation and Results

This chapter should follow a clean comparison logic.

### 4.0 Recommended actual subsection structure
Use this version instead of keeping the template wording too generic:

- 4.1 Implementation Overview
- 4.2 Baseline Results: Local vs Centralized
- 4.3 Federated Learning Results
- 4.4 Ablation Study
- 4.5 Efficiency and Training Cost Analysis
- 4.6 Optional System / Demo / Functionality Test
- 4.7 Summary

### 4.1 Implementation Overview
Write:
- code modules;
- preprocessing pipeline;
- model training scripts;
- Flower client/server workflow;
- figure of experiment pipeline.

### 4.2 Performance Comparison of Proposed Framework with Baselines
This is the most important results section.
Minimum comparison table:
- Local model A
- Local model B
- Centralized model
- Federated model

Metrics to report depending on your task:
- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Loss

Interpretation rule:
- do not just say “higher is better”;
- explain why federated may be close to centralized and better than local;
- explain any unstable result using sample size, partitioning, or heterogeneity.

### 4.3 Ablation Study Results
Pick only small, necessary ablations. Recommended:
- number of clients
- number of rounds
- learning rate
- IID vs non-IID split

If time is limited, do only **two ablations**:
1. IID vs non-IID
2. 10 rounds vs 30 rounds vs 50 rounds

### 4.4 Efficiency Comparison Results
Report:
- training time
- communication rounds
- convergence trend
- hardware setting

Possible figures:
- round vs accuracy
- round vs loss
- time vs final performance

### 4.5 If there is a web/demo component
Only include if real and finished.
Possible content:
- input form screenshot
- prediction output screenshot
- simple functional test table

If the web part is unfinished, do **not** let it dominate the thesis.
The thesis should still pass based on the research pipeline.

### 4.6 Summary
State the final message of the chapter:
- centralized gives reference performance;
- federated achieves competitive performance while preserving data locality;
- isolated local training is weaker;
- ablations show how FL settings affect results.

---

## Chapter 5. Professional Issues

This chapter is often rushed, but it is easy marks if done properly.

### 5.1 Project Management
Include:
- activities table
- schedule / mini Gantt chart
- what was planned vs completed
- repository and experiment log management

### 5.2 Risk Analysis
Use a table with:
- risk
- impact
- likelihood
- mitigation
- current status

Suggested risks:
- training environment failure
- poor model performance
- dataset quality problems
- delayed writing
- figure/table inconsistency
- reference formatting issues

### 5.3 Professional Issues
Discuss under these headings:
- Legal: privacy, consent, sensitive health data
- Ethical: bias, fairness, misuse of AI prediction, over-reliance by non-clinicians
- Social: value of early screening, accessibility, trust in AI
- Professional: transparent reporting, reproducibility, academic integrity, BCS/ACM responsibilities
- Environmental: compute resource consumption, efficient experimentation

### 5.4 Summary
One paragraph only.

---

## Chapter 6. Conclusion

### 6.1 Findings and Conclusions
State clearly what was achieved:
- a Flower-based FL pipeline was implemented for diabetes prediction;
- local, centralized, and federated settings were compared;
- FL preserved data locality while retaining competitive predictive performance.

### 6.2 Limitations of the Project
Be honest and controlled:
- simulated clients rather than real hospitals;
- limited dataset size;
- limited model diversity;
- no advanced privacy mechanism such as secure aggregation or DP;
- limited clinical validation.

### 6.3 Potential Future Work
Write realistic extensions:
- larger multi-institution datasets;
- more advanced FL algorithms;
- non-IID robustness improvement;
- secure aggregation / differential privacy;
- deployment into a real health-screening system.

---

## 4. Ten-Day Hard Schedule

## Non-negotiable rule
Each day ends with:
1. code pushed;
2. logs saved;
3. figures exported;
4. report text updated;
5. this markdown plan checked off.

If a task finishes early, move immediately to the next day’s task.

---

## Day 1 — Freeze topic, dataset, repo structure, chapter skeleton

### Must finish
- Finalize thesis title.
- Finalize dataset choice.
- Create GitHub repository structure.
- Create report skeleton in Word/Markdown.
- Write Chapter 1 draft first version.
- Write Chapter 2 subsection headings.
- Write Chapter 3 subsection headings.
- Define final experiment matrix.

### GitHub deliverables
- `/README.md`
- `/report/outline.md`
- `/report/chapter1_intro_draft.md`
- `/experiments/experiment_plan.md`
- `/refs/reference_list.bib` or `/refs/reference_notes.md`

### Done definition
By the end of Day 1, you must be able to answer in one sentence:
**What dataset, what models, what FL setup, what comparisons?**

---

## Day 2 — Preprocessing pipeline and centralized baseline

### Must finish
- Load dataset successfully.
- Clean missing values / encode / normalize.
- Build reproducible train/val/test split.
- Run at least one centralized baseline end-to-end.
- Save metrics and confusion matrix / ROC if appropriate.
- Write Methodology draft for dataset and preprocessing.

### GitHub deliverables
- `/src/preprocess.py`
- `/src/centralized_baseline.py`
- `/logs/centralized/*.log`
- `/results/centralized/*.csv`
- `/figures/centralized/*.png`
- `/report/chapter3_methodology_draft.md`

### Done definition
Centralized baseline runs successfully from command line and outputs saved metrics.

---

## Day 3 — Local baseline(s) and comparison table v1

### Must finish
- Simulate client partitions.
- Train local model on each client / partition.
- Collect average local performance.
- Build first comparison table: Local vs Centralized.
- Write Chapter 4 section draft for baseline results.

### GitHub deliverables
- `/src/local_baseline.py`
- `/results/local/*.csv`
- `/figures/local/*.png`
- `/report/chapter4_results_v1.md`

### Done definition
You have a real table showing centralized is reference and local is weaker or different.

---

## Day 4 — Flower federated pipeline runs successfully

### Must finish
- Implement Flower client/server pipeline.
- Complete at least one successful federated run.
- Save round-wise metrics.
- Export training curve figure.
- Record all config values.

### GitHub deliverables
- `/src/flwr_client.py`
- `/src/flwr_server.py`
- `/src/task.py` or equivalent
- `/configs/fl_config.yaml` or `.json`
- `/logs/federated/run1.log`
- `/results/federated/run1_metrics.csv`

### Done definition
Federated training completes without crashing and produces interpretable logs.

---

## Day 5 — Federated tuning and main result selection

### Must finish
- Run the main federated experiment you will report.
- Tune only the essential hyperparameters.
- Choose the **final main result run**.
- Produce polished comparison table: Local vs Centralized vs Federated.
- Draft Chapter 4.2 / 4.3 text.

### GitHub deliverables
- `/logs/federated/main_run.log`
- `/results/main_comparison_table.csv`
- `/figures/main_accuracy_curve.png`
- `/figures/main_loss_curve.png`
- `/report/chapter4_main_results.md`

### Done definition
You now have the **core thesis evidence**.

---

## Day 6 — Ablation study and efficiency analysis

### Must finish
- Run 2 small ablations only.
- Run efficiency/time comparison.
- Make ablation table and one figure.
- Write Chapter 4.4 and 4.5.

### Recommended ablation priority
1. IID vs non-IID
2. rounds = 10 / 30 / 50
3. learning rate comparison

### GitHub deliverables
- `/results/ablation/*.csv`
- `/figures/ablation/*.png`
- `/report/chapter4_ablation_efficiency.md`

### Done definition
All necessary experiments are finished by the end of Day 6.

---

## Day 7 — Literature review finalization and references

### Must finish
- Finalize Chapter 2 in paragraph form.
- Organize references in IEEE style.
- Add all citations into Chapters 1–4.
- Build literature comparison table if useful.

### GitHub deliverables
- `/report/chapter2_literature_final.md`
- `/refs/final_references.bib` or `/refs/final_references.md`

### Done definition
The report is no longer missing literature support.

---

## Day 8 — Professional issues + conclusion + abstract

### Must finish
- Write Chapter 5 completely.
- Write Chapter 6 completely.
- Write Abstract.
- Write Acknowledgement.
- Fill Abbreviations and Glossary.

### GitHub deliverables
- `/report/chapter5_professional_issues.md`
- `/report/chapter6_conclusion.md`
- `/report/abstract.md`

### Done definition
All chapters exist in full draft form by the end of Day 8.

---

## Day 9 — Merge into template and fix formatting

### Must finish
- Move all text into the official Word template.
- Insert figures and tables with correct numbering.
- Update table of contents.
- Check formatting rules: Arial 11, spacing, captions, chapter page starts.
- Check chapter numbering and cross-references.

### GitHub deliverables
- `/report/final_report_v1.docx`
- `/report/final_report_v1.pdf`
- `/report/figure_inventory.md`

### Done definition
A complete submission-ready version exists by the end of Day 9.

---

## Day 10 — Final review, polishing, anti-error pass

### Must finish
- Full proofreading.
- Remove template red guidance text.
- Verify all figures/tables are referenced in text.
- Verify all citations exist in references.
- Check grammar, consistency, and page numbering.
- Freeze final version.
- Submit early if possible.

### GitHub deliverables
- `/report/final_report_submitted_version.docx`
- `/report/final_report_submitted_version.pdf`
- `/report/submission_checklist.md`

### Done definition
The report is fully finished and can be submitted immediately.

---

## 5. Minimum Experiment Matrix

To avoid overexpansion, keep the experiment matrix small and sufficient.

### Mandatory experiments
1. Centralized baseline
2. Local isolated baseline
3. Federated main run
4. One ablation on data partition or rounds
5. One efficiency/time comparison

### Good enough final set
- Centralized NN
- Federated NN (IID)
- Federated NN (non-IID)
- Local NN
- Optional traditional ML baseline: Logistic Regression or XGBoost

### Do not do
- too many models with weak analysis
- too many rounds of hyperparameter search
- advanced privacy mechanisms that you cannot finish
- unnecessary web features before the report is stable

---

## 6. Recommended Repository Structure

```text
project-root/
├─ README.md
├─ data/
├─ src/
│  ├─ preprocess.py
│  ├─ centralized_baseline.py
│  ├─ local_baseline.py
│  ├─ flwr_client.py
│  ├─ flwr_server.py
│  └─ utils.py
├─ configs/
├─ logs/
│  ├─ centralized/
│  ├─ local/
│  └─ federated/
├─ results/
│  ├─ centralized/
│  ├─ local/
│  ├─ federated/
│  └─ ablation/
├─ figures/
├─ report/
│  ├─ outline.md
│  ├─ chapter1_intro_draft.md
│  ├─ chapter2_literature_final.md
│  ├─ chapter3_methodology_draft.md
│  ├─ chapter4_results_v1.md
│  ├─ chapter5_professional_issues.md
│  ├─ chapter6_conclusion.md
│  └─ final_report_v1.docx
└─ refs/
```

---

## 7. Daily Commit Rule

Commit at least 3 times per day:

1. `morning-plan`
2. `experiment-update`
3. `writing-update`

Recommended commit style:
- `feat: add centralized diabetes baseline`
- `exp: finish federated iid 30-round run`
- `write: draft chapter 3 methodology`
- `fix: update figure numbering and table captions`

---

## 8. Hard Anti-Delay Rules

### Rule 1
No rewriting old paragraphs for perfection before the whole draft exists.

### Rule 2
No adding new experiments after Day 6 unless a core run failed.

### Rule 3
No web/demo expansion before all report chapters are drafted.

### Rule 4
If one experiment fails repeatedly for more than half a day, downgrade scope immediately and keep the thesis core intact.

### Rule 5
The thesis passes by **clear comparison + sound methodology + honest discussion**, not by having the most complicated system.

---

## 9. Final Writing Order

Do the writing in this order, not template order:

1. Chapter 3 Methodology
2. Chapter 4 Results
3. Chapter 1 Introduction
4. Chapter 2 Literature Review
5. Chapter 5 Professional Issues
6. Chapter 6 Conclusion
7. Abstract
8. Acknowledgement / Abbreviations / Glossary

This order is faster and safer.

---

## 10. Submission Checklist

- [ ] Title finalized
- [ ] Dataset finalized
- [ ] Centralized baseline finished
- [ ] Local baseline finished
- [ ] Federated main run finished
- [ ] Ablation finished
- [ ] Efficiency comparison finished
- [ ] All figures exported in high quality
- [ ] All tables inserted and numbered
- [ ] Chapters 1-6 complete
- [ ] Abstract under 250 words
- [ ] References in IEEE style
- [ ] Table of contents updated
- [ ] Formatting matches template
- [ ] Red guidance text removed
- [ ] Final PDF exported
- [ ] Submission-ready version archived

---

## 11. Today’s Immediate First Actions

If starting **right now**, do these in order:

1. Create the GitHub repository folders.
2. Decide the final diabetes dataset.
3. Write the one-sentence aim and five objectives.
4. Create the experiment plan table.
5. Run the centralized baseline first.
6. Start Chapter 3 before trying to perfect Chapter 2.

---

## 12. One-Sentence Reminder

**Your job in the next 10 days is not to build the biggest system. Your job is to finish a complete, defensible, well-structured diabetes federated learning thesis early.**
