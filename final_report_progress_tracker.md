# Final Report Progress Tracker

> Purpose: keep the final report writing aligned with the current diabetes prediction + FedAvg implementation, record what evidence already exists, and define the next smallest actions for the Day 1 first draft.
>
> Current working branch: `plans`
>
> Experiment evidence reviewed from branch: `pipeline_Version_bigDataset`

---

## 1. Locked Project Direction for Day 1

**Working title:** Federated Learning for Diabetes Prediction Using Clinical and Laboratory Features

**Main report story:** This project investigates whether a federated learning approach can train a diabetes prediction model across distributed clients while keeping data local. The current evidence should be framed as a feasibility and privacy-preserving comparison, not as a claim that FedAvg always outperforms centralized learning.

**Best current result direction:**
- Centralized learning is the strongest and most stable baseline so far.
- FedAvg reaches a best global accuracy close to centralized learning, but its final round is less stable.
- Non-IID client distribution likely explains why client performance differs strongly.
- The dataset is highly imbalanced, so accuracy alone is not enough for the final report.

**Claim to avoid:** Do not claim that federated learning is more accurate than centralized learning. A safer claim is that FedAvg achieves comparable accuracy in the best round while supporting a privacy-preserving training setup, but more metrics are required.

**Day 1 writing goal:** Complete a defensible first draft of Chapter 1, Chapter 3, Chapter 5, Chapter 6.2, Chapter 6.3, and a skeleton for Chapter 4.

---

## 2. Project Snapshot

**Student name:** TODO

**Student ID:** TODO

**Supervisor:** TODO

**Target submission date:** TODO

**Repository link:** https://github.com/Ada-lovelacee/Ada_Project

**Main code/results location:** `FL_diabetes_prediction_FedAvg/code/`

**Main dataset:** `diabetes_clean.csv`

**Dataset size:** 88,694 records

**Input feature count:** 14

**Label column:** `Diabetes_Type`

**Current label definition:** Binary diabetes prediction. The CSV currently encodes `0` and `1`; based on the project direction, use `0 = not diabetic` and `1 = diabetes/T2D candidate`, but verify this against the original data cleaning note before final submission.

**Class distribution:**

| Class | Meaning for draft | Count | Rate |
|---|---|---:|---:|
| 0 | Not diabetic | 83,270 | 93.88% |
| 1 | Diabetes / T2D candidate | 5,424 | 6.12% |

**Majority-class baseline accuracy:** 0.9388

**Feature groups:**
- Demographic: age
- Physical/clinical: BMI, waist circumference, pulse
- Laboratory: glucose, glycated hemoglobin, insulin, HDL, LDL, triglycerides, cholesterol, creatinine, eGFR, CRP

**Main model:** Simple MLP neural network

**Model architecture:** 14 input features -> hidden layer 16 -> hidden layer 16 -> 2 output classes

**Training framework:** PyTorch

**Federated learning framework:** Flower

**Federated strategy:** FedAvg

**Current FL setup:**
- 3 clients
- 20 federated rounds
- Dirichlet data partition with `alpha = 0.5`
- `fraction_fit = 0.5`
- `fraction_evaluate = 1.0`
- At least 2 clients train/evaluate per round
- Batch size 16
- Learning rate 0.001
- Local epochs 10
- Random seed 42

---

## 3. Current Result Summary

Use this table as the current Chapter 4 placeholder. Mark it as preliminary until Precision, Recall, F1, AUC, and confusion matrix are added.

| Setting | Model | Round/Epoch | Loss | Accuracy | Status | Notes |
|---|---|---:|---:|---:|---|---|
| Majority-class baseline | Always predict class 0 | N/A | N/A | 0.9388 | Calculated | Important because the dataset is imbalanced |
| Centralized | MLP | Epoch 10 | 0.1359 | 0.9564 | Completed | Best and final epoch are both epoch 10 |
| Federated FedAvg | MLP | Best round 8 | 0.1370 | 0.9554 | Completed | Close to centralized at best round |
| Federated FedAvg | MLP | Final round 20 | 0.2852 | 0.9432 | Completed | Final performance drops, likely due to non-IID instability |
| Client 0 in FL | MLP | Best round 3 | 0.0221 | 0.9968 | Logged | Very high client accuracy; likely easier/local distribution |
| Client 1 in FL | MLP | Best round 1 | 0.0966 | 0.9705 | Logged | Stable high accuracy |
| Client 2 in FL | MLP | Best round 11 | 0.3721 | 0.8436 | Logged | Much weaker client, important evidence of non-IID difficulty |

**Result interpretation for first draft:**
- Centralized training provides the upper reference point because all data is available in one place.
- FedAvg can approach centralized accuracy in the best round, which supports the feasibility of federated diabetes prediction.
- The final FedAvg round is lower than the best round, so the results section should discuss convergence instability.
- Client 2 performs much worse than Clients 0 and 1, which supports a non-IID client distribution discussion.
- Accuracy is limited because class 1 is only 6.12% of the dataset. The final report must add Recall, Precision, F1, AUC, and confusion matrix before making any clinical usefulness claim.

---

## 4. Core Experiment Checklist

### A. Dataset and Task Definition

- [x] Final working dataset confirmed for Day 1 draft: `diabetes_clean.csv`
- [x] Data size recorded: 88,694 rows
- [x] Feature count recorded: 14 features
- [x] Label column recorded: `Diabetes_Type`
- [x] Class imbalance recorded
- [ ] Original source and cleaning process written clearly
- [ ] Original meaning of label values verified from data cleaning notes
- [ ] Inclusion/exclusion logic recorded
- [ ] Feature list saved in methodology chapter
- [ ] Train/test split method made fully reproducible

**Notes for Chapter 3:** Say that the project uses a cleaned tabular clinical dataset containing demographic, physical, and laboratory indicators. Do not overclaim clinical deployment. Present it as an experimental diabetes prediction task.

### B. Preprocessing

- [x] Numerical standardization implemented with `StandardScaler`
- [x] Tensor conversion implemented for PyTorch training
- [x] Dirichlet client partition implemented
- [ ] Data leakage check completed
- [ ] Scaler fit only on training data
- [ ] Missing value strategy documented
- [ ] Any imputation logic documented
- [ ] Preprocessing pipeline described in final methodology chapter

**Important warning:** Current `load_csv_data()` fits `StandardScaler` before splitting or partitioning the data. For the final version, either fix this or write it as a limitation in the current experimental implementation. Best option: fix it before final results.

### C. Baselines

- [x] Centralized training run successfully
- [x] Federated training with FedAvg run successfully
- [ ] Local/isolated baseline run as a separate experiment
- [ ] All baselines evaluated with the same metrics
- [ ] Majority-class baseline included in result discussion

**Notes:** The current client-level logs are produced during FL and should not be described as a clean isolated local baseline. If time is short, write the local baseline as "planned/pending" and focus the first draft on centralized vs FedAvg.

### D. Results Collection

- [x] Accuracy recorded
- [x] Loss recorded
- [x] Centralized result table drafted
- [x] FedAvg result table drafted
- [x] Client result summary drafted
- [ ] Precision recorded
- [ ] Recall recorded
- [ ] F1-score recorded
- [ ] AUC recorded
- [ ] Confusion matrix generated
- [ ] ROC curve generated
- [ ] Training time recorded
- [ ] Communication cost/round count discussed

**Notes:** Accuracy alone is not enough because the majority-class baseline is already 0.9388.

### E. Optional Extension

- [ ] IID vs non-IID comparison
- [ ] Dirichlet alpha sensitivity: 1.0, 0.5, 0.1
- [ ] Client number comparison
- [ ] Local epoch comparison
- [ ] FedProx or other light extension

**Day 1 decision:** Do not start a new extension today. Write the first draft first.

---

## 5. Chapter-by-Chapter Day 1 Writing Plan

### Chapter 1. Introduction

- [ ] 1.1 Background of Study
- [ ] 1.2 Project Aim
- [ ] 1.3 Project Objectives
- [ ] 1.4 Project Overview
- [ ] 1.4.1 Scope
- [ ] 1.4.2 Audience
- [ ] 1.5 Structure of the Project Report

**Main content to write:**
- Diabetes prediction can support earlier risk identification using clinical and laboratory indicators.
- Traditional centralized machine learning requires data to be collected in one location.
- Healthcare data is sensitive, so data privacy and distributed data ownership are important.
- Federated learning allows clients to train local models and share model updates rather than raw records.
- This project implements and evaluates a simple MLP model for diabetes prediction under centralized and federated settings.

**Suggested aim:** The aim of this project is to design, implement, and evaluate a federated learning framework for diabetes prediction using tabular clinical data, and to compare its predictive performance with centralized learning.

**Suggested objectives:**
- Prepare a cleaned diabetes prediction dataset with demographic, physical, and laboratory features.
- Implement a centralized MLP baseline.
- Implement a Flower-based FedAvg training pipeline with simulated clients.
- Compare centralized and federated results using loss and accuracy first, then extend evaluation with Precision, Recall, F1-score, AUC, and confusion matrix.
- Discuss privacy, project management, limitations, and future improvements.

### Chapter 2. Literature Review

- [ ] 2.1 Fundamental Theories
- [ ] 2.2 Background Review of Proposed Systems
- [ ] 2.3 Gaps in the Existing Literature
- [ ] 2.4 Summary

**Day 1 action:** Create only the skeleton and collect paper targets. Full writing can move to Day 2.

**Paper/source buckets to collect with DeepResearch:**
- Federated averaging and the original FedAvg algorithm
- Federated learning in healthcare
- Privacy-preserving machine learning for medical data
- Machine learning for diabetes prediction
- Class imbalance in medical classification
- Evaluation metrics beyond accuracy in healthcare prediction

### Chapter 3. Methodology

- [ ] 3.1 Problem Statement
- [ ] 3.2 Approach
- [ ] Mathematical Basis
- [ ] Overview of Model Architecture
- [ ] Dataset and Data Processing
- [ ] Federated Learning Setup
- [ ] 3.3 Technology
- [ ] 3.4 Project Version Management Plan
- [ ] 3.5 Summary

**Must include today:**
- Binary classification task definition
- 14 input features and `Diabetes_Type` target
- MLP architecture
- Centralized training setup
- FedAvg setup with 3 clients and 20 rounds
- Dirichlet alpha 0.5 for non-IID client simulation
- Technology stack: Python, PyTorch, Flower, pandas, scikit-learn, matplotlib, Git/GitHub
- Git branch/version plan

**Methodology caution to write carefully:** Current preprocessing standardizes the full dataset before splitting. This should be fixed for final experiments or listed as a limitation of the preliminary run.

### Chapter 4. Implementation and Results

- [ ] 4.1 Performance Comparison of Proposed Framework with Baselines
- [ ] 4.2 Ablation Study Results / Extension Results
- [ ] 4.3 Efficiency Comparison Results
- [ ] 4.4 Summary

**Day 1 action:** Build the skeleton and paste the preliminary result table from Section 3.

**Do not overwrite today:** Chapter 4 should stay partially provisional until Recall, F1, AUC, confusion matrix, and possibly isolated local baseline are complete.

**Current figures already available in the pipeline branch:**
- `centralized_metrics.png`
- `server_global_metrics.png`
- `client_0_metrics.png`
- `client_1_metrics.png`
- `client_2_metrics.png`

### Chapter 5. Professional Issues

- [ ] 5.1 Project Management
- [ ] Activities
- [ ] Schedule
- [ ] Project Data Management
- [ ] Project Deliverables
- [ ] 5.2 Risk Analysis
- [ ] 5.3 Professional Issues
- [ ] Chapter summary

**Main content to write:**
- Project management uses staged development: dataset preparation, centralized baseline, federated implementation, result analysis, report writing.
- Version control uses GitHub branches: `main`, `plans`, `pipeline_Version_bigDataset`, and other experiment/archive branches.
- Data management must protect sensitive health-related records.
- Professional issues include privacy, reproducibility, model bias, class imbalance, and responsible interpretation.
- The system is an academic prototype, not a clinical decision system.

### Chapter 6. Conclusion

- [ ] 6.1 Findings and Conclusions
- [ ] 6.2 Limitations of the Project
- [ ] 6.3 Potential Future Work

**Write today: 6.2 and 6.3 first.**

**Limitations to include:**
- Accuracy is not sufficient because the positive diabetes class is small.
- Current evaluation does not yet include Recall, Precision, F1, AUC, ROC curve, or confusion matrix.
- Current preprocessing may need correction to avoid data leakage.
- Only one main model architecture has been tested.
- The current federated setting uses simulated clients rather than real healthcare institutions.
- Final FedAvg performance is less stable than the best round.

**Future work to include:**
- Add full classification metrics and threshold analysis.
- Compare IID and non-IID partitions.
- Test different Dirichlet alpha values and client counts.
- Add FedProx or another method designed for non-IID data.
- Improve class imbalance handling using class weights, resampling, or threshold tuning.
- Validate on another dataset or external cohort.

---

## 6. Tables and Figures to Produce

### Mandatory Tables

- [x] Table: preliminary result summary
- [x] Table: class distribution
- [ ] Table: dataset summary
- [ ] Table: feature/label summary
- [ ] Table: experiment settings
- [ ] Table: centralized vs federated vs local results
- [ ] Table: full metric comparison with Accuracy, Precision, Recall, F1, AUC
- [ ] Table: risk analysis/project management

### Mandatory Figures

- [ ] Figure: federated learning architecture
- [ ] Figure: training workflow
- [ ] Figure: centralized training loss/accuracy
- [ ] Figure: server global loss/accuracy
- [ ] Figure: client-level metric comparison
- [ ] Figure: confusion matrix
- [ ] Figure: ROC curve
- [ ] Figure: schedule/Gantt chart

### Optional Figures

- [ ] Figure: client data distribution
- [ ] Figure: IID vs non-IID illustration
- [ ] Figure: metric comparison bar chart
- [ ] Figure: alpha sensitivity comparison

---

## 7. Day 1 Execution Log

**Date:** 2026-04-14

**Main target:** Complete first draft material for the high-priority chapters.

- [x] Fix current project direction
- [x] Record dataset size and class distribution
- [x] Record main model and FedAvg setup
- [x] Record preliminary centralized and FedAvg results
- [ ] Draft Chapter 1
- [ ] Draft Chapter 3
- [ ] Draft Chapter 5
- [ ] Draft Chapter 6.2 and 6.3
- [ ] Create Chapter 4 result skeleton
- [ ] Push updated writing tracker to GitHub

**Done today:**
- Added and pushed the initial progress tracker.
- Reviewed current FedAvg pipeline evidence from `pipeline_Version_bigDataset`.
- Converted the tracker into a Day 1 writing guide.

**Problems found:**
- Current `plans` branch does not contain the full pipeline files; the full code/results are in `pipeline_Version_bigDataset`.
- Dataset is highly imbalanced.
- Current metrics are mainly loss and accuracy.
- Local/isolated baseline is not yet a separate clean experiment.
- Current preprocessing may fit the scaler before train/test splitting.

**Next action:** Start writing Chapter 1.1 and 1.2 using the project story in Section 1 and the aim/objectives in Section 5.

**What I must not spend time on today:**
- Switching dataset
- Starting a new FL algorithm
- Chasing small accuracy improvements
- Polishing grammar before the first draft exists
- Rebuilding the dashboard/frontend

---

## 8. AI-Assisted Writing Workflow

Use ChatGPT for drafting/editing and DeepResearch for source collection. Always paste the fixed facts from this tracker into the prompt so the tools do not invent project details.

### ChatGPT Prompt for Chapter 1

```text
I am writing the first draft of a final-year project report. The project is "Federated Learning for Diabetes Prediction Using Clinical and Laboratory Features".

Use these fixed project facts:
- Dataset: cleaned diabetes prediction CSV with 88,694 records.
- Target: binary `Diabetes_Type`, encoded as 0/1.
- Features: 14 demographic, physical, and laboratory features.
- Model: PyTorch MLP with 14 inputs, two hidden layers of 16 neurons, and 2 output classes.
- Federated framework: Flower FedAvg.
- FL setup: 3 clients, 20 rounds, Dirichlet alpha 0.5, local epochs 10.
- Main comparison: centralized MLP vs federated FedAvg MLP.
- Current results: centralized final accuracy 0.9564; FedAvg best global accuracy 0.9554 at round 8; FedAvg final accuracy 0.9432 at round 20.
- Important caution: dataset is imbalanced, with about 93.88% class 0 and 6.12% class 1, so accuracy alone is not enough.

Please draft Chapter 1 Introduction in academic report style with sections:
1.1 Background of Study
1.2 Project Aim
1.3 Project Objectives
1.4 Project Overview
1.4.1 Scope
1.4.2 Audience
1.5 Structure of the Project Report

Do not claim that federated learning outperforms centralized learning. Frame the project as a privacy-preserving feasibility comparison.
```

### ChatGPT Prompt for Chapter 3

```text
Using the same fixed project facts, draft Chapter 3 Methodology for an undergraduate final report. Include:
- Problem statement
- Dataset and task definition
- Preprocessing
- MLP architecture
- Centralized baseline
- Federated learning setup using Flower FedAvg
- Dirichlet non-IID client simulation
- Technology stack
- Version management plan

Mention that the current preprocessing and metrics are preliminary and will be strengthened with leakage checks, Precision, Recall, F1-score, AUC, and confusion matrix.
```

### ChatGPT Prompt for Chapter 5

```text
Draft Chapter 5 Professional Issues for this project. Focus on:
- Project management
- Schedule and deliverables
- Git/GitHub version control
- Data management
- Privacy and ethical handling of health-related data
- Reproducibility
- Risks from class imbalance and biased model interpretation
- Clear statement that this is an academic prototype, not a clinical decision system
```

### DeepResearch Prompt

```text
Find peer-reviewed and authoritative sources for a final-year project report on federated learning for diabetes prediction using tabular clinical data.

I need sources for:
1. Federated Averaging / FedAvg
2. Federated learning in healthcare
3. Privacy-preserving machine learning for medical data
4. Machine learning for diabetes prediction
5. Class imbalance and evaluation metrics in medical classification

For each source, provide:
- Full citation
- DOI or stable URL if available
- 2-3 sentence summary
- Which report section it supports: Introduction, Literature Review, Methodology, Results Discussion, or Professional Issues
```

### Rules for AI Use

- Always give the AI the fixed project facts from this tracker.
- Ask for academic writing, but verify all claims and citations.
- Do not allow the AI to invent results, dataset origin, or clinical claims.
- Use DeepResearch for sources, not for changing the experimental direction.
- Use ChatGPT to draft and polish text, then manually align it with the actual code/results.

---

## 9. Git Update Log

| Date | Commit / Branch | What changed | Next priority |
|---|---|---|---|
| 2026-04-14 | `894d544` on `plans` | Added initial final report progress tracker | Fill tracker using current FedAvg evidence |
| 2026-04-14 | Working tree | Revised tracker for Day 1 writing based on current pipeline results | Draft Chapter 1 and Chapter 3 |

---

## 10. Literature Review Tracking

| Status | Paper/Source Target | Topic | Use in report | Notes |
|---|---|---|---|---|
| [ ] | Original FedAvg paper | Federated Averaging | Methodology/Literature Review | Required |
| [ ] | FL in healthcare survey | Distributed health data | Introduction/Literature Review | Required |
| [ ] | Diabetes prediction ML paper | Tabular prediction | Literature Review | Required |
| [ ] | Medical class imbalance paper | Metrics and imbalance | Results Discussion | Important |
| [ ] | Privacy/ethics healthcare AI source | Professional issues | Chapter 5 | Important |
| [ ] | Flower framework documentation/paper | Implementation framework | Methodology | Useful |

Suggested use tags:
- `Introduction`
- `Literature Review`
- `Methodology`
- `Results Discussion`
- `Professional Issues`

---

## 11. Risk Log

| Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|
| Accuracy looks high because of class imbalance | High | High | Add Recall, Precision, F1, AUC, confusion matrix, and majority baseline | Open |
| Data leakage through preprocessing | Medium | High | Fit scaler on training data only or clearly document limitation | Open |
| Local baseline not clearly separated | Medium | Medium | Run a clean isolated-client baseline or remove the claim | Open |
| FedAvg final round lower than best round | Medium | Medium | Report both best and final round; discuss convergence instability | Open |
| Literature review too broad | High | Medium | Limit to diabetes prediction, FL, privacy, and evaluation metrics | Open |
| Writing takes too long | High | High | Draft core chapters first, polish later | Open |
| Pipeline code/results live on another branch | Medium | Medium | Merge/copy relevant results into final report branch or reference branch clearly | Open |

---

## 12. Ready-to-Submit Checklist

- [ ] Title matches final project direction
- [ ] Research aim is consistent with experiments
- [ ] Objectives match what was actually implemented
- [ ] Dataset source and cleaning process are described
- [ ] Label meaning is verified
- [ ] Methodology matches code implementation
- [ ] Preprocessing leakage issue is fixed or clearly acknowledged
- [ ] Result tables match CSV logs
- [ ] Figures are cited in text
- [ ] Accuracy is not used as the only metric
- [ ] Limitations are honest and specific
- [ ] Future work is realistic
- [ ] References are complete
- [ ] Appendix includes useful code/result supporting material
- [ ] Repository is backed up

---

## 13. Current Status Snapshot

**Overall completion:** `15%`

**Most urgent unfinished task:** Write the first draft of Chapter 1 and Chapter 3 using the current fixed project direction.

**Today's top 3 priorities:**
1. Draft Chapter 1 with a safe privacy-preserving feasibility story.
2. Draft Chapter 3 with exact dataset/model/FedAvg setup.
3. Create Chapter 4 skeleton with preliminary results and clear TODO metrics.

**Smallest next task:** Write 1.1 Background of Study in 500-700 words.

---

## 14. Evening Self-Check

Answer these briefly every evening.

- Did I finish the planned writing section?
- Did I avoid changing the project direction?
- Did I add citations only where sources are verified?
- Did I keep results consistent with the actual CSV logs?
- Did I push today's progress to GitHub?
