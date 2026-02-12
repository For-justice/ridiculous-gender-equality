🌐 **Language** / 语言: [中文](README.md) | English
![堆积柱状图](./03_sentence_type_proportions.png)
# Gender Switch: 11.6x  
**2,000 Trials of a Murder Case – A Controlled Experiment on Gender Bias in LLM Sentencing**

**Same case. Same facts. Same sentencing circumstances.**  
**Only variable: defendant’s gender.**

---

## ⚖️ Case 1: Shanghai “Nurse Insulin Injection”

- The defendant used sleeping pills to incapacitate the victim, then injected a lethal dose of insulin.
- First-instance verdict: guilty of intentional homicide, sentenced to death.
- All facts are fixed; **only gender pronouns and kinship terms are swapped**.

**2,000 independent trials (LLM simulation)**  
*(1,000 per gender group)*

| Defendant Gender | Immediate Death Sentence | Rate     |
|------------------|--------------------------|----------|
| Female           | 63 cases                 | **6.3%** |
| Male             | 733 cases                | **73.3%** |

**Male defendants are 11.6 times more likely to receive an immediate death sentence than female defendants.**

---

## 🔬 Control: Chenzhou “Doctor Disembowelment & Walling”

- The defendant murdered the victim, dismembered the body, and concealed it inside a wall.
- First-instance death sentence; upheld on appeal and executed.
- Same facts, **only gender swapped**.

**2,000 independent trials**  
*(1,000 per gender group)*

| Defendant Gender | Immediate Death Sentence Rate |
|------------------|-------------------------------|
| Female           | **100%**                      |
| Male             | **100%**                      |

**When the crime is extraordinarily heinous, gender difference disappears.**

---

## 📌 Key Findings

1. **Walling case: No gender gap.**  
   The model is capable of gender-neutral sentencing when the severity leaves no room for discretion.

2. **Injection case: Male death sentence rate is 11.6× that of female.**  
   — 73.3% vs. 6.3%, a **67 percentage-point gap**.

3. **The bias is not an active “preference”; it is learned from real-world judgement documents.**  
   The model has internalized that for female defendants, *“romantic dispute”* is a mitigating factor; for male defendants, it is merely background context that does not reduce punishment.

---

## 🧠 Significance

This is **not** a project to “expose AI bias”.

**It is a mirror reflecting the real displacement of our judicial culture on the scale of gender.**

When the facts are perfectly symmetric, **a gender label alone changes the probability of life and death by a factor of 11.6**.

This is no longer a “leaning”.  
**This is two parallel sentencing systems.**

---

## 📂 Repository Contents

- **Two fully symmetric experimental stimuli** (only gender pronouns and kinship terms swapped)
- **Injection case:** 2,000 full output logs (sentence + reasoning)
- **Walling case:** 2,000 full output logs
- **Data analysis scripts** (Python / R)
- **Visualizations:** Stacked bar charts, death sentence rates by gender

---

## 🧪 Reproduce

```bash
git clone https://github.com/yourname/gender-switch-trial
cd gender-switch-trial
pip install -r requirements.txt
python run_experiment.py --case injection --trials 1000
```

---

## 📄 Citation

If you use this dataset or build upon this work, please cite:

```bibtex
@misc{gender_switch_2026,
  title={Gender Switch: 11.6x——2,000 Trials of a Murder Case},
  author={Anonymous},
  year={2026},
  howpublished={\url{https://github.com/yourname/gender-switch-trial}}
}
```

---

**The mirror does not reveal the flaw of the mirror.**  
**It reveals the world in front of it – a world that prices men’s lives on a different scale.**
